#!/usr/bin/env python3
"""Plan B: DPPO — Diffusion Policy Policy Optimization.

Treats the DDIM denoising chain as a multi-step MDP and applies PPO
directly on the denoising process.

Key idea (Allen Ren et al., CoRL 2024 / RSS 2025):
- State_k = (x_k, obs)             where x_k is noisy action at step k
- Action_k = ε_θ(x_k, k, obs)      denoiser output
- Transition: x_{k-1} = DDIM(x_k, ε_θ, k)
- Reward: R_0 = env reward when x_0 is executed; R_k = 0 for k > 0

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_rl/train_dppo.py \\
        --policy_A_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out runs/rl_dppo_A \\
        --num_envs 16 --total_env_steps 500000 --headless

    # Resume
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_rl/train_dppo.py \\
        --policy_A_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out runs/rl_dppo_A \\
        --resume --headless
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Denoising Value Network
# ============================================================================
class DenoisingValueNetwork(nn.Module):
    """Estimates V(x_k, obs, k) for the denoising MDP.

    Takes the same inputs as the diffusion denoiser but outputs a scalar
    value instead of noise prediction.
    """

    def __init__(
        self,
        obs_dim: int = 15,
        action_chunk_dim: int = 128,   # action_dim * horizon (8 * 16)
        denoising_steps: int = 10,
        hidden_dims: list[int] = (512, 512, 256),
        vision_dim: int = 256,
    ):
        super().__init__()

        # Simple vision encoder (lighter than actor's for efficiency)
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.vision_features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.vision_pool = nn.AdaptiveAvgPool2d(1)
        self.vision_fc = nn.Linear(512, vision_dim)

        # Wrist encoder (share architecture)
        base_w = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.wrist_features = nn.Sequential(
            base_w.conv1, base_w.bn1, base_w.relu, base_w.maxpool,
            base_w.layer1, base_w.layer2, base_w.layer3, base_w.layer4,
        )
        self.wrist_pool = nn.AdaptiveAvgPool2d(1)
        self.wrist_fc = nn.Linear(512, vision_dim)

        # Timestep embedding
        self.step_embed = nn.Embedding(denoising_steps + 1, 64)

        # MLP head
        input_dim = vision_dim * 2 + obs_dim + action_chunk_dim + 64  # vis + state + x_k + step_embed
        layers = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU(inplace=True))
            d_in = d_out
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def encode_vision(self, table_img: torch.Tensor, wrist_img: torch.Tensor) -> torch.Tensor:
        t = self.vision_pool(self.vision_features(table_img)).flatten(1)
        t = self.vision_fc(t)
        w = self.wrist_pool(self.wrist_features(wrist_img)).flatten(1)
        w = self.wrist_fc(w)
        return torch.cat([t, w], dim=-1)

    def forward(
        self,
        table_img: torch.Tensor,    # (B, 3, H, W)
        wrist_img: torch.Tensor,    # (B, 3, H, W)
        state: torch.Tensor,        # (B, obs_dim)
        x_k: torch.Tensor,          # (B, action_chunk_dim) — flattened noisy action
        k: torch.Tensor,            # (B,) — denoising step index (int)
    ) -> torch.Tensor:
        vis = self.encode_vision(table_img, wrist_img)
        step_emb = self.step_embed(k)
        x = torch.cat([vis, state, x_k, step_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


# ============================================================================
# GAE computation
# ============================================================================
def compute_gae(
    rewards: torch.Tensor,     # (T,)
    values: torch.Tensor,      # (T,)
    next_value: float,
    dones: torch.Tensor,       # (T,)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Returns:
        (advantages, returns) — both (T,)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ============================================================================
# Denoising chain rollout
# ============================================================================
class DenoisingRolloutBuffer:
    """Stores denoising MDP trajectories for PPO updates."""

    def __init__(self):
        self.obs_table_imgs = []
        self.obs_wrist_imgs = []
        self.obs_states = []
        self.x_ks = []           # noisy action at step k
        self.k_steps = []        # denoising step index
        self.log_probs = []      # log prob of denoiser output
        self.rewards = []        # reward (only non-zero at k=0)
        self.values = []         # V(x_k, obs, k)
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(
        self,
        table_img, wrist_img, state, x_k, k_step,
        log_prob, reward, value, done,
    ):
        self.obs_table_imgs.append(table_img)
        self.obs_wrist_imgs.append(wrist_img)
        self.obs_states.append(state)
        self.x_ks.append(x_k)
        self.k_steps.append(k_step)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def finalize(self, next_value: float, gamma: float, lam: float):
        """Compute advantages and returns."""
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.stack(self.dones)
        self.advantages, self.returns = compute_gae(
            rewards, values, next_value, dones, gamma, lam,
        )
        # Normalize advantages
        if len(self.advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size: int, device: str):
        """Yield mini-batches for PPO updates."""
        n = len(self.rewards)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            yield {
                "table_img": torch.stack([self.obs_table_imgs[i] for i in idx]).to(device),
                "wrist_img": torch.stack([self.obs_wrist_imgs[i] for i in idx]).to(device),
                "state": torch.stack([self.obs_states[i] for i in idx]).to(device),
                "x_k": torch.stack([self.x_ks[i] for i in idx]).to(device),
                "k_step": torch.stack([self.k_steps[i] for i in idx]).to(device),
                "log_prob_old": torch.stack([self.log_probs[i] for i in idx]).to(device),
                "advantage": self.advantages[idx].to(device),
                "return_": self.returns[idx].to(device),
            }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ============================================================================
# DDIM denoising chain with log_prob tracking
# ============================================================================
def run_ddim_chain_with_logprob(
    policy,
    obs_dict: dict,
    num_inference_steps: int = 10,
    action_dim: int = 8,
    horizon: int = 16,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Run DDIM denoising chain and record per-step info.

    Returns:
        (clean_action, chain_info)
        clean_action: (B, horizon, action_dim)
        chain_info: list of (x_k, epsilon_pred, log_prob) per step
    """
    B = obs_dict["observation.state"].shape[0]

    # Access the diffusion model internals
    diffusion = policy.diffusion
    noise_scheduler = diffusion.noise_scheduler

    # Prepare observation encoding
    # We need to manually encode observations as the policy does
    n_obs_steps = policy.config.n_obs_steps

    # Build batch for diffusion
    batch = {}
    if "observation.state" in obs_dict:
        state = obs_dict["observation.state"]
        if state.dim() == 2:
            state = state.unsqueeze(1).expand(-1, n_obs_steps, -1)
        batch["observation.state"] = state

    # Stack image features
    image_keys = [k for k in obs_dict if "image" in k]
    image_list = []
    for key in sorted(image_keys):
        img = obs_dict[key]
        if img.dim() == 4:
            img = img.unsqueeze(1).expand(-1, n_obs_steps, -1, -1, -1)
        image_list.append(img)
    if image_list:
        batch["observation.images"] = torch.stack(image_list, dim=2)

    # Encode observations through the diffusion model's encoder
    with torch.no_grad():
        # Get global conditioning from observation encoder
        obs_enc = diffusion.obs_encoding(batch)  # (B, obs_enc_dim)

    # Start from pure noise
    x_k = torch.randn(B, horizon, action_dim, device=device)

    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps)

    chain_info = []

    for i, t_val in enumerate(noise_scheduler.timesteps):
        t = t_val.expand(B).to(device)

        # Predict noise — this goes through the denoiser (UNet1D)
        # Enable gradient for policy update
        eps_pred = diffusion.unet(x_k, t, global_cond=obs_enc)

        # Compute log probability (treating prediction as mean of unit-var Gaussian)
        # log p(ε_pred | x_k, obs, k) ∝ -0.5 * ||ε_pred||^2  (standard normal prior)
        log_prob = -0.5 * eps_pred.flatten(1).pow(2).sum(dim=-1)

        chain_info.append((
            x_k.detach().clone(),
            eps_pred.detach().clone(),
            log_prob.detach().clone(),
        ))

        # DDIM step: x_{k-1} = f(x_k, eps_pred, t)
        with torch.no_grad():
            scheduler_output = noise_scheduler.step(eps_pred, t[0].item(), x_k)
            x_k = scheduler_output.prev_sample

    clean_action = x_k  # (B, horizon, action_dim)
    return clean_action, chain_info


# ============================================================================
# Argument parser
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPPO: Diffusion Policy Policy Optimization (Plan B)")

    # Policy checkpoint
    p.add_argument("--policy_A_ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    # Environment
    p.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=1500,
                   help="Max env steps per episode.")
    p.add_argument("--image_width", type=int, default=128)
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    p.add_argument("--distance_threshold", type=float, default=0.03)
    p.add_argument("--reward_type", type=str, default="dense",
                   choices=["dense", "sparse", "distance"])

    # PPO hyperparams
    p.add_argument("--total_env_steps", type=int, default=500000)
    p.add_argument("--rollout_steps", type=int, default=512,
                   help="Env steps per rollout collection phase.")
    p.add_argument("--ppo_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--entropy_coeff", type=float, default=0.01)
    p.add_argument("--value_coeff", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--lr_policy", type=float, default=1e-5)
    p.add_argument("--lr_value", type=float, default=3e-4)

    # KL regularization to BC prior
    p.add_argument("--kl_weight_start", type=float, default=1.0)
    p.add_argument("--kl_weight_end", type=float, default=0.01)

    # Action
    p.add_argument("--n_action_steps_execute", type=int, default=8)
    p.add_argument("--n_action_steps_predict", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=10,
                   help="DDIM denoising steps (K in the paper).")

    # Logging
    p.add_argument("--log_freq", type=int, default=1)
    p.add_argument("--eval_freq", type=int, default=20)
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="rev2fwd-rl-dppo")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    return args


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        _run_training(args)
    finally:
        simulation_app.close()


def _run_training(args: argparse.Namespace) -> None:
    from scripts.scripts_rl.rl_env_wrapper import PickPlaceRLEnv, load_pretrained_diffusion_policy
    from scripts.scripts_rl.utils import (
        set_seed, save_metrics, linear_schedule,
    )

    set_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), dir=str(out_dir))

    # ---- Environment ----
    print("Creating RL environment...")
    env = PickPlaceRLEnv(
        task_id=args.task,
        num_envs=args.num_envs,
        device=device,
        image_width=args.image_width,
        image_height=args.image_height,
        goal_xy=tuple(args.goal_xy),
        distance_threshold=args.distance_threshold,
        horizon=args.horizon,
        reward_type=args.reward_type,
        headless=getattr(args, "headless", True),
    )

    # ---- Load pretrained policy ----
    print(f"Loading pretrained policy from {args.policy_A_ckpt}...")
    policy, preprocessor, postprocessor, config_info = load_pretrained_diffusion_policy(
        checkpoint_dir=args.policy_A_ckpt,
        device=device,
        n_action_steps=args.n_action_steps_execute,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    policy.train()

    # Frozen reference for KL regularization
    policy_ref = deepcopy(policy)
    policy_ref.eval()
    for p in policy_ref.parameters():
        p.requires_grad_(False)

    # ---- Value network ----
    state_dim = config_info.get("state_dim", 15)
    action_dim = config_info.get("action_dim", 8)
    action_chunk_dim = action_dim * args.n_action_steps_predict

    value_net = DenoisingValueNetwork(
        obs_dim=state_dim,
        action_chunk_dim=action_chunk_dim,
        denoising_steps=args.num_inference_steps,
    ).to(device)

    # ---- Optimizers ----
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr_policy)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr_value)

    # ---- Resume ----
    start_step = 0
    if args.resume:
        ckpt_path = out_dir / "latest_checkpoint.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            policy.load_state_dict(state["policy_state_dict"])
            value_net.load_state_dict(state["value_net_state_dict"])
            policy_optimizer.load_state_dict(state["policy_optimizer"])
            value_optimizer.load_state_dict(state["value_optimizer"])
            start_step = state["env_step"]
            print(f"Resumed from step {start_step}")

    # ---- Training loop ----
    env_step = start_step
    iteration = 0
    K = args.num_inference_steps  # denoising steps

    obs = env.reset()
    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_successes = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    print(f"\n{'='*60}")
    print(f"DPPO Training")
    print(f"  Total env steps: {args.total_env_steps}")
    print(f"  Num envs:        {args.num_envs}")
    print(f"  Denoising steps: {K}")
    print(f"  Rollout steps:   {args.rollout_steps}")
    print(f"  PPO epochs:      {args.ppo_epochs}")
    print(f"{'='*60}\n")

    while env_step < args.total_env_steps:
        iteration += 1
        iter_start = time.time()

        # ====== Phase 1: Collect rollouts ======
        rollout_buffer = DenoisingRolloutBuffer()
        collect_rewards = []
        collect_successes = []

        policy.eval()
        for env_t in range(0, args.rollout_steps, args.n_action_steps_execute):
            # Prepare observation for policy
            obs_processed = {}
            for k_name, v in obs.items():
                obs_processed[k_name] = v.clone()
            if preprocessor is not None:
                obs_processed = preprocessor(obs_processed)

            # Run DDIM chain with log_prob tracking (per-env, sequentially)
            # For efficiency, batch across envs
            with torch.no_grad():
                clean_actions, chain_infos = run_ddim_chain_with_logprob(
                    policy, obs_processed,
                    num_inference_steps=K,
                    action_dim=action_dim,
                    horizon=args.n_action_steps_predict,
                    device=device,
                )

            # Postprocess actions
            if postprocessor is not None:
                # Reshape for postprocessor: (B, horizon, action_dim) → process per step
                clean_actions_post = clean_actions.clone()
                for i in range(clean_actions_post.shape[1]):
                    clean_actions_post[:, i] = postprocessor(clean_actions[:, i])
            else:
                clean_actions_post = clean_actions

            # Execute action chunk in environment
            chunk_reward = torch.zeros(args.num_envs, device=device)
            chunk_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

            for step_in_chunk in range(args.n_action_steps_execute):
                if step_in_chunk >= clean_actions_post.shape[1]:
                    break
                action_t = clean_actions_post[:, step_in_chunk]
                next_obs, reward, terminated, truncated, info = env.step(action_t)

                chunk_reward += reward * (args.gamma ** step_in_chunk)
                chunk_done |= terminated | truncated
                episode_rewards += reward
                episode_successes |= info["success"]

                obs = next_obs

            # Handle episode resets
            done_envs = chunk_done.nonzero(as_tuple=True)[0]
            if len(done_envs) > 0:
                for idx in done_envs:
                    collect_rewards.append(episode_rewards[idx].item())
                    collect_successes.append(episode_successes[idx].item())
                episode_rewards[done_envs] = 0
                episode_successes[done_envs] = False
                obs = env.reset(env_ids=done_envs)

            # Store denoising MDP transitions into rollout buffer
            # Each denoising step k produces a transition
            for env_idx in range(args.num_envs):
                for k_idx, (x_k, eps_pred, log_prob) in enumerate(chain_infos):
                    # Compute value estimate
                    with torch.no_grad():
                        val = value_net(
                            obs_processed["observation.image"][env_idx:env_idx+1],
                            obs_processed.get("observation.wrist_image",
                                              obs_processed["observation.image"])[env_idx:env_idx+1],
                            obs_processed["observation.state"][env_idx:env_idx+1],
                            x_k[env_idx:env_idx+1].flatten(1),
                            torch.tensor([k_idx], device=device),
                        )

                    # Reward: only at the last denoising step (k=0 → index K-1)
                    r = chunk_reward[env_idx] if k_idx == len(chain_infos) - 1 else 0.0
                    if isinstance(r, torch.Tensor):
                        r = r.clone()
                    else:
                        r = torch.tensor(r, device=device)

                    d = chunk_done[env_idx].clone() if k_idx == len(chain_infos) - 1 else torch.tensor(False, device=device)

                    rollout_buffer.add(
                        table_img=obs_processed["observation.image"][env_idx].detach().cpu(),
                        wrist_img=obs_processed.get("observation.wrist_image",
                                                     obs_processed["observation.image"])[env_idx].detach().cpu(),
                        state=obs_processed["observation.state"][env_idx].detach().cpu(),
                        x_k=x_k[env_idx].flatten().detach().cpu(),
                        k_step=torch.tensor(k_idx, dtype=torch.long),
                        log_prob=log_prob[env_idx].detach().cpu(),
                        reward=r.cpu(),
                        value=val[0].detach().cpu(),
                        done=d.float().cpu(),
                    )

            env_step += args.num_envs * args.n_action_steps_execute

        # Finalize rollout buffer (compute GAE)
        rollout_buffer.finalize(next_value=0.0, gamma=args.gamma, lam=args.gae_lambda)

        # ====== Phase 2: PPO update ======
        policy.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_loss = 0
        n_updates = 0

        kl_weight = linear_schedule(
            args.kl_weight_start, args.kl_weight_end,
            env_step, args.total_env_steps,
        )

        for epoch in range(args.ppo_epochs):
            for batch in rollout_buffer.get_batches(args.batch_size, device):
                # ---- Policy loss (clipped PPO) ----
                # Re-evaluate log_prob under current policy
                # Run denoiser on stored x_k at stored step k
                k_step = batch["k_step"]
                x_k = batch["x_k"].view(-1, args.n_action_steps_predict, action_dim)

                noise_scheduler = policy.diffusion.noise_scheduler
                noise_scheduler.set_timesteps(K)

                # Get timestep for this denoising step
                # k_step is the index into scheduler.timesteps
                t_vals = noise_scheduler.timesteps.to(device)

                # Encode observations
                n_obs_steps = policy.config.n_obs_steps
                inf_batch = {}
                state = batch["state"]
                if state.dim() == 2:
                    state = state.unsqueeze(1).expand(-1, n_obs_steps, -1)
                inf_batch["observation.state"] = state

                imgs = [batch["table_img"], batch["wrist_img"]]
                img_stack = []
                for img in imgs:
                    if img.dim() == 4:
                        img = img.unsqueeze(1).expand(-1, n_obs_steps, -1, -1, -1)
                    img_stack.append(img)
                inf_batch["observation.images"] = torch.stack(img_stack, dim=2)

                obs_enc = policy.diffusion.obs_encoding(inf_batch)

                # Compute noise prediction for each sample
                # Map k_step indices to actual timestep values
                t = t_vals[k_step.clamp(0, len(t_vals) - 1)]
                t = t.to(device)

                eps_new = policy.diffusion.unet(x_k, t, global_cond=obs_enc)
                log_prob_new = -0.5 * eps_new.flatten(1).pow(2).sum(dim=-1)

                # KL regularization: compare with reference policy
                with torch.no_grad():
                    eps_ref = policy_ref.diffusion.unet(x_k, t, global_cond=obs_enc)
                kl_loss = F.mse_loss(eps_new, eps_ref.detach())

                # PPO clipped objective
                ratio = torch.exp(log_prob_new - batch["log_prob_old"])
                adv = batch["advantage"]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value loss ----
                v_pred = value_net(
                    batch["table_img"], batch["wrist_img"],
                    batch["state"],
                    batch["x_k"],
                    batch["k_step"],
                )
                value_loss = F.mse_loss(v_pred, batch["return_"])

                # ---- Total loss ----
                loss = policy_loss + args.value_coeff * value_loss + kl_weight * kl_loss

                # Update policy
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
                policy_optimizer.step()
                value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl_loss += kl_loss.item()
                n_updates += 1

        iter_time = time.time() - iter_start

        # ====== Logging ======
        if iteration % args.log_freq == 0 and n_updates > 0:
            avg_policy = total_policy_loss / n_updates
            avg_value = total_value_loss / n_updates
            avg_kl = total_kl_loss / n_updates
            ep_reward = np.mean(collect_rewards) if collect_rewards else 0
            success_rate = np.mean(collect_successes) if collect_successes else 0

            print(f"[Iter {iteration}] step={env_step}/{args.total_env_steps} "
                  f"pi={avg_policy:.4f} v={avg_value:.4f} kl={avg_kl:.4f} "
                  f"kl_w={kl_weight:.3f} "
                  f"ep_rew={ep_reward:.2f} success={success_rate:.2%} "
                  f"buf={len(rollout_buffer)} t={iter_time:.1f}s")

            metrics = {
                "env_step": env_step,
                "iteration": iteration,
                "policy_loss": avg_policy,
                "value_loss": avg_value,
                "kl_loss": avg_kl,
                "kl_weight": kl_weight,
                "episode_reward": ep_reward,
                "success_rate": success_rate,
            }
            save_metrics(out_dir / "metrics.jsonl", metrics)

            if args.wandb:
                import wandb
                wandb.log(metrics)

        # ====== Save checkpoint ======
        if iteration % args.save_freq == 0:
            ckpt = {
                "policy_state_dict": policy.state_dict(),
                "value_net_state_dict": value_net.state_dict(),
                "policy_optimizer": policy_optimizer.state_dict(),
                "value_optimizer": value_optimizer.state_dict(),
                "env_step": env_step,
                "iteration": iteration,
            }
            torch.save(ckpt, out_dir / "latest_checkpoint.pt")
            torch.save(ckpt, out_dir / f"checkpoint_{env_step}.pt")
            print(f"  Saved checkpoint at step {env_step}")

        rollout_buffer.clear()

    # ---- Final save ----
    ckpt = {
        "policy_state_dict": policy.state_dict(),
        "value_net_state_dict": value_net.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
        "env_step": env_step,
        "iteration": iteration,
    }
    torch.save(ckpt, out_dir / "latest_checkpoint.pt")
    torch.save(ckpt, out_dir / "final_checkpoint.pt")
    print(f"\nDPPO training complete. {env_step} env steps, {iteration} iterations.")
    print(f"Checkpoints saved to {out_dir}")

    env.close()
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
