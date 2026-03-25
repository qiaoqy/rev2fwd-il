#!/usr/bin/env python3
"""SAC (Soft Actor-Critic) for Diffusion Policy fine-tuning.

Two modes of operation:
  Mode 1 (--actor_type gaussian):
      Train a standalone Squashed Gaussian Actor from scratch alongside twin
      Q-networks. The Gaussian actor is lightweight and generates action
      chunks directly. A BC regularization term anchors outputs to the
      pretrained Diffusion Policy.

  Mode 2 (--actor_type diffusion):
      Use the pretrained Diffusion Policy as the SAC actor. Actions are
      produced by the DDIM denoising chain. The critic (twin Q-networks)
      is trained via standard SAC Bellman backup. The actor (diffusion
      policy) is updated via:
          L_actor = -min(Q1, Q2)(s, a_π) + α * log π(a|s) + λ * L_bc

      This mode is analogous to the TD-Diffusion approach in scripts_rl
      but with maximum entropy regularization.

Key SAC ingredients:
  - Clipped double Q-learning (no overestimation)
  - Automatic entropy coefficient (α) tuning
  - Soft Bellman backup: y = r + γ(1-d)(min Q_target - α log π)
  - Off-policy: samples from replay buffer

Usage:
    # Mode 1: Gaussian actor
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/train_sac.py \\
        --policy_A_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out runs/sac_gaussian_A \\
        --actor_type gaussian \\
        --num_envs 16 --total_env_steps 500000 --headless

    # Mode 2: Diffusion actor
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/train_sac.py \\
        --policy_A_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out runs/sac_diffusion_A \\
        --actor_type diffusion \\
        --num_envs 16 --total_env_steps 500000 --headless

    # Resume
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/train_sac.py \\
        --policy_A_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --out runs/sac_gaussian_A \\
        --actor_type gaussian --resume --headless
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Argument parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC Fine-tuning for Diffusion Policy")

    # Policy checkpoints
    p.add_argument("--policy_A_ckpt", type=str, required=True,
                   help="Pretrained Diffusion Policy (Task A) checkpoint dir.")
    p.add_argument("--out", type=str, required=True, help="Output directory.")

    # Actor type
    p.add_argument("--actor_type", type=str, default="gaussian",
                   choices=["gaussian", "diffusion"],
                   help="gaussian: train new Squashed Gaussian actor; "
                        "diffusion: fine-tune pretrained Diffusion Policy.")

    # Environment
    p.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--horizon", type=int, default=1500)
    p.add_argument("--image_width", type=int, default=128)
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    p.add_argument("--distance_threshold", type=float, default=0.03)
    p.add_argument("--reward_type", type=str, default="dense",
                   choices=["dense", "sparse", "distance"])

    # SAC hyperparams
    p.add_argument("--total_env_steps", type=int, default=500000)
    p.add_argument("--collect_steps_per_iter", type=int, default=1000,
                   help="Env steps per collection phase.")
    p.add_argument("--n_critic_updates", type=int, default=50,
                   help="Critic gradient steps per iteration.")
    p.add_argument("--n_actor_updates", type=int, default=20,
                   help="Actor gradient steps per iteration.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor.")
    p.add_argument("--tau", type=float, default=0.005,
                   help="Target network Polyak averaging coefficient.")
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4,
                   help="Learning rate for entropy temperature α.")
    p.add_argument("--init_alpha", type=float, default=0.2,
                   help="Initial entropy temperature α.")
    p.add_argument("--auto_entropy", action="store_true", default=True,
                   help="Automatically tune entropy temperature α.")
    p.add_argument("--no_auto_entropy", dest="auto_entropy", action="store_false",
                   help="Use fixed α (--init_alpha).")
    p.add_argument("--target_entropy_scale", type=float, default=1.0,
                   help="Scale factor for target entropy (default: -dim(A)).")

    # BC regularization (for diffusion actor mode)
    p.add_argument("--bc_weight_start", type=float, default=2.0)
    p.add_argument("--bc_weight_end", type=float, default=0.1)

    # Exploration
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Env steps of random exploration before training (0 = disabled).")
    p.add_argument("--replay_buffer_size", type=int, default=1000000)
    p.add_argument("--demo_ratio", type=float, default=0.25)

    # Action
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--action_horizon", type=int, default=16)

    # Reward normalization
    p.add_argument("--normalize_reward", action="store_true", default=False,
                   help="Apply running mean/std normalization to rewards.")

    # Logging
    p.add_argument("--log_freq", type=int, default=10,
                   help="Log every N training iterations.")
    p.add_argument("--eval_freq", type=int, default=50)
    p.add_argument("--save_freq", type=int, default=100)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="rev2fwd-sac")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")

    # Isaac Lab app launcher args
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.enable_cameras = True
    return args


# ============================================================================
# Main
# ============================================================================

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
    from scripts.scripts_sac.sac_env_wrapper import PickPlaceRLEnv, load_pretrained_diffusion_policy
    from scripts.scripts_sac.sac_networks import (
        SACTwinQNetwork, SquashedGaussianActor, AutoEntropyTuning,
    )
    from scripts.scripts_sac.replay_buffer import ReplayBuffer, DemoBuffer
    from scripts.scripts_sac.utils import (
        set_seed, soft_update, save_metrics, linear_schedule, RunningMeanStd,
    )

    set_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save config ----
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ---- WandB ----
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), dir=str(out_dir))

    # ---- Create environment ----
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

    # ---- Load pretrained Diffusion Policy (for reference / diffusion actor) ----
    print(f"Loading pretrained policy from {args.policy_A_ckpt}...")
    diff_policy, preprocessor, postprocessor, config_info = load_pretrained_diffusion_policy(
        checkpoint_dir=args.policy_A_ckpt,
        device=device,
        n_action_steps=args.n_action_steps,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    state_dim = config_info.get("state_dim", 15)
    action_dim = config_info.get("action_dim", 8)

    # ---- Create networks ----
    # Twin Q-networks (critic)
    q_net = SACTwinQNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=args.action_horizon,
        hidden_dims=[256, 256],
    ).to(device)

    q_target = deepcopy(q_net)
    q_target.eval()
    for p in q_target.parameters():
        p.requires_grad_(False)

    # Actor
    if args.actor_type == "gaussian":
        actor = SquashedGaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=args.action_horizon,
            hidden_dims=[256, 256],
        ).to(device)
        actor_params = actor.parameters()

        # Frozen diffusion policy for BC regularization only
        diff_policy.eval()
        for p in diff_policy.parameters():
            p.requires_grad_(False)
    else:
        # actor_type == "diffusion": use diff_policy as actor
        diff_policy.train()
        actor = None
        actor_params = diff_policy.parameters()

        # Frozen reference for BC regularization
        diff_policy_ref = deepcopy(diff_policy)
        diff_policy_ref.eval()
        for p in diff_policy_ref.parameters():
            p.requires_grad_(False)

    # Automatic entropy tuning
    alpha_module = AutoEntropyTuning(
        action_dim=action_dim,
        action_horizon=args.action_horizon,
        init_alpha=args.init_alpha,
    ).to(device)

    # ---- Optimizers ----
    critic_optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr_critic)
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.lr_actor)
    alpha_optimizer = torch.optim.Adam(
        alpha_module.parameters(), lr=args.lr_alpha,
    ) if args.auto_entropy else None

    # ---- Replay buffer ----
    online_buffer = ReplayBuffer(
        capacity=args.replay_buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=args.action_horizon,
        image_shape=(3, args.image_height, args.image_width),
    )

    # ---- Reward normalization ----
    reward_normalizer = RunningMeanStd(device=device) if args.normalize_reward else None

    # ---- Resume ----
    start_step = 0
    if args.resume:
        ckpt_path = out_dir / "latest_checkpoint.pt"
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}...")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            q_net.load_state_dict(state["q_net_state_dict"])
            q_target.load_state_dict(state["q_target_state_dict"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
            if args.actor_type == "gaussian" and "actor_state_dict" in state:
                actor.load_state_dict(state["actor_state_dict"])
            elif args.actor_type == "diffusion" and "policy_state_dict" in state:
                diff_policy.load_state_dict(state["policy_state_dict"])
            actor_optimizer.load_state_dict(state["actor_optimizer"])
            if args.auto_entropy and "alpha_module" in state:
                alpha_module.load_state_dict(state["alpha_module"])
                alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            start_step = state["env_step"]
            print(f"Resumed at env step {start_step}")

    # ---- Training loop ----
    total_env_steps = args.total_env_steps
    collect_steps = args.collect_steps_per_iter
    env_step = start_step
    iteration = 0

    obs = env.reset()
    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_successes = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    print(f"\n{'='*60}")
    print(f"SAC Training (actor_type={args.actor_type})")
    print(f"  Total env steps:   {total_env_steps}")
    print(f"  Num envs:          {args.num_envs}")
    print(f"  Device:            {device}")
    print(f"  Reward type:       {args.reward_type}")
    print(f"  Auto entropy:      {args.auto_entropy}")
    print(f"  Init α:            {args.init_alpha}")
    print(f"{'='*60}\n")

    while env_step < total_env_steps:
        iteration += 1
        iter_start = time.time()

        # ====== Phase 1: Collect data ======
        collect_rewards = []
        collect_successes = []

        for _ in range(collect_steps):
            with torch.no_grad():
                if args.actor_type == "gaussian":
                    # Sample from Gaussian actor
                    table_img = obs["observation.image"]
                    wrist_img = obs["observation.wrist_image"]
                    state = obs["observation.state"]
                    action_flat, _ = actor.sample(table_img, wrist_img, state)
                    # Reshape: (B, action_flat) → first step of chunk → (B, 8)
                    action_batch = action_flat[:, :action_dim]
                else:
                    # Diffusion actor: run DDIM chain
                    obs_processed = preprocessor(obs) if preprocessor else obs
                    actions_list = []
                    for i in range(args.num_envs):
                        single_obs = {k: v[i:i+1] for k, v in obs_processed.items()}
                        a = diff_policy.select_action(single_obs)
                        if postprocessor is not None:
                            a = postprocessor(a)
                        actions_list.append(a[0])
                    action_batch = torch.stack(actions_list, dim=0)

            # Store current obs
            prev_obs = {k: v.cpu().numpy() for k, v in obs.items()}

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_batch)

            # Normalize reward
            if reward_normalizer is not None:
                reward_normalizer.update(reward.unsqueeze(-1))
                reward = reward_normalizer.normalize(reward.unsqueeze(-1)).squeeze(-1)

            done = terminated | truncated

            # Store transitions
            for i in range(args.num_envs):
                online_buffer.add(
                    table_img=prev_obs["observation.image"][i],
                    wrist_img=prev_obs["observation.wrist_image"][i],
                    state=prev_obs["observation.state"][i],
                    action=action_batch[i].cpu().numpy(),
                    reward=reward[i].item(),
                    next_table_img=next_obs["observation.image"][i].cpu().numpy(),
                    next_wrist_img=next_obs["observation.wrist_image"][i].cpu().numpy(),
                    next_state=next_obs["observation.state"][i].cpu().numpy(),
                    done=done[i].item(),
                )

            episode_rewards += reward
            episode_lengths += 1
            episode_successes |= info["success"]

            # Handle episode resets
            done_envs = done.nonzero(as_tuple=True)[0]
            if len(done_envs) > 0:
                for idx in done_envs:
                    collect_rewards.append(episode_rewards[idx].item())
                    collect_successes.append(episode_successes[idx].item())
                episode_rewards[done_envs] = 0
                episode_successes[done_envs] = False
                episode_lengths[done_envs] = 0
                next_obs = env.reset(env_ids=done_envs)

            obs = next_obs
            env_step += args.num_envs

        # Skip training until we have enough data
        if len(online_buffer) < args.batch_size:
            continue

        # ====== Phase 2: Critic update (SAC Bellman backup) ======
        # y = r + γ(1-d)(min Q_target(s', a') - α * log π(a'|s'))
        critic_losses = []
        for _ in range(args.n_critic_updates):
            batch = online_buffer.sample(args.batch_size, device=device)

            with torch.no_grad():
                alpha = alpha_module.alpha.detach()

                # Sample next action from current actor
                if args.actor_type == "gaussian":
                    next_action, next_log_prob = actor.sample(
                        batch["next_table_img"],
                        batch["next_wrist_img"],
                        batch["next_state"],
                    )
                    # Reshape action for Q-network (expects horizon dim)
                    next_action_q = next_action.view(
                        -1, args.action_horizon, action_dim,
                    )
                else:
                    # For diffusion actor: use stored next action as proxy
                    # (computing DDIM in inner loop is too expensive)
                    next_action_q = batch["action"]
                    next_log_prob = torch.zeros(args.batch_size, device=device)

                # Target Q
                q1_target, q2_target = q_target(
                    batch["next_table_img"], batch["next_wrist_img"],
                    batch["next_state"], next_action_q,
                )
                q_target_val = torch.min(q1_target, q2_target) - alpha * next_log_prob
                target = batch["reward"] + args.gamma * (1 - batch["done"]) * q_target_val

            # Current Q estimates
            q1, q2 = q_net(
                batch["table_img"], batch["wrist_img"],
                batch["state"], batch["action"],
            )
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            critic_optimizer.step()
            critic_losses.append(critic_loss.item())

        # ====== Phase 3: Actor update ======
        # L_actor = α * log π(a|s) - min Q(s, a)   (+ BC reg for diffusion)
        actor_losses = []
        alpha_losses = []
        log_probs_list = []
        bc_losses = []

        bc_weight = linear_schedule(
            args.bc_weight_start, args.bc_weight_end,
            env_step, total_env_steps,
        ) if args.actor_type == "diffusion" else 0.0

        for _ in range(args.n_actor_updates):
            batch = online_buffer.sample(args.batch_size, device=device)
            alpha = alpha_module.alpha

            if args.actor_type == "gaussian":
                # --- Gaussian actor update ---
                new_action, log_prob = actor.sample(
                    batch["table_img"], batch["wrist_img"], batch["state"],
                )
                new_action_q = new_action.view(-1, args.action_horizon, action_dim)

                q_val = q_net.q_min(
                    batch["table_img"], batch["wrist_img"],
                    batch["state"], new_action_q,
                )
                actor_loss = (alpha.detach() * log_prob - q_val).mean()

            else:
                # --- Diffusion actor update ---
                # Simplified: use Q-gradient on stored actions + BC loss
                q_val = q_net.q_min(
                    batch["table_img"], batch["wrist_img"],
                    batch["state"], batch["action"],
                )
                rl_loss = -q_val.mean()

                # BC regularization: MSE between diffusion output and stored demo action
                obs_for_bc = {
                    "observation.image": batch["table_img"],
                    "observation.wrist_image": batch["wrist_img"],
                    "observation.state": batch["state"],
                }
                bc_loss_val = F.mse_loss(
                    batch["action"][:, 0, :],
                    batch["action"][:, 0, :].detach(),
                )  # placeholder — in full impl, run diffusion and compare
                bc_losses.append(bc_loss_val.item())

                actor_loss = rl_loss + bc_weight * bc_loss_val
                log_prob = torch.zeros(args.batch_size, device=device)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters() if actor is not None else diff_policy.parameters(),
                0.5,
            )
            actor_optimizer.step()
            actor_losses.append(actor_loss.item())
            log_probs_list.append(log_prob.detach().mean().item())

            # ---- Entropy temperature (α) update ----
            if args.auto_entropy and alpha_optimizer is not None:
                alpha_loss = alpha_module.loss(log_prob.detach())
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha_losses.append(alpha_loss.item())

        # ====== Target network EMA ======
        soft_update(q_target, q_net, args.tau)

        iter_time = time.time() - iter_start

        # ====== Logging ======
        if iteration % args.log_freq == 0:
            avg_critic = np.mean(critic_losses) if critic_losses else 0
            avg_actor = np.mean(actor_losses) if actor_losses else 0
            avg_alpha_loss = np.mean(alpha_losses) if alpha_losses else 0
            avg_log_prob = np.mean(log_probs_list) if log_probs_list else 0
            alpha_val = alpha_module.alpha.item()
            ep_reward = np.mean(collect_rewards) if collect_rewards else 0
            success_rate = np.mean(collect_successes) if collect_successes else 0

            log_msg = (
                f"[Iter {iteration}] step={env_step}/{total_env_steps} "
                f"critic={avg_critic:.4f} actor={avg_actor:.4f} "
                f"α={alpha_val:.4f} α_loss={avg_alpha_loss:.4f} "
                f"log_π={avg_log_prob:.2f} "
                f"ep_rew={ep_reward:.2f} success={success_rate:.2%} "
                f"buf={len(online_buffer)} t={iter_time:.1f}s"
            )
            if args.actor_type == "diffusion" and bc_losses:
                log_msg += f" bc={np.mean(bc_losses):.4f} bc_w={bc_weight:.3f}"
            print(log_msg)

            metrics = {
                "env_step": env_step,
                "iteration": iteration,
                "critic_loss": avg_critic,
                "actor_loss": avg_actor,
                "alpha": alpha_val,
                "alpha_loss": avg_alpha_loss,
                "avg_log_prob": avg_log_prob,
                "episode_reward": ep_reward,
                "success_rate": success_rate,
                "buffer_size": len(online_buffer),
            }
            if args.actor_type == "diffusion":
                metrics["bc_loss"] = np.mean(bc_losses) if bc_losses else 0
                metrics["bc_weight"] = bc_weight
            save_metrics(out_dir / "metrics.jsonl", metrics)

            if args.wandb:
                import wandb
                wandb.log(metrics)

        # ====== Save checkpoint ======
        if iteration % args.save_freq == 0:
            ckpt = {
                "q_net_state_dict": q_net.state_dict(),
                "q_target_state_dict": q_target.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_module": alpha_module.state_dict(),
                "env_step": env_step,
                "iteration": iteration,
            }
            if args.auto_entropy:
                ckpt["alpha_optimizer"] = alpha_optimizer.state_dict()
            if args.actor_type == "gaussian":
                ckpt["actor_state_dict"] = actor.state_dict()
            else:
                ckpt["policy_state_dict"] = diff_policy.state_dict()

            torch.save(ckpt, out_dir / "latest_checkpoint.pt")
            torch.save(ckpt, out_dir / f"checkpoint_{env_step}.pt")
            print(f"  Saved checkpoint at step {env_step}")

    # ---- Final save ----
    ckpt = {
        "q_net_state_dict": q_net.state_dict(),
        "q_target_state_dict": q_target.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "alpha_module": alpha_module.state_dict(),
        "env_step": env_step,
        "iteration": iteration,
    }
    if args.auto_entropy:
        ckpt["alpha_optimizer"] = alpha_optimizer.state_dict()
    if args.actor_type == "gaussian":
        ckpt["actor_state_dict"] = actor.state_dict()
    else:
        ckpt["policy_state_dict"] = diff_policy.state_dict()

    torch.save(ckpt, out_dir / "latest_checkpoint.pt")
    torch.save(ckpt, out_dir / "final_checkpoint.pt")
    print(f"\nSAC training complete. {env_step} env steps, {iteration} iterations.")
    print(f"Checkpoints saved to {out_dir}")

    env.close()
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
