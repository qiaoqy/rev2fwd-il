#!/usr/bin/env python3
"""Evaluate an SAC-finetuned policy on pick-place.

Runs independent episodes and reports success rate, compatible with
the existing 7_eval_fair.py output format for fair comparison.

Supports both actor types:
  - gaussian: loads SquashedGaussianActor checkpoint
  - diffusion: loads Diffusion Policy with RL-updated weights

Usage:
    # Evaluate Gaussian actor
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/eval_sac.py \\
        --checkpoint runs/sac_gaussian_A/latest_checkpoint.pt \\
        --bc_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --actor_type gaussian \\
        --out runs/sac_gaussian_A/eval_results.json \\
        --num_episodes 50 --headless

    # Evaluate Diffusion actor
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_sac/eval_sac.py \\
        --checkpoint runs/sac_diffusion_A/latest_checkpoint.pt \\
        --bc_ckpt data/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --actor_type diffusion \\
        --out runs/sac_diffusion_A/eval_results.json \\
        --num_episodes 50 --headless
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAC-finetuned Policy")

    p.add_argument("--checkpoint", type=str, required=True,
                   help="SAC checkpoint (.pt file).")
    p.add_argument("--bc_ckpt", type=str, required=True,
                   help="Original BC checkpoint dir (for architecture/config).")
    p.add_argument("--actor_type", type=str, default="gaussian",
                   choices=["gaussian", "diffusion"])
    p.add_argument("--out", type=str, required=True,
                   help="Output stats JSON path.")
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=1500)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--image_width", type=int, default=128)
    p.add_argument("--image_height", type=int, default=128)
    p.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    p.add_argument("--distance_threshold", type=float, default=0.03)
    p.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    p.add_argument("--seed", type=int, default=42)

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
        _run_eval(args)
    finally:
        simulation_app.close()


def _run_eval(args: argparse.Namespace) -> None:
    from scripts.scripts_sac.sac_env_wrapper import PickPlaceRLEnv, load_pretrained_diffusion_policy
    from scripts.scripts_sac.utils import set_seed

    set_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load BC architecture
    diff_policy, preprocessor, postprocessor, config_info = load_pretrained_diffusion_policy(
        checkpoint_dir=args.bc_ckpt,
        device=device,
        n_action_steps=args.n_action_steps,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    state_dim = config_info.get("state_dim", 15)
    action_dim = config_info.get("action_dim", 8)

    # Load SAC checkpoint
    rl_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rl_step = rl_ckpt.get("env_step", 0)

    if args.actor_type == "gaussian":
        from scripts.scripts_sac.sac_networks import SquashedGaussianActor
        actor = SquashedGaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=args.action_horizon,
            hidden_dims=[256, 256],
        ).to(device)
        actor.load_state_dict(rl_ckpt["actor_state_dict"])
        actor.eval()
        print(f"Loaded Gaussian actor from step {rl_step}")
    else:
        diff_policy.load_state_dict(rl_ckpt["policy_state_dict"])
        diff_policy.eval()
        print(f"Loaded Diffusion actor from step {rl_step}")

    # Create environment (single env for fair evaluation)
    env = PickPlaceRLEnv(
        task_id=args.task,
        num_envs=1,
        device=device,
        image_width=args.image_width,
        image_height=args.image_height,
        goal_xy=tuple(args.goal_xy),
        distance_threshold=args.distance_threshold,
        horizon=args.horizon,
        reward_type="dense",
        headless=getattr(args, "headless", True),
    )

    # Run evaluation
    results = []
    total_rewards = []
    start_time = time.time()

    for ep in range(args.num_episodes):
        obs = env.reset()
        if args.actor_type == "diffusion":
            diff_policy.reset()
        episode_reward = 0.0
        success = False

        for t in range(args.horizon):
            with torch.no_grad():
                if args.actor_type == "gaussian":
                    table_img = obs["observation.image"]
                    wrist_img = obs["observation.wrist_image"]
                    state = obs["observation.state"]
                    action = actor.deterministic_action(
                        table_img, wrist_img, state,
                    )
                    # Take first action_dim elements as the action step
                    action = action[:, :action_dim]
                else:
                    obs_processed = preprocessor(obs) if preprocessor else obs
                    action = diff_policy.select_action(obs_processed)
                    if postprocessor is not None:
                        action = postprocessor(action)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward.item()

            if info["success"].any():
                success = True

            if terminated.any() or truncated.any():
                break

        results.append(success)
        total_rewards.append(episode_reward)
        status = "SUCCESS" if success else "FAILED"
        print(f"  Episode {ep+1}/{args.num_episodes}: {status} "
              f"(reward={episode_reward:.1f}, steps={t+1})")

    elapsed = time.time() - start_time
    num_success = sum(results)
    success_rate = num_success / len(results) if results else 0.0

    print(f"\n{'='*50}")
    print(f"Evaluation Results (SAC {args.actor_type})")
    print(f"{'='*50}")
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Success:      {num_success}/{args.num_episodes} = {success_rate:.1%}")
    print(f"  Avg reward:   {np.mean(total_rewards):.2f}")
    print(f"  Time:         {elapsed:.1f}s")

    # Save results (compatible with existing fair test format)
    stats = {
        "task": "A",
        "method": f"sac_{args.actor_type}",
        "rl_checkpoint": args.checkpoint,
        "bc_checkpoint": args.bc_ckpt,
        "rl_env_step": rl_step,
        "num_total": args.num_episodes,
        "num_success": num_success,
        "success_rate": success_rate,
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "elapsed_seconds": elapsed,
        "per_episode": [
            {"episode": i, "success": results[i], "reward": total_rewards[i]}
            for i in range(len(results))
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
