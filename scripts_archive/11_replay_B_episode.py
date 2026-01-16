#!/usr/bin/env python3
"""Replay a single episode from B (reverse) dataset in Isaac Lab and save as mp4.

This script visualizes the collected reverse trajectories by:
1. Loading a specific episode from the B dataset (NPZ file)
2. Teleporting the object to its initial position (first frame)
3. Replaying the EE pose + gripper actions in the simulator
4. Recording the visualization as an mp4 video

Usage:
    # Replay episode 0 (default) with GUI
    python scripts/11_replay_B_episode.py

    # Replay episode 5 in headless mode and save video
    python scripts/11_replay_B_episode.py --episode 5 --headless

    # Custom dataset and output
    python scripts/11_replay_B_episode.py --dataset data/B_reverse_100eps.npz --episode 10 --out data/B_ep10.mp4
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay a B (reverse) episode in Isaac Lab and save as mp4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/B_reverse_latest.npz",
        help="Path to the B dataset NPZ file.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay (0-indexed).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mp4 file path. Default: data/B_ep{episode}.mp4",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Gym task ID.",
    )
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disables Fabric backend.",
    )
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time).",
    )

    # Isaac Lab AppLauncher arguments
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # Set default output path
    if args.out is None:
        args.out = f"data/B_ep{args.episode}.mp4"

    # Enable cameras for video recording
    args.enable_cameras = True

    return args


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    # Launch Isaac Sim (must be before other imports)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import other modules
    import gymnasium as gym
    import numpy as np
    import torch
    import shutil
    from pathlib import Path

    from rev2fwd_il.data.io_npz import load_episodes
    from rev2fwd_il.sim.scene_api import teleport_object_to_pose
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    import isaaclab_tasks  # noqa: F401

    try:
        # =====================================================================
        # Step 1: Load episode data
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"Loading B (reverse) dataset: {args.dataset}")
        print(f"{'='*60}")
        
        # Check if dataset file exists
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {args.dataset}\n"
                f"Please run the data collection script first to generate the B dataset."
            )
        
        episodes = load_episodes(args.dataset)

        if args.episode < 0 or args.episode >= len(episodes):
            raise ValueError(f"Episode {args.episode} out of range [0, {len(episodes)-1}]")

        ep = episodes[args.episode]
        print(f"Episode {args.episode}: length={ep.length}, success={ep.success}")
        print(f"  - EE pose shape: {ep.ee_pose.shape}")
        print(f"  - Gripper shape: {ep.gripper.shape}")
        print(f"  - Object initial pos: {ep.obj_pose[0, :3]}")

        # =====================================================================
        # Step 2: Create environment with video recording
        # =====================================================================
        print("\nCreating environment...")

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device if args.device else "cuda:0",
            num_envs=1,
            use_fabric=not bool(args.disable_fabric),
        )

        # Adjust camera position to be 1/3 of the default distance (closer to robot)
        # Default eye is typically around (2.5, 2.5, 2.5) or similar
        # We set it to 1/3 distance to make the robot appear larger
        env_cfg.viewer.eye = (1.5, 1.5, 1.5)
        env_cfg.viewer.lookat = (0.2, 0.0, 0.0)  # Look at the table center

        # Disable terminations to prevent auto-reset during replay
        # This ensures the robot moves naturally without teleporting back
        env_cfg.episode_length_s = 100.0  # Prevent timeout reset
        if hasattr(env_cfg, 'terminations'):
            if hasattr(env_cfg.terminations, 'time_out'):
                env_cfg.terminations.time_out = None
            if hasattr(env_cfg.terminations, 'object_dropping'):
                env_cfg.terminations.object_dropping = None

        # Create environment with render_mode for video recording
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

        # Setup video recording directory
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        video_dir = out_path.parent / f"_tmp_video_B_ep{args.episode}"

        # Wrap with RecordVideo
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: True,  # Record all episodes
            disable_logger=True,
            name_prefix=f"B_ep{args.episode}",
        )

        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs

        # =====================================================================
        # Step 3: Reset and teleport object to initial position
        # =====================================================================
        print("\nResetting environment...")
        obs, _ = env.reset()

        # Teleport object to first frame position
        obj_init_pose = torch.tensor(ep.obj_pose[0:1], dtype=torch.float32, device=device)
        teleport_object_to_pose(env, obj_init_pose, name="object")

        # Let physics settle
        zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        for _ in range(10):
            env.step(zero_action)

        print(f"Object teleported to: {ep.obj_pose[0, :3]}")

        # =====================================================================
        # Step 4: Replay trajectory
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"Replaying B episode {args.episode} ({ep.length} steps)")
        print(f"This is REVERSE trajectory: goal -> table")
        print(f"{'='*60}\n")

        # Compute step skip based on playback speed
        step_skip = max(1, int(args.playback_speed))

        for t in range(0, ep.length, step_skip):
            # Construct action from saved EE pose and gripper
            ee_pose_t = torch.tensor(ep.ee_pose[t], dtype=torch.float32, device=device).unsqueeze(0)
            gripper_t = torch.tensor([[ep.gripper[t]]], dtype=torch.float32, device=device)

            # Action = [ee_pose(7), gripper(1)]
            action = torch.cat([ee_pose_t, gripper_t], dim=-1)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print progress every 50 steps
            if t % 50 == 0:
                gripper_str = "OPEN" if ep.gripper[t] > 0 else "CLOSE"
                print(
                    f"Step {t:4d}/{ep.length} | "
                    f"EE: [{ep.ee_pose[t, 0]:.3f}, {ep.ee_pose[t, 1]:.3f}, {ep.ee_pose[t, 2]:.3f}] | "
                    f"Gripper: {gripper_str}"
                )

        print(f"\nReplay finished!")

        # =====================================================================
        # Step 5: Close and rename video file
        # =====================================================================
        env.close()

        # Find the generated video file and rename it
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            # Move the first mp4 to the desired output path
            shutil.move(str(video_files[0]), str(out_path))
            print(f"\n{'='*60}")
            print(f"Video saved to: {out_path}")
            print(f"{'='*60}")
            # Clean up temp directory
            shutil.rmtree(video_dir, ignore_errors=True)
        else:
            print(f"\nWarning: No video file generated in {video_dir}")

        print(f"\nB Episode {args.episode} Replay Summary")
        print(f"  Dataset: {args.dataset}")
        print(f"  Episode length: {ep.length}")
        print(f"  Success (original): {ep.success}")
        print(f"  Output video: {out_path}")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
