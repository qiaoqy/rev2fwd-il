#!/usr/bin/env python3
"""Replay a single episode from A (forward) dataset in Isaac Lab and save as mp4.

This script visualizes the forward trajectories (time-reversed from B) by:
1. Loading a specific episode from the A dataset (flat NPZ file with obs/act/ep_id)
2. Teleporting the object to its initial position (first frame of this episode)
3. Replaying the EE pose + gripper actions in the simulator
4. Recording the visualization as an mp4 video

The A dataset is the time-reversed version of B, so this shows the FORWARD task:
table random position -> goal (plate center)

Usage:
    # Replay episode 0 (default) with GUI
    python scripts/21_replay_A_episode.py

    # Replay episode 5 in headless mode and save video
    python scripts/21_replay_A_episode.py --episode 5 --headless

    # Custom dataset and output
    python scripts/21_replay_A_episode.py --dataset data/A_forward_from_reverse.npz --episode 10 --out data/A_ep10.mp4

    # Use corresponding B dataset for object initial position
    python scripts/21_replay_A_episode.py --episode 0 --b_dataset data/B_reverse_100eps.npz
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay an A (forward) episode in Isaac Lab and save as mp4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/A_forward_from_reverse.npz",
        help="Path to the A dataset NPZ file (flat format with obs/act/ep_id).",
    )
    parser.add_argument(
        "--b_dataset",
        type=str,
        default="data/B_reverse_100eps.npz",
        help="Path to the B dataset NPZ file (for object initial position, since A is time-reversed B).",
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
        help="Output mp4 file path. Default: data/A_ep{episode}.mp4",
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
        args.out = f"data/A_ep{args.episode}.mp4"

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
        # Step 1: Load A dataset (flat format)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"Loading A (forward) dataset: {args.dataset}")
        print(f"{'='*60}")

        with np.load(args.dataset) as data:
            all_obs = data["obs"]      # (N, 36)
            all_act = data["act"]      # (N, 8) = [ee_pose(7), gripper(1)]
            all_ep_id = data["ep_id"]  # (N,)

        # Find unique episodes
        unique_eps = np.unique(all_ep_id)
        print(f"Total steps: {len(all_obs)}, Unique episodes: {len(unique_eps)}")

        if args.episode < 0 or args.episode >= len(unique_eps):
            raise ValueError(f"Episode {args.episode} out of range [0, {len(unique_eps)-1}]")

        # Extract this episode's data
        ep_mask = all_ep_id == args.episode
        ep_act = all_act[ep_mask]  # (T, 8)
        ep_length = len(ep_act)

        print(f"Episode {args.episode}: length={ep_length}")
        print(f"  - Action shape: {ep_act.shape}")

        # =====================================================================
        # Step 2: Load B dataset to get object initial position
        # A is time-reversed B, so A's initial obj position = B's final obj position
        # =====================================================================
        print(f"\nLoading B dataset for object position: {args.b_dataset}")
        b_episodes = load_episodes(args.b_dataset)

        if args.episode >= len(b_episodes):
            raise ValueError(f"Episode {args.episode} not found in B dataset (only {len(b_episodes)} episodes)")

        b_ep = b_episodes[args.episode]
        # A's initial position = B's final position (time reversal)
        # But actually, for visualization we use the LAST frame of B as A's START
        obj_init_pos = b_ep.obj_pose[-1, :3]  # Final position in B = Initial position in A
        obj_init_pose = b_ep.obj_pose[-1]     # Full pose (7,)

        print(f"  Object initial pos (from B's final frame): {obj_init_pos}")

        # =====================================================================
        # Step 3: Create environment with video recording
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
        video_dir = out_path.parent / f"_tmp_video_A_ep{args.episode}"

        # Wrap with RecordVideo
        env = gym.wrappers.RecordVideo(
            env,
            str(video_dir),
            episode_trigger=lambda x: True,  # Record all episodes
            disable_logger=True,
            name_prefix=f"A_ep{args.episode}",
        )

        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs

        # =====================================================================
        # Step 4: Reset and teleport object to initial position
        # =====================================================================
        print("\nResetting environment...")
        obs, _ = env.reset()

        # Teleport object to initial position (B's final position)
        obj_pose_tensor = torch.tensor(obj_init_pose, dtype=torch.float32, device=device).unsqueeze(0)
        teleport_object_to_pose(env, obj_pose_tensor, name="object")

        # Let physics settle
        zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        for _ in range(10):
            env.step(zero_action)

        print(f"Object teleported to: {obj_init_pos}")

        # =====================================================================
        # Step 5: Replay trajectory
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"Replaying A episode {args.episode} ({ep_length} steps)")
        print(f"This is FORWARD trajectory: table -> goal (time-reversed from B)")
        print(f"{'='*60}\n")

        # Compute step skip based on playback speed
        step_skip = max(1, int(args.playback_speed))

        for t in range(0, ep_length, step_skip):
            # Action from A dataset: [ee_pose(7), gripper(1)]
            action = torch.tensor(ep_act[t], dtype=torch.float32, device=device).unsqueeze(0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Print progress every 50 steps
            if t % 50 == 0:
                ee_pose = ep_act[t, :3]
                gripper_val = ep_act[t, 7]
                gripper_str = "OPEN" if gripper_val > 0 else "CLOSE"
                print(
                    f"Step {t:4d}/{ep_length} | "
                    f"EE: [{ee_pose[0]:.3f}, {ee_pose[1]:.3f}, {ee_pose[2]:.3f}] | "
                    f"Gripper: {gripper_str}"
                )

        print(f"\nReplay finished!")

        # =====================================================================
        # Step 6: Close and rename video file
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

        print(f"\nA Episode {args.episode} Replay Summary")
        print(f"  Dataset: {args.dataset}")
        print(f"  Episode length: {ep_length}")
        print(f"  Object initial pos: {obj_init_pos}")
        print(f"  Output video: {out_path}")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
