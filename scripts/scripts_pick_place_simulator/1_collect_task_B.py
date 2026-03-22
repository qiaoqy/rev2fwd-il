#!/usr/bin/env python3
"""Step 1: Collect Task B data using FSM expert (Mode 3 — red rectangle only).

Expert B picks the cube from the green goal position and places it at a random
position inside the red rectangular region.

Action convention: action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]

Wrist camera is always enabled.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_task_B.py \
        --out data/exp_new/task_B_100.npz \
        --num_episodes 100 --headless
"""

from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Task B data with FSM expert (red rectangle region).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")

    # Region parameters (Mode 3 hardcoded)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2, default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2, default=[0.3, 0.3])
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- imports that require Isaac Sim running ----
    import importlib.util
    from pathlib import Path

    import numpy as np

    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.utils.seed import set_seed

    # Reuse env / marker utilities from the original collection script
    _orig_spec = importlib.util.spec_from_file_location(
        "collect_orig",
        str(Path(__file__).resolve().parent.parent / "scripts_pick_place" / "1_collect_data_pick_place.py"),
    )
    _orig = importlib.util.module_from_spec(_orig_spec)
    _orig_spec.loader.exec_module(_orig)

    make_env_with_camera = _orig.make_env_with_camera
    rollout_expert_B_with_goal_actions = _orig.rollout_expert_B_with_goal_actions
    save_episodes_with_goal_actions = _orig.save_episodes_with_goal_actions

    try:
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        # ---- Create environment ----
        env = make_env_with_camera(
            task_id=args.task,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,
            disable_terminations=True,
        )
        device = env.unwrapped.device

        # ---- Task spec (Mode 3: red rectangle) ----
        task_spec = PickPlaceTaskSpec(
            goal_xy=tuple(args.goal_xy),
            hover_z=0.25,
            grasp_z_offset=0.0,
            success_radius=0.03,
            settle_steps=10,
        )
        task_spec.taskB_target_mode = "red_region"
        task_spec.red_region_center_xy = tuple(args.red_region_center_xy)
        task_spec.red_region_size_xy = tuple(args.red_region_size_xy)
        task_spec.red_marker_shape = "rectangle"
        task_spec.red_marker_size_xy = tuple(args.red_region_size_xy)
        task_spec.fix_red_marker_pose = True

        # ---- Expert B ----
        expert = PickPlaceExpertB(
            num_envs=args.num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            release_z_offset=-0.04,
            position_threshold=0.015,
            wait_steps=task_spec.settle_steps,
        )

        # ---- Collection loop ----
        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        markers = None
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} Task B episodes (Mode 3: red rectangle)")
        print(f"  num_envs={args.num_envs}, horizon={args.horizon}, settle={args.settle_steps}")
        print(f"  red_region_center={args.red_region_center_xy}, size={args.red_region_size_xy}")
        print(f"{'='*60}\n")

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            results, markers = rollout_expert_B_with_goal_actions(
                env=env,
                expert=expert,
                task_spec=task_spec,
                rng=rng,
                horizon=args.horizon,
                settle_steps=args.settle_steps,
                markers=markers,
            )
            batch_completed = 0
            batch_success = 0
            for episode_dict, expert_completed_flag in results:
                if expert_completed_flag:
                    batch_completed += 1
                    episodes.append(episode_dict)
                    if episode_dict["success"]:
                        batch_success += 1
                    if len(episodes) >= args.num_episodes:
                        break

            elapsed = time.time() - start_time
            total_attempts = batch_count * args.num_envs
            rate = total_attempts / elapsed if elapsed > 0 else 0
            print(
                f"Batch {batch_count:3d} | Saved: {len(episodes)}/{args.num_episodes} | "
                f"This batch: {batch_completed}/{args.num_envs} completed, {batch_success} success | "
                f"Rate: {rate:.1f} ep/s"
            )

        # ---- Summary ----
        elapsed = time.time() - start_time
        success_count = sum(1 for ep in episodes if ep["success"])
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Saved: {len(episodes)}, Success: {success_count} ({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)
        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
