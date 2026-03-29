#!/usr/bin/env python3
"""Collect Task B data using FSM expert for multiple object types (Exp28).

This script generalises 1_collect_task_B.py to work with any object registered
in the object_registry.  It picks the object from the green goal position and
places it at a random position inside the red rectangular region.

Action convention: action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]

Wrist camera is always enabled.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_task_B_multi_obj.py \
        --object_type cylinder \
        --out data/pick_place_isaac_lab_simulation/exp28/task_B_cylinder_100.npz \
        --num_episodes 100 --headless
"""

from __future__ import annotations

import argparse
import time

# Map from object_type to registered gym task id
TASK_IDS = {
    "cube": "Isaac-Lift-Cube-Franka-IK-Abs-v0",
    "cylinder": "Isaac-Lift-Cylinder-Franka-IK-Abs-v0",
    "sphere": "Isaac-Lift-Sphere-Franka-IK-Abs-v0",
    "bottle": "Isaac-Lift-Bottle-Franka-IK-Abs-v0",
    "gear": "Isaac-Lift-Gear-Franka-IK-Abs-v0",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Task B data with FSM expert for multi-object (Exp28).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--object_type", type=str, required=True,
        choices=list(TASK_IDS.keys()),
        help="Object type to use.",
    )
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")

    # Region parameters
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
    from rev2fwd_il.sim.object_registry import get_object_config
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

        # ---- Object config ----
        obj_cfg = get_object_config(args.object_type)
        task_id = TASK_IDS[args.object_type]

        # ---- Create environment ----
        env = make_env_with_camera(
            task_id=task_id,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,
            disable_terminations=True,
        )
        device = env.unwrapped.device

        # ---- Task spec (use object-specific parameters) ----
        task_spec = PickPlaceTaskSpec(
            goal_xy=tuple(args.goal_xy),
            hover_z=obj_cfg.hover_z,
            grasp_z_offset=obj_cfg.grasp_z_offset,
            success_radius=obj_cfg.success_radius,
            settle_steps=obj_cfg.wait_steps,
            object_height=obj_cfg.object_height,
        )
        task_spec.taskB_target_mode = "red_region"
        task_spec.red_region_center_xy = tuple(args.red_region_center_xy)
        task_spec.red_region_size_xy = tuple(args.red_region_size_xy)
        task_spec.red_marker_shape = "rectangle"
        task_spec.red_marker_size_xy = tuple(args.red_region_size_xy)
        task_spec.fix_red_marker_pose = True

        # ---- Expert B (with object-specific parameters) ----
        expert = PickPlaceExpertB(
            num_envs=args.num_envs,
            device=device,
            hover_z=obj_cfg.hover_z,
            grasp_z_offset=obj_cfg.grasp_z_offset,
            release_z_offset=obj_cfg.release_z_offset,
            position_threshold=obj_cfg.position_threshold,
            wait_steps=obj_cfg.wait_steps,
        )

        # ---- Collection loop ----
        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        markers = None
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} Task B episodes")
        print(f"  object_type={args.object_type}")
        print(f"  task_id={task_id}")
        print(f"  object_height={obj_cfg.object_height}, grasp_z_offset={obj_cfg.grasp_z_offset}")
        print(f"  hover_z={obj_cfg.hover_z}, release_z_offset={obj_cfg.release_z_offset}")
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
                    # Tag episode with object type
                    episode_dict["object_type"] = args.object_type
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
        print(f"Collection finished in {elapsed:.1f}s  [{args.object_type}]")
        print(f"Saved: {len(episodes)}, Success: {success_count} ({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)
        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
