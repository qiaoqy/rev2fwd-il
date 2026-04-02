#!/usr/bin/env python3
"""Exp31: Collect Task B data with symmetric green/red rectangle markers.

The green circle start marker is replaced by a green rectangle (same size as
the red rectangle) so that start and end zones are visually symmetric about y=0.
The cube spawns at a **random position** within the green region each batch,
mirroring the per-env random sampling in the red region.

  Green rectangle (goal / cube spawn):  center = (0.5, -0.2), size = 0.3×0.3
  Red   rectangle (place / target):     center = (0.5, +0.2), size = 0.3×0.3

Both zones use identical randomization (region shrunk by cube_half_size=0.02m)
and the same success_radius (0.03m) for symmetric Task A/B evaluation.

Everything else (FSM expert, data format, action convention) is identical to
``1_collect_task_B.py``.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_task_B_symmetric.py \
        --out data/pick_place_isaac_lab_simulation/exp31/task_B_1.npz \
        --num_episodes 1 --headless
"""

from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Task B data with symmetric green/red rectangle markers.",
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

    # Symmetric region parameters
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2],
                        help="Green zone center (cube spawn region center).")
    parser.add_argument("--red_region_center_xy", type=float, nargs=2, default=[0.5, 0.2],
                        help="Red zone center (place target region).")
    parser.add_argument("--region_size_xy", type=float, nargs=2, default=[0.3, 0.3],
                        help="Size of both green and red rectangles (shared).")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def create_symmetric_markers(num_envs, device, region_size_xy):
    """Create symmetric green (goal) and red (place) rectangle markers.

    Both markers are flat cuboids of the same size, differing only in colour.

    Returns:
        (place_markers, goal_markers, marker_z)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    marker_height = 0.002
    table_z = 0.0
    marker_z = table_z + marker_height / 2 + 0.001

    sx, sy = float(region_size_xy[0]), float(region_size_xy[1])

    # Red rectangle — place / target zone
    place_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/StartMarkers",
        markers={
            "start": sim_utils.CuboidCfg(
                size=(sx, sy, marker_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        },
    )
    place_markers = VisualizationMarkers(place_marker_cfg)

    # Green rectangle — goal / cube-spawn zone
    goal_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.CuboidCfg(
                size=(sx, sy, marker_height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            ),
        },
    )
    goal_markers = VisualizationMarkers(goal_marker_cfg)

    return place_markers, goal_markers, marker_z


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

    # Reuse env / rollout utilities from the original collection script
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

        # ---- Task spec (symmetric regions) ----
        task_spec = PickPlaceTaskSpec(
            goal_xy=tuple(args.goal_xy),
            hover_z=0.25,
            grasp_z_offset=0.0,
            success_radius=0.03,
            settle_steps=10,
        )
        task_spec.taskB_target_mode = "red_region"
        task_spec.red_region_center_xy = tuple(args.red_region_center_xy)
        task_spec.red_region_size_xy = tuple(args.region_size_xy)
        # Marker shape/size — still passed so rollout won't create its own
        task_spec.red_marker_shape = "rectangle"
        task_spec.red_marker_size_xy = tuple(args.region_size_xy)
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

        # ---- Pre-create symmetric markers ----
        markers = create_symmetric_markers(
            args.num_envs, device, args.region_size_xy,
        )

        # ---- Collection loop ----
        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        start_time = time.time()

        # Random spawn setup: sample goal_xy within green region each batch
        green_cx, green_cy = float(args.goal_xy[0]), float(args.goal_xy[1])
        green_sx, green_sy = float(args.region_size_xy[0]), float(args.region_size_xy[1])
        cube_half_size = 0.02  # DexCube default 0.05m * scale 0.8 = 0.04m edge, half = 0.02m
        green_sample_half_x = max(green_sx * 0.5 - cube_half_size, 0.0)
        green_sample_half_y = max(green_sy * 0.5 - cube_half_size, 0.0)

        print(f"\n{'='*60}")
        print(f"Exp31: Collecting {args.num_episodes} Task B episodes (symmetric markers)")
        print(f"  goal (green) center=({green_cx}, {green_cy}), size={args.region_size_xy}")
        print(f"  place (red)  center={args.red_region_center_xy}, size={args.region_size_xy}")
        print(f"  random spawn: x=[{green_cx-green_sample_half_x:.4f}, {green_cx+green_sample_half_x:.4f}], "
              f"y=[{green_cy-green_sample_half_y:.4f}, {green_cy+green_sample_half_y:.4f}]")
        print(f"  num_envs={args.num_envs}, horizon={args.horizon}")
        print(f"{'='*60}\n")

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1

            # Randomize cube spawn within green region (per-batch)
            rand_goal_x = float(rng.uniform(green_cx - green_sample_half_x,
                                            green_cx + green_sample_half_x))
            rand_goal_y = float(rng.uniform(green_cy - green_sample_half_y,
                                            green_cy + green_sample_half_y))
            task_spec.goal_xy = (rand_goal_x, rand_goal_y)

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
        print(f"Saved: {len(episodes)}, Success: {success_count} "
              f"({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)
        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
