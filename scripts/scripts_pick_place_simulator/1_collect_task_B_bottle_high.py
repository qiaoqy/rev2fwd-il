#!/usr/bin/env python3
"""Collect Task B data for Bottle with higher hover height (Exp47).

Based on 1_collect_task_B_multi_obj.py, using PickPlaceExpertBottle which has
episode_hover_z ~ U(0.4, 0.45) instead of U(0.2, 0.3).

Action convention: action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]

Wrist camera is always enabled.

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_task_B_bottle_high.py \
        --object_type bottle \
        --out data/pick_place_isaac_lab_simulation/exp47/task_B_demo_p0.npz \
        --num_episodes 25 --num_envs 1 --horizon 600 --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime


def _ts() -> str:
    """Return current timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S")

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
        description="Collect Task B data for Bottle with higher hover (Exp47).",
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
    from rev2fwd_il.experts.pickplace_expert_bottle import PickPlaceExpertBottle
    from rev2fwd_il.utils.seed import set_seed

    # Reuse env / marker utilities from the original collection script
    _orig_spec = importlib.util.spec_from_file_location(
        "collect_orig",
        str(Path(__file__).resolve().parent.parent / "scripts_pick_place" / "1_collect_data_pick_place.py"),
    )
    _orig = importlib.util.module_from_spec(_orig_spec)
    _orig_spec.loader.exec_module(_orig)

    make_env_with_camera = _orig.make_env_with_camera
    add_camera_to_env_cfg = _orig.add_camera_to_env_cfg
    rollout_expert_B_with_goal_actions = _orig.rollout_expert_B_with_goal_actions
    save_episodes_with_goal_actions = _orig.save_episodes_with_goal_actions

    import gymnasium as gym

    def _make_env_for_object(object_type, num_envs, device, use_fabric,
                             image_width, image_height, episode_length_s,
                             disable_terminations):
        """Create environment, using cube env config as base and swapping object if needed."""
        if object_type == "cube":
            return make_env_with_camera(
                task_id=TASK_IDS["cube"],
                num_envs=num_envs, device=device, use_fabric=use_fabric,
                image_width=image_width, image_height=image_height,
                episode_length_s=episode_length_s,
                disable_terminations=disable_terminations,
            )

        # For non-cube objects: start from cube config, swap the object
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from isaaclab.assets import RigidObjectCfg

        obj_cfg = get_object_config(object_type)
        cube_task_id = TASK_IDS["cube"]

        print(f"[{_ts()}] Parsing base cube config...", flush=True)
        env_cfg = parse_env_cfg(cube_task_id, device=device,
                                num_envs=int(num_envs), use_fabric=bool(use_fabric))

        # Swap the object spawn config
        env_cfg.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=list(obj_cfg.init_pos),
                rot=list(obj_cfg.init_rot),
            ),
            spawn=obj_cfg.spawn_cfg_fn(),
        )
        print(f"[{_ts()}] Object swapped to {object_type}.", flush=True)

        # USD-file objects (e.g. gear) have internal body names that differ
        # from the default "Object" → drop body_names from reset_object_position
        # so the regex matcher doesn't fail.
        from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg as _UsdCheck
        if isinstance(env_cfg.scene.object.spawn, _UsdCheck):
            from isaaclab.managers import EventTermCfg as EventTerm
            from isaaclab.managers import SceneEntityCfg
            import isaaclab_tasks.manager_based.manipulation.lift.mdp as _mdp
            env_cfg.events.reset_object_position = EventTerm(
                func=_mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("object"),
                },
            )
            print(f"[{_ts()}] USD object detected — reset_object_position body_names dropped.", flush=True)

        if episode_length_s is not None:
            env_cfg.episode_length_s = episode_length_s
        if disable_terminations:
            if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'time_out'):
                env_cfg.terminations.time_out = None
            if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'object_dropping'):
                env_cfg.terminations.object_dropping = None

        add_camera_to_env_cfg(env_cfg, image_width, image_height)

        print(f"[{_ts()}] Creating gym environment...", flush=True)
        env = gym.make(cube_task_id, cfg=env_cfg)
        return env

    print(f"[{_ts()}] Helper functions loaded.", flush=True)

    try:
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        print(f"[{_ts()}] Seeds set. Starting collection for '{args.object_type}'...", flush=True)

        # ---- Object config ----
        obj_cfg = get_object_config(args.object_type)
        task_id = TASK_IDS[args.object_type]
        print(f"[{_ts()}] Object config loaded: {obj_cfg.description}", flush=True)

        # ---- Create environment ----
        print(f"[{_ts()}] Creating environment (object_type={args.object_type})...", flush=True)
        env = _make_env_for_object(
            object_type=args.object_type,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,
            disable_terminations=True,
        )
        device = env.unwrapped.device
        print(f"[{_ts()}] Environment created. device={device}", flush=True)

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

        # ---- Expert (PickPlaceExpertBottle with higher hover_z) ----
        expert = PickPlaceExpertBottle(
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

        print(f"\n[{_ts()}] {'='*60}")
        print(f"[{_ts()}] Collecting {args.num_episodes} Task B episodes (Exp47 — high hover)")
        print(f"[{_ts()}]   object_type={args.object_type}")
        print(f"[{_ts()}]   task_id={task_id}")
        print(f"[{_ts()}]   object_height={obj_cfg.object_height}, grasp_z_offset={obj_cfg.grasp_z_offset}")
        print(f"[{_ts()}]   hover_z={obj_cfg.hover_z}, release_z_offset={obj_cfg.release_z_offset}")
        print(f"[{_ts()}]   episode_hover_z ~ U(0.4, 0.45) [Exp47 higher transport]")
        print(f"[{_ts()}]   place_z={obj_cfg.place_z}, goal_z={obj_cfg.goal_z}")
        print(f"[{_ts()}]   mesh_origin_offset={obj_cfg.mesh_origin_offset}")
        print(f"[{_ts()}]   num_envs={args.num_envs}, horizon={args.horizon}, settle={args.settle_steps}")
        print(f"[{_ts()}]   red_region_center={args.red_region_center_xy}, size={args.red_region_size_xy}")
        print(f"[{_ts()}] {'='*60}\n", flush=True)

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            print(f"[{_ts()}] Starting batch {batch_count}/{max_batches}...", flush=True)
            results, markers = rollout_expert_B_with_goal_actions(
                env=env,
                expert=expert,
                task_spec=task_spec,
                rng=rng,
                horizon=args.horizon,
                settle_steps=args.settle_steps,
                markers=markers,
                place_z=obj_cfg.place_z,
                goal_z=obj_cfg.goal_z,
                mesh_origin_offset=obj_cfg.mesh_origin_offset,
            )
            print(f"[{_ts()}] Batch {batch_count} rollout done. Processing results...", flush=True)
            batch_completed = 0
            batch_success = 0
            for episode_dict, expert_completed_flag in results:
                # Save ALL episodes (including incomplete ones) for debugging
                episode_dict["object_type"] = args.object_type
                episode_dict["expert_completed"] = bool(expert_completed_flag)
                episodes.append(episode_dict)
                if expert_completed_flag:
                    batch_completed += 1
                if episode_dict["success"]:
                    batch_success += 1
                if len(episodes) >= args.num_episodes:
                    break

            elapsed = time.time() - start_time
            total_attempts = batch_count * args.num_envs
            rate = total_attempts / elapsed if elapsed > 0 else 0
            print(
                f"[{_ts()}] Batch {batch_count:3d} | Saved: {len(episodes)}/{args.num_episodes} | "
                f"This batch: {batch_completed}/{args.num_envs} completed, {batch_success} success | "
                f"Rate: {rate:.1f} ep/s",
                flush=True,
            )

        # ---- Summary ----
        elapsed = time.time() - start_time
        success_count = sum(1 for ep in episodes if ep["success"])
        completed_count = sum(1 for ep in episodes if ep.get("expert_completed", True))
        print(f"\n[{_ts()}] {'='*60}")
        print(f"[{_ts()}] Collection finished in {elapsed:.1f}s  [{args.object_type}] (Exp47 high hover)")
        print(f"[{_ts()}] Total: {len(episodes)}, Completed: {completed_count}, Success: {success_count} ({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        episodes = episodes[:args.num_episodes]
        print(f"[{_ts()}] Saving {len(episodes)} episodes to {args.out}...", flush=True)
        save_episodes_with_goal_actions(args.out, episodes)
        print(f"[{_ts()}] Save done. Closing environment...", flush=True)
        env.close()
        print(f"[{_ts()}] Environment closed.", flush=True)

    except Exception:
        import traceback
        print(f"\n[{_ts()}] *** EXCEPTION ***", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        print(f"[{_ts()}] Shutting down simulation app...", flush=True)
        simulation_app.close()
        print("Simulation app closed. Exiting.", flush=True)


if __name__ == "__main__":
    main()
