#!/usr/bin/env python3
"""Step 9: Collect successful rollout trajectories from a single policy.

Runs a policy in fair-test mode (hard reset each episode), keeps only
successful episodes, and saves them to an NPZ file.  Continues until the
target number of successes is reached or max attempts is exceeded.

The output NPZ format is identical to the one produced by
``6_eval_cyclic.py`` so downstream scripts (``2_reverse_to_task_A.py``,
``4_train.py``, etc.) can consume it directly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/9_collect_rollout.py \
        --policy data/exp21/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
        --task A \
        --target_successes 200 \
        --max_attempts 500 \
        --horizon 1200 \
        --out_npz data/exp21/rollout_A_200.npz \
        --out_stats data/exp21/rollout_A_200.stats.json \
        --headless
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect successful rollout trajectories (fair test mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--policy", type=str, required=True,
                        help="Policy checkpoint path.")
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"],
                        help="Which task to evaluate: A or B.")
    parser.add_argument("--out_npz", type=str, required=True,
                        help="Output NPZ path for successful episodes.")
    parser.add_argument("--out_stats", type=str, default=None,
                        help="Output statistics JSON path. Default: <out_npz>.stats.json")

    parser.add_argument("--target_successes", type=int, default=200,
                        help="Number of successful episodes to collect.")
    parser.add_argument("--max_attempts", type=int, default=500,
                        help="Maximum rollout attempts (to prevent infinite loop).")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=16)

    # Region (Mode 3)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.3, 0.3])

    # Environment
    parser.add_argument("--env_task", type=str,
                        default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


def main() -> None:
    args = _parse_args()

    if args.out_stats is None:
        args.out_stats = str(Path(args.out_npz).with_suffix("")) + ".stats.json"

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import importlib.util
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_ee_pose_w,
            teleport_object_to_pose,
            pre_position_gripper_down,
        )

        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).resolve().parent.parent
                / "scripts_pick_place" / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        make_env_with_camera = _alt_mod.make_env_with_camera
        load_policy_config = _alt_mod.load_policy_config
        load_policy_auto = _alt_mod.load_policy_auto
        AlternatingTester = _alt_mod.AlternatingTester
        create_target_markers = _alt_mod.create_target_markers

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load policy ----
        config = load_policy_config(args.policy)

        env = make_env_with_camera(
            task_id=args.env_task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        policy, preproc, postproc, _, n_act = load_policy_auto(
            args.policy, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy.eval()

        # AlternatingTester needs both policy slots; fill unused with same policy
        tester = AlternatingTester(
            env=env,
            policy_A=policy, preprocessor_A=preproc, postprocessor_A=postproc,
            policy_B=policy, preprocessor_B=preproc, postprocessor_B=postproc,
            n_action_steps_A=n_act, n_action_steps_B=n_act,
            goal_xy=tuple(args.goal_xy),
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
            fix_red_marker_pose=True,
            taskA_source_mode="red_region",
            taskB_target_mode="red_region",
            red_region_center_xy=tuple(args.red_region_center_xy),
            red_region_size_xy=tuple(args.red_region_size_xy),
            height_threshold=0.15,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=config["has_wrist"],
            has_wrist_B=config["has_wrist"],
            include_obj_pose_A=config["include_obj_pose"],
            include_obj_pose_B=config["include_obj_pose"],
            include_gripper_A=config["include_gripper"],
            include_gripper_B=config["include_gripper"],
        )

        goal_xy = np.array(args.goal_xy)

        # ---- Initial setup ----
        env.reset()
        pre_position_gripper_down(env)

        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        tester.place_markers = place_markers
        tester.goal_markers = goal_markers
        tester.marker_z = marker_z

        first_place_xy = tester._sample_taskA_source_target()
        tester.current_place_xy = first_place_xy
        tester._update_place_marker(first_place_xy)

        # ---- Hard reset helpers ----
        def _hard_reset_for_task_A():
            env.reset()
            pre_position_gripper_down(env)
            if tester.current_place_xy is not None:
                tester._update_place_marker(tester.current_place_xy)
            rand_xy = tester._sample_taskA_source_target()
            obj_pose = torch.tensor(
                [rand_xy[0], rand_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7] = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy.reset()

        def _hard_reset_for_task_B():
            env.reset()
            pre_position_gripper_down(env)
            new_place_xy = tester._sample_new_place_target()
            tester.current_place_xy = new_place_xy
            tester._update_place_marker(new_place_xy)
            obj_pose = torch.tensor(
                [goal_xy[0], goal_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7] = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy.reset()

        # ---- Collect successful episodes ----
        successful_episodes = []
        all_results = []
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Rollout Collection: Task {args.task}")
        print(f"  Target: {args.target_successes} successes")
        print(f"  Max attempts: {args.max_attempts}")
        print(f"  Horizon: {args.horizon}")
        print(f"{'='*60}\n")

        for attempt in range(1, args.max_attempts + 1):
            if len(successful_episodes) >= args.target_successes:
                break

            if args.task == "A":
                _hard_reset_for_task_A()
                ep, success = tester.run_task_A()
            else:
                _hard_reset_for_task_B()
                ep, success = tester.run_task_B()

            steps = len(ep.get("images", []))
            all_results.append(success)

            if success:
                successful_episodes.append(ep)

            running_rate = sum(all_results) / len(all_results) * 100
            status = "SUCCESS" if success else "FAILED"
            collected = len(successful_episodes)
            print(f"  [{attempt}/{args.max_attempts}] {status} "
                  f"({steps} steps) — collected: "
                  f"{collected}/{args.target_successes} "
                  f"(running rate: {running_rate:.1f}%)")

        elapsed = time.time() - start_time
        n_collected = len(successful_episodes)
        n_attempts = len(all_results)
        success_rate = sum(all_results) / n_attempts if n_attempts > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Collection Complete")
        print(f"{'='*60}")
        print(f"  Collected:    {n_collected}/{args.target_successes} successful episodes")
        print(f"  Attempts:     {n_attempts}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Time:         {elapsed:.1f}s")

        # ---- Save NPZ ----
        if successful_episodes:
            out_npz = Path(args.out_npz)
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_npz,
                episodes=np.array(successful_episodes, dtype=object),
            )
            file_mb = out_npz.stat().st_size / (1024 * 1024)
            total_frames = sum(len(ep["images"]) for ep in successful_episodes)
            avg_len = total_frames / n_collected
            print(f"  Saved {n_collected} episodes to {out_npz}  ({file_mb:.1f} MB)")
            print(f"  Total frames: {total_frames}, avg episode length: {avg_len:.1f}")
        else:
            total_frames = 0
            avg_len = 0
            print("  WARNING: No successful episodes collected!")

        # ---- Save stats ----
        stats = {
            "task": args.task,
            "target_successes": args.target_successes,
            "collected_successes": n_collected,
            "total_attempts": n_attempts,
            "success_rate": success_rate,
            "total_frames": total_frames,
            "avg_episode_length": avg_len,
            "elapsed_sec": elapsed,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy": args.policy,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
                "seed": args.seed,
            },
        }

        out_stats = Path(args.out_stats)
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(out_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to: {out_stats}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
