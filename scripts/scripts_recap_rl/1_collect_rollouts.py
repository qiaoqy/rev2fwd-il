#!/usr/bin/env python3
"""Step 1: Collect rollout data for RECAP RL (alternating A→B cycles).

Runs N alternating A→B cycles using both policy_A and policy_B.
Saves ALL episodes (success + failure) — failures are essential for training
the value function to distinguish good from bad states.

Designed for 10-GPU parallel collection: each GPU runs --num_cycles cycles,
outputs per-GPU NPZ shards which are later merged.

Usage:
    # Single GPU: 10 A→B cycles = 10 Task A + 10 Task B episodes
    CUDA_VISIBLE_DEVICES=3 python scripts/scripts_recap_rl/1_collect_rollouts.py \\
        --policy_A exp19/pretrain_A \\
        --policy_B exp19/pretrain_B \\
        --out_A exp19/rollouts_gpu3_A.npz \\
        --out_B exp19/rollouts_gpu3_B.npz \\
        --num_cycles 10 --seed 3 --headless
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
        description="Collect alternating A→B rollouts for RECAP RL "
                    "(all episodes, success + failure).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy_A", type=str, required=True,
                        help="Task A policy checkpoint.")
    parser.add_argument("--policy_B", type=str, required=True,
                        help="Task B policy checkpoint.")
    parser.add_argument("--out_A", type=str, required=True,
                        help="Output NPZ for Task A episodes.")
    parser.add_argument("--out_B", type=str, required=True,
                        help="Output NPZ for Task B episodes.")
    parser.add_argument("--num_cycles", type=int, default=10,
                        help="Number of A→B cycles (default: 10).")
    parser.add_argument("--horizon", type=int, default=1200,
                        help="Max steps per task episode.")
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
    parser.add_argument("--seed", type=int, default=0)

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

        make_env_with_camera   = _alt_mod.make_env_with_camera
        load_policy_config     = _alt_mod.load_policy_config
        load_policy_auto       = _alt_mod.load_policy_auto
        AlternatingTester      = _alt_mod.AlternatingTester
        create_target_markers  = _alt_mod.create_target_markers

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load both policy configs ----
        config_A = load_policy_config(args.policy_A)
        config_B = load_policy_config(args.policy_B)

        env = make_env_with_camera(
            task_id=args.env_task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        # ---- Load both policies ----
        policy_A, preproc_A, postproc_A, _, n_act_A = load_policy_auto(
            args.policy_A, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_A.eval()

        policy_B, preproc_B, postproc_B, _, n_act_B = load_policy_auto(
            args.policy_B, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_B.eval()

        # ---- Build tester ----
        tester = AlternatingTester(
            env=env,
            policy_A=policy_A, preprocessor_A=preproc_A, postprocessor_A=postproc_A,
            policy_B=policy_B, preprocessor_B=preproc_B, postprocessor_B=postproc_B,
            n_action_steps_A=n_act_A, n_action_steps_B=n_act_B,
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
            has_wrist_A=config_A["has_wrist"],
            has_wrist_B=config_B["has_wrist"],
            include_obj_pose_A=config_A["include_obj_pose"],
            include_obj_pose_B=config_B["include_obj_pose"],
            include_gripper_A=config_A["include_gripper"],
            include_gripper_B=config_B["include_gripper"],
        )

        goal_xy = np.array(args.goal_xy)
        env.reset()
        pre_position_gripper_down(env)

        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        tester.place_markers = place_markers
        tester.goal_markers  = goal_markers
        tester.marker_z      = marker_z

        first_place_xy = tester._sample_taskA_source_target()
        tester.current_place_xy = first_place_xy
        tester._update_place_marker(first_place_xy)

        # ---- Hard-reset helpers ----
        def _hard_reset_for_task_A():
            env.reset()
            pre_position_gripper_down(env)
            if tester.current_place_xy is not None:
                tester._update_place_marker(tester.current_place_xy)
            rand_xy = tester._sample_taskA_source_target()
            obj_pose = torch.tensor(
                [rand_xy[0], rand_xy[1], 0.022, 1., 0., 0., 0.],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7]  = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy_A.reset()

        def _hard_reset_for_task_B():
            env.reset()
            pre_position_gripper_down(env)
            new_place_xy = tester._sample_new_place_target()
            tester.current_place_xy = new_place_xy
            tester._update_place_marker(new_place_xy)
            obj_pose = torch.tensor(
                [goal_xy[0], goal_xy[1], 0.022, 1., 0., 0., 0.],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7]  = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy_B.reset()

        # ---- Initial teleport for first Task A ----
        init_pose = torch.tensor(
            [first_place_xy[0], first_place_xy[1], 0.022, 1., 0., 0., 0.],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        teleport_object_to_pose(env, init_pose, name="object")
        ee_hold = get_ee_pose_w(env)
        hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
        hold_action[0, :7] = ee_hold[0, :7]
        hold_action[0, 7] = 1.0
        for _ in range(10):
            env.step(hold_action)

        # ---- Alternating A→B collection loop ----
        print(f"\n{'='*60}")
        print(f"RECAP Alternating Collection: {args.num_cycles} A→B cycles")
        print(f"  = {args.num_cycles} Task A + {args.num_cycles} Task B episodes")
        print(f"  Saving ALL episodes (success + failure)")
        print(f"  Seed: {args.seed}")
        print(f"{'='*60}\n")

        episodes_A, episodes_B = [], []
        results_A, results_B = [], []
        start_time = time.time()

        for cycle in range(args.num_cycles):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{args.num_cycles}")
            print(f"{'='*50}")

            # ---- Task A ----
            ep_A, success_A = tester.run_task_A()
            ep_A["success"] = bool(success_A)
            episodes_A.append(ep_A)
            results_A.append(success_A)
            steps_A = len(ep_A.get("images", []))
            print(f"  Task A: {'SUCCESS' if success_A else 'FAILED '} "
                  f"({steps_A:4d} steps) | "
                  f"A rate: {sum(results_A)/len(results_A)*100:.1f}%")

            # Transition after Task A success
            if success_A:
                new_place_xy = tester._sample_new_place_target()
                tester._update_place_marker(new_place_xy)
                tester._run_transition(
                    n_frames=100,
                    policy=tester.policy_A, preprocessor=tester.preprocessor_A,
                    postprocessor=tester.postprocessor_A,
                    n_action_steps=tester.n_action_steps_A,
                    include_obj_pose=tester.include_obj_pose_A,
                    include_gripper=tester.include_gripper_A,
                    has_wrist=tester.has_wrist_A,
                    task_name="Task A (transition)",
                )
            else:
                new_place_xy = tester._sample_new_place_target()
                tester._update_place_marker(new_place_xy)

            _hard_reset_for_task_B()

            # ---- Task B ----
            ep_B, success_B = tester.run_task_B()
            ep_B["success"] = bool(success_B)
            episodes_B.append(ep_B)
            results_B.append(success_B)
            steps_B = len(ep_B.get("images", []))
            print(f"  Task B: {'SUCCESS' if success_B else 'FAILED '} "
                  f"({steps_B:4d} steps) | "
                  f"B rate: {sum(results_B)/len(results_B)*100:.1f}%")

            # Transition after Task B success
            if success_B:
                tester._run_transition(
                    n_frames=100,
                    policy=tester.policy_B, preprocessor=tester.preprocessor_B,
                    postprocessor=tester.postprocessor_B,
                    n_action_steps=tester.n_action_steps_B,
                    include_obj_pose=tester.include_obj_pose_B,
                    include_gripper=tester.include_gripper_B,
                    has_wrist=tester.has_wrist_B,
                    task_name="Task B (transition)",
                )

            _hard_reset_for_task_A()

        elapsed = time.time() - start_time
        n_suc_A = sum(results_A)
        n_suc_B = sum(results_B)

        print(f"\n{'='*60}")
        print(f"Collection complete ({elapsed:.1f}s)")
        print(f"  Task A: {n_suc_A}/{len(results_A)} = "
              f"{n_suc_A/len(results_A):.1%} success")
        print(f"  Task B: {n_suc_B}/{len(results_B)} = "
              f"{n_suc_B/len(results_B):.1%} success")
        print(f"{'='*60}")

        # ---- Save all episodes ----
        for label, episodes, out_path_str, results, n_suc in [
            ("A", episodes_A, args.out_A, results_A, n_suc_A),
            ("B", episodes_B, args.out_B, results_B, n_suc_B),
        ]:
            out_path = Path(out_path_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(out_path),
                episodes=np.array(episodes, dtype=object),
            )
            print(f"Saved {len(episodes)} Task {label} episodes "
                  f"({n_suc} success) to: {out_path}")

            stats = {
                "task": label,
                "num_episodes": len(episodes),
                "num_success": n_suc,
                "success_rate": n_suc / len(results) if results else 0.0,
                "per_episode": [
                    {"episode": i, "success": bool(results[i]),
                     "steps": len(episodes[i].get("images", []))}
                    for i in range(len(episodes))
                ],
                "config": {
                    "policy_A": args.policy_A,
                    "policy_B": args.policy_B,
                    "num_cycles": args.num_cycles,
                    "horizon": args.horizon,
                    "n_action_steps": args.n_action_steps,
                    "seed": args.seed,
                },
                "timestamp": datetime.now().isoformat(),
                "elapsed_sec": elapsed,
            }
            stats_path = out_path.with_suffix(".stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
