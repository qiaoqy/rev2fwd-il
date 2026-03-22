#!/usr/bin/env python3
"""Step 6: Cyclic A→B evaluation with hard-reset (for iterative data collection).

Runs N A→B cycles.  After every task (success or failure) the environment is
hard-reset and the object is teleported to the correct starting pose for the
next task.  Successful rollout episodes are saved for later finetune.

Mode 3 only (red rectangle region).  DDIM policies assumed.

Usage:
    conda activate rev2fwd_il
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/6_eval_cyclic.py \\
        --policy_A weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --policy_B weights/PP_B/checkpoints/checkpoints/last/pretrained_model \\
        --out_A data/exp_new/iter1_collect_A.npz \\
        --out_B data/exp_new/iter1_collect_B.npz \\
        --num_cycles 50 --headless
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
        description="Cyclic A→B evaluation with hard-reset recovery (Mode 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Policies
    parser.add_argument("--policy_A", type=str, required=True)
    parser.add_argument("--policy_B", type=str, required=True)

    # Output
    parser.add_argument("--out_A", type=str, required=True,
                        help="Task A successful rollout NPZ.")
    parser.add_argument("--out_B", type=str, required=True,
                        help="Task B successful rollout NPZ.")

    # Test parameters
    parser.add_argument("--num_cycles", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1500,
                        help="Max steps per task (default: 1500).")
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=8,
                        help="Inference action steps (default: 8, see README §1.4).")

    # Region (Mode 3 fixed defaults)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.3, 0.3])

    # Video
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=30)

    # Environment
    parser.add_argument("--task", type=str,
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

        # Load utilities from 6_test_alternating.py
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

        # ---- Load policy configs ----
        config_A = load_policy_config(args.policy_A)
        config_B = load_policy_config(args.policy_B)

        # ---- Create environment ----
        env = make_env_with_camera(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        # ---- Load policies ----
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

        # ---- Build tester (Mode 3 hardcoded) ----
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

        # ---- Helper resets ----
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
            policy_A.reset()

        def _hard_reset_for_task_B():
            env.reset()
            pre_position_gripper_down(env)
            if tester.current_place_xy is not None:
                tester._update_place_marker(tester.current_place_xy)
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
            policy_B.reset()

        # Teleport object for first Task A
        init_pose = torch.tensor(
            [first_place_xy[0], first_place_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        teleport_object_to_pose(env, init_pose, name="object")
        ee_hold = get_ee_pose_w(env)
        hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
        hold_action[0, :7] = ee_hold[0, :7]
        hold_action[0, 7] = 1.0
        for _ in range(10):
            env.step(hold_action)

        # ---- Main loop ----
        results_A, results_B = [], []
        start_time = time.time()

        for cycle in range(args.num_cycles):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{args.num_cycles}")
            print(f"{'='*50}")

            # Task A
            ep_A, success_A = tester.run_task_A()
            tester.episodes_A.append(ep_A)
            results_A.append(success_A)
            print(f"  Task A: {'SUCCESS' if success_A else 'FAILED'}")

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

            # Task B
            ep_B, success_B = tester.run_task_B()
            tester.episodes_B.append(ep_B)
            results_B.append(success_B)
            print(f"  Task B: {'SUCCESS' if success_B else 'FAILED'}")

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

            a_rate = sum(results_A) / len(results_A) * 100
            b_rate = sum(results_B) / len(results_B) * 100
            print(f"  Running rates: A={a_rate:.1f}% ({sum(results_A)}/{len(results_A)})  "
                  f"B={b_rate:.1f}% ({sum(results_B)}/{len(results_B)})")

        elapsed = time.time() - start_time
        n_success_A = sum(results_A)
        n_success_B = sum(results_B)
        rate_A = n_success_A / len(results_A) if results_A else 0.0
        rate_B = n_success_B / len(results_B) if results_B else 0.0

        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"  Cycles:         {args.num_cycles}")
        print(f"  Task A success: {n_success_A}/{len(results_A)} = {rate_A:.1%}")
        print(f"  Task B success: {n_success_B}/{len(results_B)} = {rate_B:.1%}")
        print(f"  Time:           {elapsed:.1f}s")

        # Save successful rollout data
        tester.save_data(args.out_A, args.out_B)

        # Save JSON statistics
        stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_A": args.policy_A,
                "policy_B": args.policy_B,
                "num_cycles": args.num_cycles,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
            },
            "summary": {
                "total_cycles": args.num_cycles,
                "task_A_success_count": n_success_A,
                "task_B_success_count": n_success_B,
                "task_A_success_rate": rate_A,
                "task_B_success_rate": rate_B,
                "total_elapsed_seconds": elapsed,
            },
            "per_cycle_results": [
                {"cycle": i + 1,
                 "task_A_success": results_A[i],
                 "task_B_success": results_B[i]}
                for i in range(len(results_A))
            ],
        }

        stats_path = Path(args.out_A).with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to: {stats_path}")

        if args.save_video:
            video_path = Path(args.video_path or args.out_A).with_suffix(".mp4")
            tester.save_video(str(video_path), fps=args.video_fps)

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
