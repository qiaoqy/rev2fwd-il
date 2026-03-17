#!/usr/bin/env python3
"""Step 10: Independent evaluation of Task A and Task B.

Unlike 9_eval_with_recovery.py which alternates A→B→A→B, this script
evaluates each task INDEPENDENTLY with a full env.reset() before every
single episode. This removes any coupling between the two tasks.

=============================================================================
PURPOSE
=============================================================================
Test whether the task-chaining in the alternating A→B pipeline introduces
a confound (e.g., starting Task B from a slightly imperfect Task A result).

The evaluation runs:
  1.  N independent Task A episodes (reset + teleport obj to random pos)
  2.  N independent Task B episodes (reset + teleport obj to goal pos)

=============================================================================
USAGE
=============================================================================
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/10_eval_independent.py \
    --policy_A runs/.../pretrained_model \
    --policy_B runs/.../pretrained_model \
    --num_episodes 50 --headless
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent per-task evaluation with hard reset before every episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Policy checkpoints
    parser.add_argument("--policy_A", type=str, required=True,
                        help="Path to Task A policy checkpoint.")
    parser.add_argument("--policy_B", type=str, required=True,
                        help="Path to Task B policy checkpoint.")

    # Output
    parser.add_argument("--out", type=str, default="data/eval_independent.stats.json",
                        help="Output path for statistics JSON.")

    # Test parameters
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of independent episodes PER task. Default: 50.")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Maximum steps per episode.")
    parser.add_argument("--height_threshold", type=float, default=0.15)
    parser.add_argument("--distance_threshold", type=float, default=0.05)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, 0.0])
    parser.add_argument(
        "--fixed_start_xy",
        type=float,
        nargs=2,
        default=None,
        help="Optional fixed table XY for all sampled starts/targets.",
    )
    parser.add_argument(
        "--red_marker_shape",
        type=str,
        choices=["circle", "rectangle"],
        default="circle",
        help="Red marker shape. Default keeps legacy circle marker.",
    )
    parser.add_argument(
        "--red_marker_size_xy",
        type=float,
        nargs=2,
        default=None,
        help="Optional red marker size (sx sy) in meters.",
    )
    parser.add_argument(
        "--fix_red_marker_pose",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, red marker stays at region center while target points can still be random.",
    )
    parser.add_argument(
        "--taskA_source_mode",
        type=str,
        choices=["legacy", "green_region", "red_region"],
        default="legacy",
        help="Task A source sampling mode. 'red_region' samples from the red rectangle region.",
    )
    parser.add_argument(
        "--taskB_target_mode",
        type=str,
        choices=["legacy", "red_region"],
        default="legacy",
        help="Task B target sampling mode.",
    )
    parser.add_argument("--red_region_center_xy", type=float, nargs=2, default=None)
    parser.add_argument("--red_region_size_xy", type=float, nargs=2, default=None)
    parser.add_argument("--green_region_center_xy", type=float, nargs=2, default=None)
    parser.add_argument("--green_region_size_xy", type=float, nargs=2, default=None)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--tasks", type=str, default="AB",
                        choices=["A", "B", "AB"],
                        help="Which tasks to evaluate: A, B, or AB (default).")

    # Environment settings
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)

    # Isaac Lab AppLauncher arguments
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
            get_object_pose_w,
            teleport_object_to_pose,
            pre_position_gripper_down,
        )

        # Reuse utilities from 6_test_alternating.py
        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).parent / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        make_env_with_camera = _alt_mod.make_env_with_camera
        load_policy_config = _alt_mod.load_policy_config
        load_policy_auto = _alt_mod.load_policy_auto
        AlternatingTester = _alt_mod.AlternatingTester
        create_target_markers = _alt_mod.create_target_markers
        update_target_markers = _alt_mod.update_target_markers

        set_seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

        # =================================================================
        # Load policy configs
        # =================================================================
        print(f"\n{'='*60}")
        print("Loading policy configurations...")
        print(f"{'='*60}")
        config_A = load_policy_config(args.policy_A)
        config_B = load_policy_config(args.policy_B)

        # =================================================================
        # Create environment
        # =================================================================
        print(f"\n{'='*60}")
        print("Creating environment...")
        print(f"{'='*60}")
        env = make_env_with_camera(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        # =================================================================
        # Load policies
        # =================================================================
        print(f"\n{'='*60}")
        print("Loading policies...")
        print(f"{'='*60}")
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

        # =================================================================
        # Build tester
        # =================================================================
        tester = AlternatingTester(
            env=env,
            policy_A=policy_A, preprocessor_A=preproc_A, postprocessor_A=postproc_A,
            policy_B=policy_B, preprocessor_B=preproc_B, postprocessor_B=postproc_B,
            n_action_steps_A=n_act_A, n_action_steps_B=n_act_B,
            goal_xy=tuple(args.goal_xy),
            fixed_start_xy=(tuple(args.fixed_start_xy) if args.fixed_start_xy is not None else None),
            red_marker_shape=args.red_marker_shape,
            red_marker_size_xy=(tuple(args.red_marker_size_xy) if args.red_marker_size_xy is not None else None),
            fix_red_marker_pose=bool(args.fix_red_marker_pose),
            taskA_source_mode=args.taskA_source_mode,
            taskB_target_mode=args.taskB_target_mode,
            red_region_center_xy=(tuple(args.red_region_center_xy) if args.red_region_center_xy is not None else None),
            red_region_size_xy=(tuple(args.red_region_size_xy) if args.red_region_size_xy is not None else None),
            green_region_center_xy=(tuple(args.green_region_center_xy) if args.green_region_center_xy is not None else None),
            green_region_size_xy=(tuple(args.green_region_size_xy) if args.green_region_size_xy is not None else None),
            height_threshold=args.height_threshold,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=config_A["has_wrist"], has_wrist_B=config_B["has_wrist"],
            include_obj_pose_A=config_A["include_obj_pose"],
            include_obj_pose_B=config_B["include_obj_pose"],
            include_gripper_A=config_A["include_gripper"],
            include_gripper_B=config_B["include_gripper"],
        )

        goal_xy = np.array(args.goal_xy)
        rng = np.random.default_rng(args.seed)

        # Initial env reset + markers
        env.reset()
        # Pre-position robot to gripper-down rest pose
        pre_position_gripper_down(env)
        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1,
            device=device,
            red_marker_shape=args.red_marker_shape,
            red_marker_size_xy=(tuple(args.red_marker_size_xy) if args.red_marker_size_xy is not None else None),
        )
        tester.place_markers = place_markers
        tester.goal_markers = goal_markers
        tester.marker_z = marker_z

        first_place_xy = tester._sample_taskA_source_target()
        tester.current_place_xy = first_place_xy
        tester._update_place_marker(first_place_xy)

        def _hard_reset_for_task_A():
            """Hard reset and prepare for Task A (object at random table pos)."""
            env.reset()
            # Pre-position robot to gripper-down rest pose
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
            return rand_xy

        def _hard_reset_for_task_B():
            """Hard reset and prepare for Task B (object at goal pos)."""
            env.reset()
            # Pre-position robot to gripper-down rest pose
            pre_position_gripper_down(env)
            # Sample a random place target for Task B to place at
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
            policy_B.reset()

        # =================================================================
        # Phase 1: Evaluate Task A independently (N episodes)
        # =================================================================
        results_A = []
        episodes_A_details = []
        elapsed_A = 0.0
        n_success_A = 0
        rate_A = 0.0

        if "A" in args.tasks:
            print(f"\n{'='*60}")
            print(f"Phase 1: Independent Task A Evaluation ({args.num_episodes} episodes)")
            print(f"{'='*60}")

            start_A = time.time()

            for ep_idx in range(args.num_episodes):
                print(f"\n  [Task A] Episode {ep_idx + 1}/{args.num_episodes}")
                rand_xy = _hard_reset_for_task_A()
                print(f"    Object at random pos [{rand_xy[0]:.3f}, {rand_xy[1]:.3f}]")

                ep_A, success_A = tester.run_task_A()
                results_A.append(success_A)

                detail = {
                    "episode_index": ep_idx,
                    "success": success_A,
                    "success_step": ep_A.get("success_step"),
                    "total_steps": len(ep_A.get("images", [])),
                }
                if "obj_pose" in ep_A and len(ep_A["obj_pose"]) > 0:
                    detail["initial_obj_position"] = ep_A["obj_pose"][0][:3].tolist()
                    detail["final_obj_position"] = ep_A["obj_pose"][-1][:3].tolist()
                episodes_A_details.append(detail)

                status = "SUCCESS" if success_A else "FAILED"
                rate = sum(results_A) / len(results_A) * 100
                print(f"    {status}  (running: {sum(results_A)}/{len(results_A)} = {rate:.1f}%)")

            elapsed_A = time.time() - start_A
            n_success_A = sum(results_A)
            rate_A = n_success_A / len(results_A) if results_A else 0.0
            print(f"\n  Task A Final: {n_success_A}/{len(results_A)} = {rate_A:.1%}  ({elapsed_A:.1f}s)")

        # =================================================================
        # Phase 2: Evaluate Task B independently (N episodes)
        # =================================================================
        results_B = []
        episodes_B_details = []
        elapsed_B = 0.0
        n_success_B = 0
        rate_B = 0.0

        if "B" in args.tasks:
            print(f"\n{'='*60}")
            print(f"Phase 2: Independent Task B Evaluation ({args.num_episodes} episodes)")
            print(f"{'='*60}")

            start_B = time.time()

            for ep_idx in range(args.num_episodes):
                print(f"\n  [Task B] Episode {ep_idx + 1}/{args.num_episodes}")
                _hard_reset_for_task_B()
                print(f"    Object at goal [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}], "
                      f"target: [{tester.current_place_xy[0]:.3f}, {tester.current_place_xy[1]:.3f}]")

                ep_B, success_B = tester.run_task_B()
                results_B.append(success_B)

                detail = {
                    "episode_index": ep_idx,
                    "success": success_B,
                    "success_step": ep_B.get("success_step"),
                    "total_steps": len(ep_B.get("images", [])),
                }
                if "obj_pose" in ep_B and len(ep_B["obj_pose"]) > 0:
                    detail["initial_obj_position"] = ep_B["obj_pose"][0][:3].tolist()
                    detail["final_obj_position"] = ep_B["obj_pose"][-1][:3].tolist()
                episodes_B_details.append(detail)

                status = "SUCCESS" if success_B else "FAILED"
                rate = sum(results_B) / len(results_B) * 100
                print(f"    {status}  (running: {sum(results_B)}/{len(results_B)} = {rate:.1f}%)")

            elapsed_B = time.time() - start_B
            n_success_B = sum(results_B)
            rate_B = n_success_B / len(results_B) if results_B else 0.0
            print(f"\n  Task B Final: {n_success_B}/{len(results_B)} = {rate_B:.1%}  ({elapsed_B:.1f}s)")

        # =================================================================
        # Summary
        # =================================================================
        total_elapsed = elapsed_A + elapsed_B

        print(f"\n{'='*60}")
        print("Independent Evaluation Results")
        print(f"{'='*60}")
        if results_A:
            print(f"  Task A: {n_success_A}/{len(results_A)} = {rate_A:.1%}")
        else:
            print(f"  Task A: skipped")
        if results_B:
            print(f"  Task B: {n_success_B}/{len(results_B)} = {rate_B:.1%}")
        else:
            print(f"  Task B: skipped")
        print(f"  Total time: {total_elapsed:.1f}s")

        # Compute average success steps
        steps_A = [d["success_step"] for d in episodes_A_details
                    if d["success"] and d.get("success_step")]
        steps_B = [d["success_step"] for d in episodes_B_details
                    if d["success"] and d.get("success_step")]

        # =================================================================
        # Save statistics
        # =================================================================
        stats = {
            "experiment": "independent_evaluation",
            "description": (
                "Each task evaluated independently with full env.reset() "
                "before every episode. No A→B chaining."
            ),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_A": args.policy_A,
                "policy_B": args.policy_B,
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": args.goal_xy,
            },
            "summary": {
                "task_A_success_count": n_success_A,
                "task_A_total_episodes": len(results_A),
                "task_A_success_rate": rate_A,
                "avg_success_step_A": (sum(steps_A) / len(steps_A)) if steps_A else None,
                "task_A_elapsed_seconds": elapsed_A,
                "task_B_success_count": n_success_B,
                "task_B_total_episodes": len(results_B),
                "task_B_success_rate": rate_B,
                "avg_success_step_B": (sum(steps_B) / len(steps_B)) if steps_B else None,
                "task_B_elapsed_seconds": elapsed_B,
                "total_elapsed_seconds": total_elapsed,
            },
            "episodes_A": episodes_A_details,
            "episodes_B": episodes_B_details,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Saved statistics to: {out_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
