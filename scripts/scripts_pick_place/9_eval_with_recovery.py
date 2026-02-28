#!/usr/bin/env python3
"""Step 9: Alternating test A→B→A→B with hard-reset between every task.

Unlike 6_test_alternating.py which breaks on first failure, this script
performs a FIXED number of A-B cycles (default: 50) and always resets the
robot arm to home position between tasks. This allows computing meaningful
success rates for both tasks.

=============================================================================
INTER-TASK RESET MECHANISM
=============================================================================
After EVERY task execution (regardless of success or failure), the script:
  1. Resets the environment (arm returns to home position)
  2. Teleports the object to the correct starting position for the next task
  3. Runs settle steps to let physics stabilize
  4. Continues with the next task

Specifically:
  After Task A success:
    - Run 100 transition frames with Policy A (gripper open, not recorded)
    - env.reset() (arm home)
    - Teleport object to goal position (Task B picks from goal)

  After Task A failure:
    - env.reset() (arm home)
    - Teleport object to goal position (Task B picks from goal)

  After Task B success:
    - Run 100 transition frames with Policy B (gripper open, not recorded)
    - env.reset() (arm home)
    - Teleport object to random table position (Task A picks from table)

  After Task B failure:
    - env.reset() (arm home)
    - Teleport object to random table position (Task A picks from table)

=============================================================================
OUTPUT
=============================================================================
- Rollout NPZ files (successful episodes only) for Task A and Task B
- JSON statistics file with per-episode details and overall success rates
- Optional video recording

=============================================================================
USAGE EXAMPLES
=============================================================================
# Run 50 cycles with failure recovery
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/9_eval_with_recovery.py \
    --policy_A runs/PP_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/PP_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/eval_A_iter1.npz \
    --out_B data/eval_B_iter1.npz \
    --num_cycles 50 --headless

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run alternating A/B test with failure recovery and success rate tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Policy checkpoints
    parser.add_argument("--policy_A", type=str, required=True,
                        help="Path to Task A policy checkpoint.")
    parser.add_argument("--policy_B", type=str, required=True,
                        help="Path to Task B policy checkpoint.")

    # Output paths
    parser.add_argument("--out_A", type=str, default="data/eval_A.npz",
                        help="Output path for Task A rollout data (successful only).")
    parser.add_argument("--out_B", type=str, default="data/eval_B.npz",
                        help="Output path for Task B rollout data (successful only).")

    # Test parameters
    parser.add_argument("--num_cycles", type=int, default=50,
                        help="Number of complete A→B cycles. Default: 50.")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Maximum steps per task attempt.")
    parser.add_argument("--height_threshold", type=float, default=0.15,
                        help="Minimum object z-position threshold.")
    parser.add_argument("--distance_threshold", type=float, default=0.05,
                        help="Maximum distance from target for success.")

    # Environment settings
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
                        help="Isaac Lab Gym task ID.")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, 0.0])
    parser.add_argument("--n_action_steps", type=int, default=None)

    # Video
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=30)

    # Isaac Lab AppLauncher arguments
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        # ---- imports that require Isaac Sim running ----
        import importlib.util
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_object_pose_w,
            teleport_object_to_pose,
        )
        # Re-use utilities from the existing alternating test script (6_test_alternating.py)
        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).parent / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        make_env_with_camera = _alt_mod.make_env_with_camera
        load_policy_config = _alt_mod.load_policy_config
        load_diffusion_policy = _alt_mod.load_diffusion_policy
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
        env_needs_wrist = config_A["has_wrist"] or config_B["has_wrist"]

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
        policy_A, preproc_A, postproc_A, _, n_act_A = load_diffusion_policy(
            args.policy_A, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_A.eval()

        policy_B, preproc_B, postproc_B, _, n_act_B = load_diffusion_policy(
            args.policy_B, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_B.eval()

        # =================================================================
        # Build tester (reuse AlternatingTester for single-task execution)
        # =================================================================
        tester = AlternatingTester(
            env=env,
            policy_A=policy_A, preprocessor_A=preproc_A, postprocessor_A=postproc_A,
            policy_B=policy_B, preprocessor_B=preproc_B, postprocessor_B=postproc_B,
            n_action_steps_A=n_act_A, n_action_steps_B=n_act_B,
            goal_xy=tuple(args.goal_xy),
            height_threshold=args.height_threshold,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=config_A["has_wrist"], has_wrist_B=config_B["has_wrist"],
            include_obj_pose_A=config_A["include_obj_pose"],
            include_obj_pose_B=config_B["include_obj_pose"],
            include_gripper_A=config_A["include_gripper"],
            include_gripper_B=config_B["include_gripper"],
        )

        # =================================================================
        # Run alternating test with hard-reset recovery
        # =================================================================
        print(f"\n{'='*60}")
        print("Starting Alternating Test WITH Recovery")
        print(f"{'='*60}")
        print(f"  Num cycles:           {args.num_cycles}")
        print(f"  Horizon per task:     {args.horizon}")
        print(f"  Distance threshold:   {args.distance_threshold}m")
        print(f"  Goal XY:              {args.goal_xy}")

        goal_xy = np.array(args.goal_xy)
        rng = np.random.default_rng(args.seed)

        # ---- initial environment setup ----
        env.reset()

        # Create markers
        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
        )
        tester.place_markers = place_markers
        tester.goal_markers = goal_markers
        tester.marker_z = marker_z

        # Sample first place target (red marker)
        first_place_xy = tester._sample_new_place_target()
        tester.current_place_xy = first_place_xy
        update_target_markers(
            place_markers, goal_markers,
            first_place_xy, tuple(goal_xy), marker_z, env,
        )

        def _hard_reset_for_task_A():
            """Hard reset: prepare environment for Task A.
            
            Resets the robot arm and teleports the object to a random
            position on the table (Task A picks from arbitrary table pos).
            """
            env.reset()
            # Re-create markers after env reset
            update_target_markers(
                place_markers, goal_markers,
                tester.current_place_xy, tuple(goal_xy), marker_z, env,
            )
            # Sample random table position
            rand_xy = tester._sample_new_place_target()
            obj_pose = torch.tensor(
                [rand_xy[0], rand_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            # Settle
            zero_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            for _ in range(10):
                env.step(zero_action)
            tester.current_gripper_state = 1.0
            policy_A.reset()
            print(f"    [Hard Reset] Robot reset, object at random pos [{rand_xy[0]:.3f}, {rand_xy[1]:.3f}] for Task A")

        def _hard_reset_for_task_B():
            """Hard reset: prepare environment for Task B.
            
            Resets the robot arm and teleports the object to the goal
            position (Task B picks from goal).
            """
            env.reset()
            update_target_markers(
                place_markers, goal_markers,
                tester.current_place_xy, tuple(goal_xy), marker_z, env,
            )
            obj_pose = torch.tensor(
                [goal_xy[0], goal_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            zero_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            for _ in range(10):
                env.step(zero_action)
            tester.current_gripper_state = 1.0
            policy_B.reset()
            print(f"    [Hard Reset] Robot reset, object at goal [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}] for Task B")

        # ---- Teleport object to initial position for first Task A ----
        init_pose = torch.tensor(
            [first_place_xy[0], first_place_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        teleport_object_to_pose(env, init_pose, name="object")
        zero_action = torch.zeros(1, env.action_space.shape[-1], device=device)
        for _ in range(10):
            env.step(zero_action)

        # ---- main loop ----
        results_A = []  # list of bools
        results_B = []
        start_time = time.time()

        for cycle in range(args.num_cycles):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{args.num_cycles}")
            print(f"{'='*50}")

            # ---- Task A ----
            print(f"  Running Task A (pick → place at goal)...")
            ep_A, success_A = tester.run_task_A()
            tester.episodes_A.append(ep_A)
            results_A.append(success_A)

            if success_A:
                print(f"  ✓ Task A SUCCESS")
                # Run 100 transition frames with policy A (not recording data)
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
                print(f"  ✗ Task A FAILED")
                new_place_xy = tester._sample_new_place_target()
                tester._update_place_marker(new_place_xy)

            # Always hard-reset arm and prepare for Task B
            _hard_reset_for_task_B()

            # ---- Task B ----
            print(f"  Running Task B (pick from goal → place at red marker)...")
            if tester.current_place_xy is not None:
                print(f"    Target: red marker at [{tester.current_place_xy[0]:.3f}, {tester.current_place_xy[1]:.3f}]")
            ep_B, success_B = tester.run_task_B()
            tester.episodes_B.append(ep_B)
            results_B.append(success_B)

            if success_B:
                print(f"  ✓ Task B SUCCESS")
                # Run 100 transition frames with policy B (not recording data)
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
            else:
                print(f"  ✗ Task B FAILED")

            # Always hard-reset arm and prepare for next Task A
            _hard_reset_for_task_A()

            # per-cycle summary
            a_rate = sum(results_A) / len(results_A) * 100
            b_rate = sum(results_B) / len(results_B) * 100
            print(f"  Running success rates: A={a_rate:.1f}% ({sum(results_A)}/{len(results_A)})  "
                  f"B={b_rate:.1f}% ({sum(results_B)}/{len(results_B)})")

        elapsed = time.time() - start_time

        # =================================================================
        # Compute final success rates
        # =================================================================
        n_success_A = sum(results_A)
        n_success_B = sum(results_B)
        rate_A = n_success_A / len(results_A) if results_A else 0.0
        rate_B = n_success_B / len(results_B) if results_B else 0.0

        print(f"\n{'='*60}")
        print("Evaluation Results (with recovery)")
        print(f"{'='*60}")
        print(f"  Total cycles:      {args.num_cycles}")
        print(f"  Task A success:    {n_success_A}/{len(results_A)} = {rate_A:.1%}")
        print(f"  Task B success:    {n_success_B}/{len(results_B)} = {rate_B:.1%}")
        print(f"  Total time:        {elapsed:.1f}s")

        # =================================================================
        # Save successful rollout data for training
        # =================================================================
        tester.save_data(args.out_A, args.out_B)

        # =================================================================
        # Save JSON statistics
        # =================================================================
        stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_A": args.policy_A,
                "policy_B": args.policy_B,
                "num_cycles": args.num_cycles,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": args.goal_xy,
            },
            "summary": {
                "total_cycles": args.num_cycles,
                "task_A_success_count": n_success_A,
                "task_B_success_count": n_success_B,
                "total_task_A_episodes": len(results_A),
                "total_task_B_episodes": len(results_B),
                "task_A_success_rate": rate_A,
                "task_B_success_rate": rate_B,
                "total_elapsed_seconds": elapsed,
            },
            "per_cycle_results": [
                {"cycle": i + 1, "task_A_success": results_A[i], "task_B_success": results_B[i]}
                for i in range(len(results_A))
            ],
            "episodes_A": [],
            "episodes_B": [],
        }

        for i, ep in enumerate(tester.episodes_A):
            ep_stat = {
                "episode_index": i,
                "success": ep["success"],
                "success_step": ep.get("success_step"),
                "total_steps": len(ep["images"]),
            }
            if "ee_pose" in ep and len(ep["ee_pose"]) > 0:
                ep_stat["initial_ee_position"] = ep["ee_pose"][0][:3].tolist()
                ep_stat["final_ee_position"] = ep["ee_pose"][-1][:3].tolist()
            if "obj_pose" in ep and len(ep["obj_pose"]) > 0:
                ep_stat["initial_obj_position"] = ep["obj_pose"][0][:3].tolist()
                ep_stat["final_obj_position"] = ep["obj_pose"][-1][:3].tolist()
            stats["episodes_A"].append(ep_stat)

        for i, ep in enumerate(tester.episodes_B):
            ep_stat = {
                "episode_index": i,
                "success": ep["success"],
                "success_step": ep.get("success_step"),
                "total_steps": len(ep["images"]),
            }
            if "ee_pose" in ep and len(ep["ee_pose"]) > 0:
                ep_stat["initial_ee_position"] = ep["ee_pose"][0][:3].tolist()
                ep_stat["final_ee_position"] = ep["ee_pose"][-1][:3].tolist()
            if "obj_pose" in ep and len(ep["obj_pose"]) > 0:
                ep_stat["initial_obj_position"] = ep["obj_pose"][0][:3].tolist()
                ep_stat["final_obj_position"] = ep["obj_pose"][-1][:3].tolist()
            stats["episodes_B"].append(ep_stat)

        stats_path = Path(args.out_A).with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to: {stats_path}")

        # =================================================================
        # Save video
        # =================================================================
        if args.save_video:
            if args.video_path is None:
                video_path = Path(args.out_A).with_suffix(".mp4")
            else:
                video_path = Path(args.video_path)
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
