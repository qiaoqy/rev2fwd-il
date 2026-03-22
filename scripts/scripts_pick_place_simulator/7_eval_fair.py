#!/usr/bin/env python3
"""Step 7: Independent fair evaluation — single task per run.

Each episode is independently hard-reset so there is no coupling between
episodes.  Only produces a statistics JSON (no rollout NPZ).

Mode 3 only (red rectangle region).

Usage:
    conda activate rev2fwd_il

    # Task A
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/7_eval_fair.py \\
        --policy weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --task A --num_episodes 50 \\
        --out data/exp_new/fair_test_A.stats.json --headless

    # Task B
    CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place_simulator/7_eval_fair.py \\
        --policy weights/PP_B/checkpoints/checkpoints/last/pretrained_model \\
        --task B --num_episodes 50 \\
        --out data/exp_new/fair_test_B.stats.json --headless
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
        description="Independent fair evaluation for a single task (Mode 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--policy", type=str, required=True,
                        help="Policy checkpoint path.")
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"],
                        help="Which task to evaluate: A or B.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output statistics JSON path.")

    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1500)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=8)

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

        # AlternatingTester still needs both policy_A and policy_B slots.
        # Fill the unused slot with the same policy (it won't be called).
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

        # ---- Evaluate ----
        results = []
        per_episode = []
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Fair Evaluation: Task {args.task}  ({args.num_episodes} episodes)")
        print(f"{'='*60}")

        for ep_idx in range(args.num_episodes):
            if args.task == "A":
                _hard_reset_for_task_A()
                ep, success = tester.run_task_A()
            else:
                _hard_reset_for_task_B()
                ep, success = tester.run_task_B()

            results.append(success)
            steps = len(ep.get("images", []))
            per_episode.append({
                "episode": ep_idx,
                "success": bool(success),
                "steps": steps,
            })

            rate = sum(results) / len(results) * 100
            print(f"  [{ep_idx + 1}/{args.num_episodes}] "
                  f"{'SUCCESS' if success else 'FAILED'} "
                  f"({steps} steps) — running: {rate:.1f}%")

        elapsed = time.time() - start_time
        n_success = sum(results)
        success_rate = n_success / len(results) if results else 0.0

        print(f"\n{'='*60}")
        print(f"Task {args.task} Results")
        print(f"{'='*60}")
        print(f"  Success rate: {n_success}/{len(results)} = {success_rate:.1%}")
        print(f"  Time:         {elapsed:.1f}s")

        # Save stats
        stats = {
            "task": args.task,
            "success_rate": success_rate,
            "num_success": n_success,
            "num_total": len(results),
            "per_episode": per_episode,
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
            },
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to: {out_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
