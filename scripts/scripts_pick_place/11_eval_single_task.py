#!/usr/bin/env python3
"""Step 11: Single-task evaluation with configurable initial position noise.

Evaluates a single task (A or B) with multiple checkpoints in one Isaac Sim
session. Supports adding noise to the object's initial position to match the
distribution of training data collected with a distance threshold.

=============================================================================
MOTIVATION
=============================================================================
When data is collected with a distance_threshold of D (e.g., 5cm), successful
episodes end with the object within D of the target. When these trajectories
are time-reversed to create training data for the opposite task, the "start"
position has up to D error from the ideal position. To evaluate the policy
under conditions consistent with its training data, we add the same D noise
to the initial object position during testing.

=============================================================================
NOISE MODEL
=============================================================================
The noise is uniformly distributed within a circle of radius `init_noise`:
  - angle ~ Uniform(0, 2π)
  - r     ~ init_noise × √(Uniform(0, 1))     [ensures uniform over area]
  - offset = (r·cos(angle), r·sin(angle))

For Task A: object starts at (random_table_xy + offset)
For Task B: object starts at (goal_xy + offset)

Positions are clamped to table bounds [0.35, 0.65] × [−0.25, 0.25].

=============================================================================
USAGE
=============================================================================
# Test multiple Task A checkpoints on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/11_eval_single_task.py \
    --task_type A \
    --checkpoints ckpt_dir1 ckpt_dir2 ... \
    --labels iter0 iter1 ... \
    --num_episodes 50 \
    --init_noise 0.05 \
    --out results_A.json \
    --headless

# Test multiple Task B checkpoints on GPU 2
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_pick_place/11_eval_single_task.py \
    --task_type B \
    --checkpoints ckpt_dir1 ckpt_dir2 ... \
    --labels iter0 iter1 ... \
    --num_episodes 50 \
    --init_noise 0.05 \
    --out results_B.json \
    --headless
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
    parser = argparse.ArgumentParser(
        description="Single-task evaluation with initial position noise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--task_type", type=str, required=True, choices=["A", "B"],
                        help="Which task to evaluate: A or B.")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="List of pretrained_model directories to evaluate.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for each checkpoint (e.g., iter0 iter1 ...). "
                             "Default: derived from checkpoint path.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSON path for aggregated results.")

    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Episodes per checkpoint. Default: 50.")
    parser.add_argument("--init_noise", type=float, default=0.05,
                        help="Radius (m) of uniform noise circle for initial object position. "
                             "Default: 0.05 (5cm, matches distance_threshold).")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Maximum steps per episode.")
    parser.add_argument("--height_threshold", type=float, default=0.15)
    parser.add_argument("--distance_threshold", type=float, default=0.05)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, 0.0])
    parser.add_argument("--n_action_steps", type=int, default=None)

    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ---- Noise helper ----
def _sample_noise_offset(rng: np.random.Generator, radius: float) -> Tuple[float, float]:
    """Sample a uniform offset within a circle of given radius."""
    if radius <= 0:
        return 0.0, 0.0
    angle = rng.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(rng.uniform(0, 1))
    return float(r * np.cos(angle)), float(r * np.sin(angle))


TABLE_X_MIN, TABLE_X_MAX = 0.35, 0.65
TABLE_Y_MIN, TABLE_Y_MAX = -0.25, 0.25


def _clamp_to_table(x: float, y: float) -> Tuple[float, float]:
    """Clamp XY to table bounds."""
    return (
        max(TABLE_X_MIN + 0.02, min(TABLE_X_MAX - 0.02, x)),
        max(TABLE_Y_MIN + 0.02, min(TABLE_Y_MAX - 0.02, y)),
    )


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import importlib.util
        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_object_pose_w,
            teleport_object_to_pose,
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
        load_diffusion_policy = _alt_mod.load_diffusion_policy
        AlternatingTester = _alt_mod.AlternatingTester
        create_target_markers = _alt_mod.create_target_markers
        update_target_markers = _alt_mod.update_target_markers

        set_seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

        goal_xy = np.array(args.goal_xy)
        rng = np.random.default_rng(args.seed)

        # Labels
        labels = args.labels
        if labels is None:
            labels = [Path(c).parent.name if Path(c).name == "pretrained_model" else Path(c).name
                      for c in args.checkpoints]
        assert len(labels) == len(args.checkpoints), \
            f"Number of labels ({len(labels)}) must match checkpoints ({len(args.checkpoints)})"

        # =================================================================
        # Create environment (once)
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

        env.reset()
        zero_action = torch.zeros(1, env.action_space.shape[-1], device=device)

        # Create markers (once)
        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
        )

        # =================================================================
        # Iterate over checkpoints
        # =================================================================
        all_results = []

        for ckpt_idx, (ckpt_path, label) in enumerate(zip(args.checkpoints, labels)):
            print(f"\n{'='*60}")
            print(f"Checkpoint {ckpt_idx + 1}/{len(args.checkpoints)}: {label}")
            print(f"  Path: {ckpt_path}")
            print(f"  Task: {args.task_type}")
            print(f"  Episodes: {args.num_episodes}")
            print(f"  Init noise: {args.init_noise}m")
            print(f"{'='*60}")

            # Load policy config
            config = load_policy_config(ckpt_path)

            # Load policy
            policy, preproc, postproc, _, n_act = load_diffusion_policy(
                ckpt_path, device,
                image_height=args.image_height, image_width=args.image_width,
                n_action_steps=args.n_action_steps,
            )
            policy.eval()

            # Build tester (same policy for both slots; only one side gets called)
            tester = AlternatingTester(
                env=env,
                policy_A=policy, preprocessor_A=preproc, postprocessor_A=postproc,
                policy_B=policy, preprocessor_B=preproc, postprocessor_B=postproc,
                n_action_steps_A=n_act, n_action_steps_B=n_act,
                goal_xy=tuple(args.goal_xy),
                height_threshold=args.height_threshold,
                distance_threshold=args.distance_threshold,
                horizon=args.horizon,
                has_wrist_A=config["has_wrist"], has_wrist_B=config["has_wrist"],
                include_obj_pose_A=config["include_obj_pose"],
                include_obj_pose_B=config["include_obj_pose"],
                include_gripper_A=config["include_gripper"],
                include_gripper_B=config["include_gripper"],
            )
            tester.place_markers = place_markers
            tester.goal_markers = goal_markers
            tester.marker_z = marker_z

            # Initialize place target
            init_place_xy = tester._sample_new_place_target()
            tester.current_place_xy = init_place_xy
            update_target_markers(
                place_markers, goal_markers,
                init_place_xy, tuple(goal_xy), marker_z, env,
            )

            # ---- Run episodes ----
            results = []
            episode_details = []
            start_time = time.time()

            for ep_idx in range(args.num_episodes):
                # Reset environment
                env.reset()
                update_target_markers(
                    place_markers, goal_markers,
                    tester.current_place_xy, tuple(goal_xy), marker_z, env,
                )

                if args.task_type == "A":
                    # Task A: object at random table pos + noise
                    base_xy = tester._sample_new_place_target()
                    dx, dy = _sample_noise_offset(rng, args.init_noise)
                    obj_x, obj_y = _clamp_to_table(base_xy[0] + dx, base_xy[1] + dy)
                    obj_pose = torch.tensor(
                        [obj_x, obj_y, 0.022, 1.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32, device=device,
                    ).unsqueeze(0)
                    teleport_object_to_pose(env, obj_pose, name="object")
                    for _ in range(10):
                        env.step(zero_action)
                    tester.current_gripper_state = 1.0
                    policy.reset()

                    print(f"  [Task A] Ep {ep_idx + 1}/{args.num_episodes}  "
                          f"obj=[{obj_x:.3f}, {obj_y:.3f}] "
                          f"(noise dx={dx:.3f}, dy={dy:.3f})")

                    ep_data, success = tester.run_task_A()

                else:  # Task B
                    # Task B: object at goal + noise, random place target
                    new_place = tester._sample_new_place_target()
                    tester.current_place_xy = new_place
                    update_target_markers(
                        place_markers, goal_markers,
                        new_place, tuple(goal_xy), marker_z, env,
                    )
                    dx, dy = _sample_noise_offset(rng, args.init_noise)
                    obj_x = goal_xy[0] + dx
                    obj_y = goal_xy[1] + dy
                    obj_x, obj_y = _clamp_to_table(obj_x, obj_y)
                    obj_pose = torch.tensor(
                        [obj_x, obj_y, 0.022, 1.0, 0.0, 0.0, 0.0],
                        dtype=torch.float32, device=device,
                    ).unsqueeze(0)
                    teleport_object_to_pose(env, obj_pose, name="object")
                    for _ in range(10):
                        env.step(zero_action)
                    tester.current_gripper_state = 1.0
                    policy.reset()

                    print(f"  [Task B] Ep {ep_idx + 1}/{args.num_episodes}  "
                          f"obj=[{obj_x:.3f}, {obj_y:.3f}] "
                          f"target=[{new_place[0]:.3f}, {new_place[1]:.3f}] "
                          f"(noise dx={dx:.3f}, dy={dy:.3f})")

                    ep_data, success = tester.run_task_B()

                results.append(success)
                detail = {
                    "episode_index": ep_idx,
                    "success": success,
                    "success_step": ep_data.get("success_step"),
                    "total_steps": len(ep_data.get("images", [])),
                    "init_obj_xy": [obj_x, obj_y],
                    "noise_offset": [dx, dy],
                }
                if "obj_pose" in ep_data and len(ep_data["obj_pose"]) > 0:
                    detail["final_obj_position"] = ep_data["obj_pose"][-1][:3].tolist()
                episode_details.append(detail)

                status = "OK" if success else "FAIL"
                rate = sum(results) / len(results) * 100
                print(f"    {status}  (running: {sum(results)}/{len(results)} = {rate:.1f}%)")

            elapsed = time.time() - start_time
            n_success = sum(results)
            rate = n_success / len(results) if results else 0.0

            # Average success steps
            s_steps = [d["success_step"] for d in episode_details
                       if d["success"] and d.get("success_step")]
            avg_step = (sum(s_steps) / len(s_steps)) if s_steps else None

            print(f"\n  ── {label}: {n_success}/{len(results)} = {rate:.1%}  "
                  f"({elapsed:.1f}s, avg_step={avg_step})")

            all_results.append({
                "label": label,
                "checkpoint": ckpt_path,
                "task_type": args.task_type,
                "success_count": n_success,
                "total_episodes": len(results),
                "success_rate": rate,
                "avg_success_step": avg_step,
                "elapsed_seconds": elapsed,
                "episodes": episode_details,
            })

            # Free policy memory for next checkpoint
            del policy, preproc, postproc, tester
            torch.cuda.empty_cache()

        # =================================================================
        # Save aggregated results
        # =================================================================
        output = {
            "experiment": f"single_task_eval_{args.task_type}",
            "description": (
                f"Independent Task {args.task_type} evaluation with {args.init_noise}m "
                f"initial position noise. {args.num_episodes} episodes per checkpoint."
            ),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "task_type": args.task_type,
                "num_episodes": args.num_episodes,
                "init_noise": args.init_noise,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "goal_xy": args.goal_xy,
                "n_action_steps": args.n_action_steps,
                "seed": args.seed,
            },
            "results": all_results,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Task {args.task_type} Evaluation Summary")
        print(f"{'='*60}")
        print(f"  {'Label':<12}  {'Rate':>8}  {'OK/Total':>10}  {'Time':>8}")
        print(f"  {'-'*44}")
        for r in all_results:
            print(f"  {r['label']:<12}  {r['success_rate']*100:>7.1f}%  "
                  f"{r['success_count']:>3}/{r['total_episodes']:<3}     "
                  f"{r['elapsed_seconds']:>6.1f}s")
        print(f"\n  Saved to: {out_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
