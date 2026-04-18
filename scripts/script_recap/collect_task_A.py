#!/usr/bin/env python3
"""Collect Task A rollout data (single-task, no Policy B).

Runs N episodes of Task A with automatic hard-reset between episodes.
Each episode: random object placement → run Policy A → record success/failure.
No Task B is involved — this is for CFG-RL self-learning (Exp56).

Supports CFG indicator injection via --cfg_indicator: appends a scalar value
to observation.state before policy inference (state_dim 15 → 16).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/script_recap/collect_task_A.py \
        --policy weights/PP_A_cfg/checkpoints/checkpoints/last/pretrained_model \
        --out_npz iter1_collect_A_p0.npz \
        --num_episodes 5 \
        --horizon 3000 \
        --save_all \
        --cfg_indicator 1.0 \
        --n_action_steps 16 \
        --goal_xy 0.5 -0.2 \
        --red_region_center_xy 0.5 0.2 \
        --red_region_size_xy 0.3 0.3 \
        --seed 42 \
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
        description="Collect Task A rollout episodes (single-task, CFG-RL).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Policy
    parser.add_argument("--policy", type=str, required=True,
                        help="Pretrained model directory for Policy A.")

    # Output
    parser.add_argument("--out_npz", type=str, required=True,
                        help="Output NPZ file for collected episodes.")

    # Collection parameters
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to collect.")
    parser.add_argument("--horizon", type=int, default=3000,
                        help="Max steps per episode.")
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=16)

    # CFG indicator
    parser.add_argument("--cfg_indicator", type=float, default=None,
                        help="CFG indicator value to concat to state during inference. "
                             "If None, no indicator is injected.")

    # Region (Mode 3)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.3, 0.3])

    # Save options
    parser.add_argument("--save_all", action="store_true",
                        help="Save all episodes (success+failure).")

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


class CFGPolicyWrapper:
    """Wraps a policy to inject indicator value into observation.state during inference."""

    def __init__(self, policy, indicator_value: float = 1.0):
        self.policy = policy
        self.indicator_value = indicator_value

    def select_action(self, batch):
        state = batch["observation.state"]
        indicator = torch.full(
            (*state.shape[:-1], 1),
            self.indicator_value,
            device=state.device,
            dtype=state.dtype,
        )
        batch["observation.state"] = torch.cat([state, indicator], dim=-1)
        return self.policy.select_action(batch)

    def reset(self):
        self.policy.reset()

    def eval(self):
        self.policy.eval()


class CFGPreprocessorWrapper:
    """Wraps a preprocessor to inject indicator into observation.state BEFORE normalization.

    The normalizer stats have 16 dims (15 + indicator). The environment provides
    15-dim state. This wrapper concatenates the indicator value as the 16th dim
    before calling the original preprocessor.
    """

    def __init__(self, preprocessor, indicator_value: float = 1.0):
        self.preprocessor = preprocessor
        self.indicator_value = indicator_value

    def __call__(self, transition):
        state = transition["observation.state"]
        indicator = torch.full(
            (*state.shape[:-1], 1),
            self.indicator_value,
            device=state.device,
            dtype=state.dtype,
        )
        transition["observation.state"] = torch.cat([state, indicator], dim=-1)
        return self.preprocessor(transition)
        return self

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped policy
        if name in ('policy', 'indicator_value'):
            return object.__getattribute__(self, name)
        return getattr(self.policy, name)


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

        # ---- Load policy config ----
        config_A = load_policy_config(args.policy)

        # For CFG models with state_dim=16 (15 + indicator), override the inference
        # to use 15-dim env state (obj_pose + gripper). The indicator is injected
        # by CFGPreprocessorWrapper before normalization.
        if config_A["state_dim"] == 16 and args.cfg_indicator is not None:
            config_A["include_obj_pose"] = True
            config_A["include_gripper"] = True
            print(f"  [CFG] state_dim=16 detected → forced include_obj_pose=True, include_gripper=True")

        # ---- Create environment ----
        env = make_env_with_camera(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        # ---- Load policy ----
        policy_A, preproc_A, postproc_A, _, n_act_A = load_policy_auto(
            args.policy, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_A.eval()

        # Wrap preprocessor with CFG indicator injection (before normalization)
        use_cfg = args.cfg_indicator is not None
        if use_cfg:
            print(f"  [CFG] Wrapping preprocessor with indicator={args.cfg_indicator}")
            preproc_A = CFGPreprocessorWrapper(preproc_A, args.cfg_indicator)
            wrapped_policy = policy_A  # No need to wrap policy; indicator injected in preprocessor
        else:
            wrapped_policy = policy_A

        # ---- Build a tester (Mode 3, but we only use Task A) ----
        # We still need an AlternatingTester to leverage _run_task and helpers.
        # Use Policy A for both slots, but we only call run_task_A.
        tester = AlternatingTester(
            env=env,
            policy_A=wrapped_policy, preprocessor_A=preproc_A, postprocessor_A=postproc_A,
            policy_B=wrapped_policy, preprocessor_B=preproc_A, postprocessor_B=postproc_A,
            n_action_steps_A=n_act_A, n_action_steps_B=n_act_A,
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
            has_wrist_B=config_A["has_wrist"],
            include_obj_pose_A=config_A["include_obj_pose"],
            include_obj_pose_B=config_A["include_obj_pose"],
            include_gripper_A=config_A["include_gripper"],
            include_gripper_B=config_A["include_gripper"],
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

        # ---- Helper reset for Task A ----
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
            wrapped_policy.reset()

        # Teleport object for first episode
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

        # ---- Main collection loop (Task A only) ----
        all_episodes = []
        results = []
        start_time = time.time()

        for ep in range(args.num_episodes):
            print(f"\n{'='*50}")
            print(f"Episode {ep + 1}/{args.num_episodes}")
            print(f"{'='*50}")

            # Run Task A
            ep_data, success = tester.run_task_A()
            all_episodes.append(ep_data)
            results.append(success)
            print(f"  Task A: {'SUCCESS' if success else 'FAILED'} "
                  f"({len(ep_data['action'])} steps)")

            # Hard reset for next episode
            new_place_xy = tester._sample_new_place_target()
            tester._update_place_marker(new_place_xy)
            _hard_reset_for_task_A()

            a_rate = sum(results) / len(results) * 100
            print(f"  Running rate: A={a_rate:.1f}% ({sum(results)}/{len(results)})")

        elapsed = time.time() - start_time
        n_success = sum(results)
        rate = n_success / len(results) if results else 0.0

        print(f"\n{'='*60}")
        print("Collection Results")
        print(f"{'='*60}")
        print(f"  Episodes:       {args.num_episodes}")
        print(f"  Task A success: {n_success}/{len(results)} = {rate:.1%}")
        print(f"  Time:           {elapsed:.1f}s")

        # Save episodes
        if args.save_all:
            episodes_to_save = all_episodes
        else:
            episodes_to_save = [ep for ep in all_episodes if ep["success"]]

        out_path = Path(args.out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if episodes_to_save:
            np.savez_compressed(
                str(out_path),
                episodes=np.array(episodes_to_save, dtype=object),
            )
            n_succ = sum(1 for ep in episodes_to_save if ep["success"])
            print(f"Saved {len(episodes_to_save)} episodes "
                  f"({n_succ} success, {len(episodes_to_save)-n_succ} fail) "
                  f"to {out_path}")
        else:
            # Save empty
            np.savez_compressed(
                str(out_path),
                episodes=np.array([], dtype=object),
            )
            print(f"WARNING: No episodes to save! Saved empty file to {out_path}")

        # Save JSON statistics
        stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy": args.policy,
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "cfg_indicator": args.cfg_indicator,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
            },
            "summary": {
                "total_episodes": args.num_episodes,
                "task_A_success_count": n_success,
                "task_A_success_rate": rate,
                "total_elapsed_seconds": elapsed,
                "total_cycles": args.num_episodes,  # compat with exp41 record format
            },
            "per_episode_results": [
                {"episode": i + 1, "task_A_success": results[i]}
                for i in range(len(results))
            ],
        }

        stats_path = out_path.with_suffix(".stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics to: {stats_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
