#!/usr/bin/env python3
"""Step 6: Cyclic A↔B evaluation for 3-cube partial-unstack task (Exp45).

Runs N A→B cycles in the Exp44 three-cube environment.  After every task
(success or failure) the environment is hard-reset:
  - Task A (stack):   3 cubes scattered at fixed positions → policy stacks
  - Task B (unstack): 3 cubes stacked at stack_pos → policy unloads to fixed positions

Success criteria (per-frame real-time detection + early stop after 20-frame buffer):
  Task A: 2-round stacking — each round requires gripper open + target cube
          height at expected z ± tolerance for 10 consecutive frames.
          Round 1: cube_medium on cube_large (z_expected ≈ 0.07m)
          Round 2: cube_small on cube_medium (z_expected ≈ 0.12m)
  Task B: cube_small AND cube_medium both XY > 10cm from stack_pos AND z < 5cm

State: 8-dim (ee_pose(7) + gripper(1)), NO obj_pose.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/6_eval_cyclic_cube_stack.py \\
        --policy_A <path_to_PP_A> \\
        --policy_B <path_to_PP_B> \\
        --out_A data/.../eval_A.npz \\
        --out_B data/.../eval_B.npz \\
        --num_cycles 5 --headless
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


TASK_ID = "Isaac-Exp44-CubeStack-Franka-IK-Abs-v0"


# ── argument parsing ──────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cyclic A↔B evaluation for 3-cube partial-unstack (Exp45).",
    )

    # Policies
    parser.add_argument("--policy_A", type=str, required=True)
    parser.add_argument("--policy_B", type=str, required=True)

    # Output
    parser.add_argument("--out_A", type=str, required=True)
    parser.add_argument("--out_B", type=str, required=True)

    # Test parameters
    parser.add_argument("--num_cycles", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=6000,
                        help="Max steps per task (default: 6000).")
    parser.add_argument("--n_action_steps", type=int, default=16)

    # Task A success parameters
    parser.add_argument("--height_tolerance", type=float, default=0.015,
                        help="Z tolerance (m) for Task A height check (default: 1.5cm).")
    parser.add_argument("--stable_frames", type=int, default=10,
                        help="Consecutive frames required for Task A round success.")

    # Task B success parameters
    parser.add_argument("--unstack_xy_threshold", type=float, default=0.10,
                        help="Min XY distance from stack for Task B (default: 10cm).")
    parser.add_argument("--unstack_z_threshold", type=float, default=0.05,
                        help="Max z for cube on table for Task B (default: 5cm).")

    # Early stop
    parser.add_argument("--early_stop_buffer", type=int, default=20,
                        help="Extra frames to record after success before stopping.")

    # Save options
    parser.add_argument("--save_all", action="store_true",
                        help="Save all episodes (success+failure).")

    # Video
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_fps", type=int, default=30)

    # Environment / rendering
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=0)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import importlib.util
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401

        from rev2fwd_il.utils.seed import set_seed
        from rev2fwd_il.sim.scene_api import (
            get_ee_pose_w,
            get_object_pose_w,
            teleport_object_to_pose,
            pre_position_gripper_down,
        )
        from rev2fwd_il.sim.exp44_registry import (
            CUBE_DEFS_44,
            DEFAULT_EXP44_CONFIG,
            PICK_ORDER_44,
            get_cube_def_44,
        )

        # Load utility functions from 6_test_alternating.py
        _alt_spec = importlib.util.spec_from_file_location(
            "test_alternating",
            str(Path(__file__).resolve().parent.parent
                / "scripts_pick_place" / "6_test_alternating.py"),
        )
        _alt_mod = importlib.util.module_from_spec(_alt_spec)
        _alt_spec.loader.exec_module(_alt_mod)

        load_policy_config = _alt_mod.load_policy_config
        load_policy_auto = _alt_mod.load_policy_auto
        add_camera_to_env_cfg = _alt_mod.add_camera_to_env_cfg

        # Load Exp44 env config
        _env_cfg_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based"
            / "manipulation" / "lift" / "config" / "franka"
            / "exp44_env_cfg.py"
        )
        _env_cfg_spec = importlib.util.spec_from_file_location(
            "exp44_env_cfg", _env_cfg_path)
        _env_cfg_mod = importlib.util.module_from_spec(_env_cfg_spec)
        _env_cfg_spec.loader.exec_module(_env_cfg_mod)
        FrankaExp44EnvCfg = _env_cfg_mod.FrankaExp44EnvCfg

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        cfg = DEFAULT_EXP44_CONFIG
        stack_xy = np.array(cfg.stack_xy, dtype=np.float64)

        # Expected stacked heights for Task A success detection
        # cube_medium on cube_large: large_half(0.04) + medium_half(0.03)
        z_expected_medium = 0.07
        # cube_small on cube_medium on cube_large: 0.04 + 0.06 + 0.02
        z_expected_small = 0.12

        # Fixed scatter positions for Task A start / Task B target
        scatter_positions = {
            "cube_small": np.array(cfg.small_place_xy, dtype=np.float64),
            "cube_medium": np.array(cfg.medium_place_xy, dtype=np.float64),
        }

        # ── create env ────────────────────────────────────────────────────
        def _make_env():
            env_cfg = FrankaExp44EnvCfg()
            env_cfg.scene.num_envs = 1
            env_cfg.sim.device = device
            if bool(args.disable_fabric):
                env_cfg.sim.use_fabric = False
            env_cfg.episode_length_s = 2000.0
            if hasattr(env_cfg, "terminations"):
                if hasattr(env_cfg.terminations, "time_out"):
                    env_cfg.terminations.time_out = None
                if hasattr(env_cfg.terminations, "object_dropping"):
                    env_cfg.terminations.object_dropping = None
            add_camera_to_env_cfg(env_cfg, args.image_width, args.image_height)
            if TASK_ID not in gym.registry:
                gym.register(
                    id=TASK_ID,
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    kwargs={"env_cfg_entry_point": "dummy"},
                    disable_env_checker=True,
                )
            return gym.make(TASK_ID, cfg=env_cfg)

        env = _make_env()
        print(f"Environment created. device={device}")

        # ── load policies ─────────────────────────────────────────────────
        config_A = load_policy_config(args.policy_A)
        config_B = load_policy_config(args.policy_B)

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

        print(f"Policy A loaded: state_dim={config_A['state_dim']}, "
              f"obj_pose={config_A['include_obj_pose']}, "
              f"gripper={config_A['include_gripper']}, "
              f"wrist={config_A['has_wrist']}")
        print(f"Policy B loaded: state_dim={config_B['state_dim']}, "
              f"obj_pose={config_B['include_obj_pose']}, "
              f"gripper={config_B['include_gripper']}, "
              f"wrist={config_B['has_wrist']}")

        # ── cameras ───────────────────────────────────────────────────────
        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

        # ── initial setup ─────────────────────────────────────────────────
        env.reset()
        pre_position_gripper_down(env)

        # ── helper: observation → policy input ────────────────────────────
        def _get_obs_gpu():
            """Return (table_rgb, wrist_rgb, ee_pose) — no obj_pose."""
            table_rgb = table_camera.data.output["rgb"]
            if table_rgb.shape[-1] > 3:
                table_rgb = table_rgb[..., :3]
            wrist_rgb = wrist_camera.data.output["rgb"]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]
            ee_pose = get_ee_pose_w(env)[0]  # (7,)
            return table_rgb, wrist_rgb, ee_pose

        def _build_policy_input(table_rgb, wrist_rgb, ee_pose,
                                gripper_state, include_obj_pose, include_gripper,
                                has_wrist):
            img = table_rgb[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            state_parts = [ee_pose.unsqueeze(0)]
            if include_obj_pose:
                # For backward compat — Exp45 policies should NOT have obj_pose
                # but if loaded policy expects it, get cube_small pose
                obj_pose = get_object_pose_w(env, name="cube_small")[0]
                state_parts.append(obj_pose.unsqueeze(0))
            if include_gripper:
                g = torch.tensor([[gripper_state]], dtype=torch.float32,
                                 device=device)
                state_parts.append(g)
            state = torch.cat(state_parts, dim=-1)
            inputs = {
                "observation.image": img,
                "observation.state": state,
            }
            if has_wrist:
                w = wrist_rgb[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                inputs["observation.wrist_image"] = w
            return inputs

        # ── helper: hold-still action ─────────────────────────────────────
        def _hold(n_steps: int = 10, gripper: float = 1.0):
            ee = get_ee_pose_w(env)
            act = torch.zeros(1, env.action_space.shape[-1], device=device)
            act[0, :7] = ee[0, :7]
            act[0, 7] = gripper
            for _ in range(n_steps):
                env.step(act)

        # ── helper: teleport cubes to stacked config ──────────────────────
        def _teleport_cubes_stacked():
            """Teleport all 3 cubes to vertical stack at stack_pos."""
            sx, sy = float(stack_xy[0]), float(stack_xy[1])
            for cd in CUBE_DEFS_44:
                pose = torch.zeros(1, 7, device=device)
                pose[0, 0] = sx
                pose[0, 1] = sy
                pose[0, 2] = cd.init_z
                pose[0, 3] = 1.0
                teleport_object_to_pose(env, pose, name=cd.scene_key)

        # ── helper: teleport cubes to scattered positions (Task A start) ──
        def _teleport_cubes_scattered():
            """Teleport cube_large to stack, small/medium to fixed scatter."""
            # cube_large stays at stack_pos
            cd_large = get_cube_def_44("cube_large")
            pose = torch.zeros(1, 7, device=device)
            pose[0, 0] = float(stack_xy[0])
            pose[0, 1] = float(stack_xy[1])
            pose[0, 2] = cd_large.init_z
            pose[0, 3] = 1.0
            teleport_object_to_pose(env, pose, name=cd_large.scene_key)

            # small and medium at fixed scatter positions
            for cube_name in PICK_ORDER_44:
                cd = get_cube_def_44(cube_name)
                xy = scatter_positions[cube_name]
                pose = torch.zeros(1, 7, device=device)
                pose[0, 0] = float(xy[0])
                pose[0, 1] = float(xy[1])
                pose[0, 2] = cd.goal_z  # resting z on table
                pose[0, 3] = 1.0
                teleport_object_to_pose(env, pose, name=cd.scene_key)

        # ── helper: hard resets ───────────────────────────────────────────
        def _hard_reset_for_task_A():
            """Reset for Task A: cube_large at stack, small+medium scattered."""
            env.reset()
            pre_position_gripper_down(env)
            _teleport_cubes_scattered()
            _hold(15, gripper=1.0)
            policy_A.reset()

        def _hard_reset_for_task_B():
            """Reset for Task B: all 3 cubes stacked."""
            env.reset()
            pre_position_gripper_down(env)
            _teleport_cubes_stacked()
            _hold(15, gripper=1.0)
            policy_B.reset()

        # ── helper: per-frame success checks ──────────────────────────────

        def _check_task_B_frame() -> bool:
            """Per-frame Task B check: both movable cubes far from stack and on table."""
            for cube_name in PICK_ORDER_44:
                cp = get_object_pose_w(env, name=cube_name)[0].cpu().numpy()
                xy_dist = np.linalg.norm(cp[:2] - stack_xy)
                z = cp[2]
                if xy_dist < args.unstack_xy_threshold or z > args.unstack_z_threshold:
                    return False
            return True

        class TaskATracker:
            """Tracks 2-round stacking success for Task A (per-frame)."""

            def __init__(self):
                self.round = 0  # 0 = round 1 (medium), 1 = round 2 (small)
                self.consecutive_ok = 0
                self.round_targets = [
                    ("cube_medium", z_expected_medium),
                    ("cube_small", z_expected_small),
                ]

            def check_frame(self, gripper_state: float) -> bool:
                """Returns True when BOTH rounds are complete."""
                if self.round >= len(self.round_targets):
                    return True  # already succeeded

                cube_name, z_exp = self.round_targets[self.round]
                cp = get_object_pose_w(env, name=cube_name)[0].cpu().numpy()
                z = cp[2]

                gripper_open = gripper_state > 0.5
                height_ok = abs(z - z_exp) < args.height_tolerance
                round_ok = gripper_open and height_ok

                if round_ok:
                    self.consecutive_ok += 1
                else:
                    self.consecutive_ok = 0

                if self.consecutive_ok >= args.stable_frames:
                    print(f"      Round {self.round + 1} done: {cube_name} stable "
                          f"at z={z:.4f} (expected {z_exp:.4f}) for {args.stable_frames} frames")
                    self.round += 1
                    self.consecutive_ok = 0
                    if self.round >= len(self.round_targets):
                        return True
                return False

        # ── helper: run one task ──────────────────────────────────────────
        def _run_task(policy, preprocessor, postprocessor, n_action_steps,
                      task_name, include_obj_pose, include_gripper, has_wrist,
                      is_task_A: bool):
            """Run a single task rollout. Returns (episode_dict, success)."""
            policy.reset()
            gripper_state = 1.0  # open

            images_list, wrist_images_list = [], []
            ee_pose_list, action_list = [], []
            cube_poses = {cd.scene_key: [] for cd in CUBE_DEFS_44}

            success = False
            success_step = None
            buffer_remaining = -1  # -1 means not in buffer mode

            # Task-specific tracker
            task_a_tracker = TaskATracker() if is_task_A else None

            for t in range(args.horizon):
                table_rgb, wrist_rgb, ee_pose_gpu = _get_obs_gpu()

                inputs = _build_policy_input(
                    table_rgb, wrist_rgb, ee_pose_gpu,
                    gripper_state, include_obj_pose, include_gripper, has_wrist,
                )

                if t == 0:
                    sd = inputs["observation.state"].shape[-1]
                    print(f"    [{task_name}] state_dim={sd}")

                if preprocessor is not None:
                    inputs = preprocessor(inputs)

                with torch.no_grad():
                    action = policy.select_action(inputs)
                if postprocessor is not None:
                    action = postprocessor(action)

                # CPU copies for recording
                table_np = table_rgb[0].cpu().numpy().astype(np.uint8)
                wrist_np = wrist_rgb[0].cpu().numpy().astype(np.uint8)
                ee_np = ee_pose_gpu.cpu().numpy()
                action_np = action[0].cpu().numpy()

                images_list.append(table_np)
                wrist_images_list.append(wrist_np)
                ee_pose_list.append(ee_np)
                action_list.append(action_np)

                for cd in CUBE_DEFS_44:
                    cp = get_object_pose_w(env, name=cd.scene_key)[0].cpu().numpy()
                    cube_poses[cd.scene_key].append(cp)

                gripper_state = float(action_np[7])

                # Execute
                env.step(action)

                # Per-frame success check
                if not success:
                    if is_task_A:
                        if task_a_tracker.check_frame(gripper_state):
                            success = True
                            success_step = t + 1
                            buffer_remaining = args.early_stop_buffer
                            print(f"    [{task_name}] SUCCESS at step {t+1}, starting {args.early_stop_buffer}-frame buffer")
                    else:
                        if _check_task_B_frame():
                            success = True
                            success_step = t + 1
                            buffer_remaining = args.early_stop_buffer
                            print(f"    [{task_name}] SUCCESS at step {t+1}, starting {args.early_stop_buffer}-frame buffer")

                # Early stop after buffer
                if buffer_remaining > 0:
                    buffer_remaining -= 1
                elif buffer_remaining == 0:
                    break

                if (t + 1) % 500 == 0:
                    print(f"    [{task_name}] Step {t+1}/{args.horizon}")

            if not success:
                print(f"    [{task_name}] FAILED (no success within {args.horizon} steps)")

            episode = {
                "images": np.array(images_list, dtype=np.uint8),
                "wrist_images": np.array(wrist_images_list, dtype=np.uint8),
                "ee_pose": np.array(ee_pose_list, dtype=np.float32),
                "action": np.array(action_list, dtype=np.float32),
                "cube_large_pose": np.array(cube_poses["cube_large"], dtype=np.float32),
                "cube_medium_pose": np.array(cube_poses["cube_medium"], dtype=np.float32),
                "cube_small_pose": np.array(cube_poses["cube_small"], dtype=np.float32),
                "success": success,
                "success_step": success_step,
            }
            return episode, success

        # ── main loop ─────────────────────────────────────────────────────
        results_A, results_B = [], []
        episodes_A, episodes_B = [], []
        start_time = time.time()

        # First reset for Task A
        _hard_reset_for_task_A()

        for cycle in range(args.num_cycles):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{args.num_cycles}")
            print(f"{'='*50}")

            # Task A (stack)
            ep_A, succ_A = _run_task(
                policy_A, preproc_A, postproc_A, n_act_A,
                "Task A (stack)",
                config_A["include_obj_pose"], config_A["include_gripper"],
                config_A["has_wrist"],
                is_task_A=True,
            )
            episodes_A.append(ep_A)
            results_A.append(succ_A)
            print(f"  Task A: {'SUCCESS' if succ_A else 'FAILED'}")

            # Hard reset for Task B
            _hard_reset_for_task_B()

            # Task B (unstack)
            ep_B, succ_B = _run_task(
                policy_B, preproc_B, postproc_B, n_act_B,
                "Task B (unstack)",
                config_B["include_obj_pose"], config_B["include_gripper"],
                config_B["has_wrist"],
                is_task_A=False,
            )
            episodes_B.append(ep_B)
            results_B.append(succ_B)
            print(f"  Task B: {'SUCCESS' if succ_B else 'FAILED'}")

            # Hard reset for next Task A
            _hard_reset_for_task_A()

            a_rate = sum(results_A) / len(results_A) * 100
            b_rate = sum(results_B) / len(results_B) * 100
            print(f"  Running rates: A={a_rate:.1f}% ({sum(results_A)}/{len(results_A)})  "
                  f"B={b_rate:.1f}% ({sum(results_B)}/{len(results_B)})")

        elapsed = time.time() - start_time
        n_succ_A = sum(results_A)
        n_succ_B = sum(results_B)
        rate_A = n_succ_A / len(results_A) if results_A else 0.0
        rate_B = n_succ_B / len(results_B) if results_B else 0.0

        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"  Cycles:         {args.num_cycles}")
        print(f"  Task A success: {n_succ_A}/{len(results_A)} = {rate_A:.1%}")
        print(f"  Task B success: {n_succ_B}/{len(results_B)} = {rate_B:.1%}")
        print(f"  Time:           {elapsed:.1f}s")

        # ── save episodes ─────────────────────────────────────────────────
        for out_path, episodes, label in [
            (args.out_A, episodes_A, "Task A"),
            (args.out_B, episodes_B, "Task B"),
        ]:
            if args.save_all:
                to_save = episodes
            else:
                to_save = [e for e in episodes if e["success"]]
            if to_save:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_path,
                                    episodes=np.array(to_save, dtype=object))
                n_s = sum(1 for e in to_save if e["success"])
                print(f"Saved {len(to_save)} {label} episodes "
                      f"({n_s} success) to {out_path}")

        # ── save stats JSON ───────────────────────────────────────────────
        stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_A": args.policy_A,
                "policy_B": args.policy_B,
                "num_cycles": args.num_cycles,
                "horizon": args.horizon,
                "height_tolerance": args.height_tolerance,
                "stable_frames": args.stable_frames,
                "unstack_xy_threshold": args.unstack_xy_threshold,
                "unstack_z_threshold": args.unstack_z_threshold,
                "early_stop_buffer": args.early_stop_buffer,
                "n_action_steps": args.n_action_steps,
            },
            "summary": {
                "total_cycles": args.num_cycles,
                "task_A_success_count": n_succ_A,
                "task_B_success_count": n_succ_B,
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

        # ── save video ────────────────────────────────────────────────────
        if args.save_video:
            print("Note: video saving not implemented for Exp45 cyclic eval yet.")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
