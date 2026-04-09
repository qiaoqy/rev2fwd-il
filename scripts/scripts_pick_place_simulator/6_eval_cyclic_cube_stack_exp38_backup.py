#!/usr/bin/env python3
"""Step 6: Cyclic A→B evaluation for 3-cube stack task (Exp38).

Runs N A→B cycles.  After every task (success or failure) the environment is
hard-reset:
  - Task A (stack):   3 cubes scattered in the red region → policy stacks on green marker
  - Task B (unstack): 3 cubes stacked at green marker → policy places in red region

Success criteria
  Task A: all 3 cubes within *goal_threshold* of goal_xy
  Task B: all 3 cubes inside the red rectangular region

The ``obj_pose`` fed to the policy is always ``cube_small``'s pose (matches the
training data convention).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/6_eval_cyclic_cube_stack.py \
        --policy_A <path_to_PP_A_pretrained_model> \
        --policy_B <path_to_PP_B_pretrained_model> \
        --out_A data/.../eval_A.npz \
        --out_B data/.../eval_B.npz \
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


TASK_ID = "Isaac-Lift-CubeStack-Franka-IK-Abs-v0"


# ── argument parsing ──────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cyclic A→B evaluation for 3-cube stack (Exp38).",
    )

    # Policies
    parser.add_argument("--policy_A", type=str, required=True)
    parser.add_argument("--policy_B", type=str, required=True)

    # Output
    parser.add_argument("--out_A", type=str, required=True)
    parser.add_argument("--out_B", type=str, required=True)

    # Test parameters
    parser.add_argument("--num_cycles", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1500,
                        help="Max steps per task (default: 1500).")
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--goal_threshold", type=float, default=0.05,
                        help="Max dist (m) from goal_xy for Task A success.")

    # Region
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2,
                        default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2,
                        default=[0.30, 0.30])

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


# ── placement sampling (same as collection script) ────────────────────────

def sample_place_targets(rng, red_center, red_size, cube_defs_pick_order,
                         min_sep: float = 0.10):
    """Sample 3 placement XY positions mutually >= *min_sep* apart."""
    cx, cy = float(red_center[0]), float(red_center[1])
    sx, sy = float(red_size[0]), float(red_size[1])
    max_half = max(c.edge_length for c in cube_defs_pick_order) / 2
    safe_hx = sx / 2 - max_half
    safe_hy = sy / 2 - max_half

    placed: list[tuple[float, float]] = []
    for _ in range(3):
        for _attempt in range(200):
            x = rng.uniform(cx - safe_hx, cx + safe_hx)
            y = rng.uniform(cy - safe_hy, cy + safe_hy)
            if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) >= min_sep
                   for px, py in placed):
                placed.append((x, y))
                break
        else:
            grid = [
                (cx - 0.08, cy - 0.08),
                (cx + 0.08, cy + 0.08),
                (cx + 0.08, cy - 0.08),
            ]
            placed.append(grid[len(placed)])
    return placed


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
        from rev2fwd_il.sim.cube_stack_registry import (
            CUBE_DEFS,
            DEFAULT_STACK_CONFIG,
            PICK_ORDER,
            get_cube_def,
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
        create_target_markers = _alt_mod.create_target_markers
        update_target_markers = _alt_mod.update_target_markers

        # Load cube stack env cfg
        _env_cfg_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based"
            / "manipulation" / "lift" / "config" / "franka"
            / "cube_stack_env_cfg.py"
        )
        _env_cfg_spec = importlib.util.spec_from_file_location(
            "cube_stack_env_cfg", _env_cfg_path)
        _env_cfg_mod = importlib.util.module_from_spec(_env_cfg_spec)
        _env_cfg_spec.loader.exec_module(_env_cfg_mod)
        FrankaCubeStackEnvCfg = _env_cfg_mod.FrankaCubeStackEnvCfg

        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        stack_cfg = DEFAULT_STACK_CONFIG
        pick_order_defs = [get_cube_def(n) for n in PICK_ORDER]
        goal_xy = np.array(args.goal_xy, dtype=np.float64)

        # ── create env ────────────────────────────────────────────────────
        def _make_env():
            env_cfg = FrankaCubeStackEnvCfg()
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

        # ── markers ───────────────────────────────────────────────────────
        env.reset()
        pre_position_gripper_down(env)
        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
            red_marker_shape="rectangle",
            red_marker_size_xy=tuple(args.red_region_size_xy),
        )
        red_center = tuple(args.red_region_center_xy)
        update_target_markers(
            place_markers, goal_markers,
            place_xy=red_center,
            goal_xy=tuple(args.goal_xy),
            marker_z=marker_z,
            env=env,
        )

        # ── helper: observation → policy input ────────────────────────────
        def _get_obs_gpu():
            """Return (table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state)."""
            table_rgb = table_camera.data.output["rgb"]
            if table_rgb.shape[-1] > 3:
                table_rgb = table_rgb[..., :3]
            wrist_rgb = wrist_camera.data.output["rgb"]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]
            ee_pose = get_ee_pose_w(env)[0]  # (7,)
            obj_pose = get_object_pose_w(env, name="cube_small")[0]  # (7,)
            return table_rgb, wrist_rgb, ee_pose, obj_pose

        def _build_policy_input(table_rgb, wrist_rgb, ee_pose, obj_pose,
                                gripper_state, include_obj_pose, include_gripper,
                                has_wrist):
            img = table_rgb[0].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            state_parts = [ee_pose.unsqueeze(0)]
            if include_obj_pose:
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

        # ── helper: teleport all cubes to stacked config ──────────────────
        def _teleport_cubes_stacked():
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            for cd in CUBE_DEFS:
                pose = torch.zeros(1, 7, device=device)
                pose[0, 0] = gx
                pose[0, 1] = gy
                pose[0, 2] = cd.init_z
                pose[0, 3] = 1.0
                teleport_object_to_pose(env, pose, name=cd.scene_key)

        # ── helper: teleport all cubes to red region (scattered) ──────────
        def _teleport_cubes_scattered():
            pts = sample_place_targets(
                rng, args.red_region_center_xy, args.red_region_size_xy,
                pick_order_defs, min_sep=stack_cfg.min_place_separation,
            )
            for i, cube_name in enumerate(PICK_ORDER):
                cd = get_cube_def(cube_name)
                pose = torch.zeros(1, 7, device=device)
                pose[0, 0] = pts[i][0]
                pose[0, 1] = pts[i][1]
                pose[0, 2] = cd.goal_z  # resting z on table
                pose[0, 3] = 1.0
                teleport_object_to_pose(env, pose, name=cd.scene_key)

        # ── helper: hard resets ───────────────────────────────────────────
        def _hard_reset_for_task_A():
            """Reset for Task A: cubes scattered in red region."""
            env.reset()
            pre_position_gripper_down(env)
            update_target_markers(
                place_markers, goal_markers,
                place_xy=red_center,
                goal_xy=tuple(args.goal_xy),
                marker_z=marker_z, env=env,
            )
            _teleport_cubes_scattered()
            _hold(15, gripper=1.0)
            policy_A.reset()

        def _hard_reset_for_task_B():
            """Reset for Task B: cubes stacked at green marker."""
            env.reset()
            pre_position_gripper_down(env)
            update_target_markers(
                place_markers, goal_markers,
                place_xy=red_center,
                goal_xy=tuple(args.goal_xy),
                marker_z=marker_z, env=env,
            )
            _teleport_cubes_stacked()
            _hold(15, gripper=1.0)
            policy_B.reset()

        # ── helper: success checks ────────────────────────────────────────
        def _check_task_A_success() -> bool:
            """Task A success: all 3 cubes near goal_xy."""
            for cd in CUBE_DEFS:
                cp = get_object_pose_w(env, name=cd.scene_key)[0, :2].cpu().numpy()
                if np.linalg.norm(cp - goal_xy) > args.goal_threshold:
                    return False
            return True

        def _check_task_B_success() -> bool:
            """Task B success: all 3 cubes inside red rectangle."""
            rcx = float(args.red_region_center_xy[0])
            rcy = float(args.red_region_center_xy[1])
            rsx = float(args.red_region_size_xy[0])
            rsy = float(args.red_region_size_xy[1])
            for cube_name in PICK_ORDER:
                cp = get_object_pose_w(env, name=cube_name)[0, :2].cpu().numpy()
                if not (abs(cp[0] - rcx) <= rsx / 2 and abs(cp[1] - rcy) <= rsy / 2):
                    return False
            return True

        # ── helper: run one task ──────────────────────────────────────────
        def _run_task(policy, preprocessor, postprocessor, n_action_steps,
                      check_success_fn, task_name,
                      include_obj_pose, include_gripper, has_wrist):
            """Run a single task rollout. Returns (episode_dict, success)."""
            policy.reset()
            gripper_state = 1.0  # open

            images_list, wrist_images_list = [], []
            ee_pose_list, obj_pose_list, action_list = [], [], []
            video_frames = []
            cube_poses = {cd.scene_key: [] for cd in CUBE_DEFS}

            success = False
            success_step = None

            for t in range(args.horizon):
                table_rgb, wrist_rgb, ee_pose_gpu, obj_pose_gpu = _get_obs_gpu()

                inputs = _build_policy_input(
                    table_rgb, wrist_rgb, ee_pose_gpu, obj_pose_gpu,
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
                obj_np = obj_pose_gpu.cpu().numpy()
                action_np = action[0].cpu().numpy()

                images_list.append(table_np)
                wrist_images_list.append(wrist_np)
                ee_pose_list.append(ee_np)
                obj_pose_list.append(obj_np)
                action_list.append(action_np)
                video_frames.append(table_np.copy())

                for cd in CUBE_DEFS:
                    cp = get_object_pose_w(env, name=cd.scene_key)[0].cpu().numpy()
                    cube_poses[cd.scene_key].append(cp)

                gripper_state = float(action_np[7])

                # Execute
                env.step(action)

                # Check success
                if not success and check_success_fn():
                    success = True
                    success_step = t + 1
                    print(f"    [{task_name}] SUCCESS at step {t+1}")
                    # Record 50 more frames then break
                    for _ in range(50):
                        tbl, wst, ee_g, obj_g = _get_obs_gpu()
                        inp = _build_policy_input(
                            tbl, wst, ee_g, obj_g, gripper_state,
                            include_obj_pose, include_gripper, has_wrist,
                        )
                        if preprocessor is not None:
                            inp = preprocessor(inp)
                        with torch.no_grad():
                            a = policy.select_action(inp)
                        if postprocessor is not None:
                            a = postprocessor(a)

                        images_list.append(tbl[0].cpu().numpy().astype(np.uint8))
                        wrist_images_list.append(wst[0].cpu().numpy().astype(np.uint8))
                        ee_pose_list.append(ee_g.cpu().numpy())
                        obj_pose_list.append(obj_g.cpu().numpy())
                        action_list.append(a[0].cpu().numpy())
                        video_frames.append(tbl[0].cpu().numpy().astype(np.uint8))
                        for cd in CUBE_DEFS:
                            cp = get_object_pose_w(env, name=cd.scene_key)[0].cpu().numpy()
                            cube_poses[cd.scene_key].append(cp)
                        gripper_state = float(a[0].cpu().numpy()[7])
                        env.step(a)
                    break

                if (t + 1) % 200 == 0:
                    print(f"    [{task_name}] Step {t+1}/{args.horizon}")

            episode = {
                "images": np.array(images_list, dtype=np.uint8),
                "wrist_images": np.array(wrist_images_list, dtype=np.uint8),
                "ee_pose": np.array(ee_pose_list, dtype=np.float32),
                "obj_pose": np.array(obj_pose_list, dtype=np.float32),
                "action": np.array(action_list, dtype=np.float32),
                "cube_large_pose": np.array(cube_poses["cube_large"], dtype=np.float32),
                "cube_medium_pose": np.array(cube_poses["cube_medium"], dtype=np.float32),
                "cube_small_pose": np.array(cube_poses["cube_small"], dtype=np.float32),
                "success": success,
                "success_step": success_step,
            }
            return episode, success, video_frames

        # ── main loop ─────────────────────────────────────────────────────
        results_A, results_B = [], []
        episodes_A, episodes_B = [], []
        all_video_frames = []
        start_time = time.time()

        # First reset for Task A
        _hard_reset_for_task_A()

        for cycle in range(args.num_cycles):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{args.num_cycles}")
            print(f"{'='*50}")

            # Task A (stack)
            ep_A, succ_A, vid_A = _run_task(
                policy_A, preproc_A, postproc_A, n_act_A,
                _check_task_A_success, "Task A (stack)",
                config_A["include_obj_pose"], config_A["include_gripper"],
                config_A["has_wrist"],
            )
            episodes_A.append(ep_A)
            results_A.append(succ_A)
            all_video_frames.extend(vid_A)
            print(f"  Task A: {'SUCCESS' if succ_A else 'FAILED'}")

            # Hard reset for Task B
            _hard_reset_for_task_B()

            # Task B (unstack)
            ep_B, succ_B, vid_B = _run_task(
                policy_B, preproc_B, postproc_B, n_act_B,
                _check_task_B_success, "Task B (unstack)",
                config_B["include_obj_pose"], config_B["include_gripper"],
                config_B["has_wrist"],
            )
            episodes_B.append(ep_B)
            results_B.append(succ_B)
            all_video_frames.extend(vid_B)
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
                "goal_threshold": args.goal_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
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
        if args.save_video and all_video_frames:
            import imageio
            video_path = Path(args.video_path or args.out_A).with_suffix(".mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=args.video_fps)
            for frame in all_video_frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Saved video to: {video_path}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
