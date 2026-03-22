#!/usr/bin/env python3
"""Step 8: Failure analysis evaluation — Task A or Task B.

Runs N independent episodes for a single task, saves all rollout data,
and renders annotated MP4 videos for failed episodes.

Merges the old ``12_eval_failure_analysis.py`` (Task B) and
``13_eval_failure_analysis_A.py`` (Task A) into a single script selected
via ``--task A`` or ``--task B``.

Mode 3 only (red rectangle region).

Usage:
    conda activate rev2fwd_il

    # Task A failure analysis (quick test, 10 episodes)
    CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place_simulator/8_eval_failure_analysis.py \\
        --policy data/pick_place_isaac_lab_simulation/exp17/weights/PP_A/checkpoints/checkpoints/last/pretrained_model \\
        --task A --run_id gpu1 \\
        --out_dir data/pick_place_isaac_lab_simulation/exp17 \\
        --headless --render_success_videos

    # Task B failure analysis (quick test, 10 episodes)
    CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place_simulator/8_eval_failure_analysis.py \\
        --policy data/pick_place_isaac_lab_simulation/exp17/weights/PP_B/checkpoints/checkpoints/last/pretrained_model \\
        --task B --run_id gpu1 \\
        --out_dir data/pick_place_isaac_lab_simulation/exp17 \\
        --headless --render_success_videos

    # Full evaluation (100 episodes)
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_simulator/8_eval_failure_analysis.py \\
        --policy <checkpoint_path> \\
        --task A --num_episodes 100 --run_id gpu0 \\
        --out_dir <out_dir> --headless
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch


# =========================================================================
# Video annotation helpers
# =========================================================================
def add_text_overlay(img: np.ndarray, text: str,
                     position=(4, 16), font_scale=0.45,
                     color=(255, 255, 255), bg_color=(0, 0, 0)) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2),
                  bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)
    return img


def add_multi_line_overlay(img: np.ndarray, lines: list[str],
                           start_y: int = 4, line_height: int = 16,
                           font_scale: float = 0.40,
                           color=(255, 255, 255),
                           bg_color=(0, 0, 0)) -> np.ndarray:
    for i, text in enumerate(lines):
        y = start_y + (i + 1) * line_height
        img = add_text_overlay(img, text, position=(4, y),
                               font_scale=font_scale, color=color,
                               bg_color=bg_color)
    return img


def render_episode_video(
    frames: list[np.ndarray],
    out_path: str | Path,
    ep_index: int,
    success: bool,
    success_step: int | None,
    target_xy: tuple[float, float],
    obj_poses: np.ndarray,
    ee_poses: np.ndarray,
    actions: np.ndarray,
    distance_threshold: float,
    goal_xy: tuple[float, float],
    task_type: str,
    fps: int = 30,
    region_center_xy: tuple[float, float] | None = None,
    region_size_xy: tuple[float, float] | None = None,
    wrist_frames: list[np.ndarray] | None = None,
) -> None:
    """Render annotated MP4 for a single episode."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(frames)
    H, W = frames[0].shape[:2]
    scale = max(1, 384 // W)
    out_H, out_W = H * scale, W * scale
    has_wrist = (wrist_frames is not None and len(wrist_frames) > 0)

    annotated = []
    for t in range(T):
        frame = frames[t]
        if scale > 1:
            frame = cv2.resize(frame, (out_W, out_H),
                               interpolation=cv2.INTER_NEAREST)

        obj_xy = obj_poses[t, :2] if t < len(obj_poses) else [0, 0]
        obj_z = obj_poses[t, 2] if t < len(obj_poses) else 0
        gripper = actions[t, 7] if t < len(actions) else 0
        gripper_str = "OPEN" if gripper > 0.5 else "CLOSED"

        lines = [
            f"Ep {ep_index} | Step {t+1}/{T}  Task {task_type}",
            f"ObjXY: ({obj_xy[0]:.3f}, {obj_xy[1]:.3f})  Z:{obj_z:.3f}",
        ]

        if task_type == "B" and region_center_xy is not None and region_size_xy is not None:
            rcx, rcy = region_center_xy
            rsx, rsy = region_size_xy
            dx = abs(float(obj_xy[0]) - rcx) - rsx * 0.5
            dy = abs(float(obj_xy[1]) - rcy) - rsy * 0.5
            in_x = dx <= 0
            in_y = dy <= 0
            lines.append(f"Region: cx{rcx:.2f} cy{rcy:.2f} "
                         f"sx{rsx:.2f} sy{rsy:.2f}")
            lines.append(f"dX:{dx:+.3f}{'OK' if in_x else ''} "
                         f"dY:{dy:+.3f}{'OK' if in_y else ''}")
        else:
            dist = np.linalg.norm(np.array(obj_xy) - np.array(goal_xy))
            lines.append(f"Dist to goal: {dist:.4f}  Thr: {distance_threshold}")

        lines.append(f"Gripper: {gripper_str} ({gripper:.2f})")

        if success and success_step is not None and t + 1 >= success_step:
            lines.append("STATUS: SUCCESS")
        else:
            lines.append("STATUS: RUNNING..." if not success else "STATUS: SUCCESS")

        frame = add_multi_line_overlay(frame, lines, start_y=2,
                                       line_height=int(14 * scale / 3 + 2),
                                       font_scale=0.35 * scale / 3 + 0.1)

        # Progress bar
        bar_h = max(4, scale * 2)
        bar_y = out_H - bar_h - 2
        progress = (t + 1) / T
        bar_w = int(progress * (out_W - 8))
        bar_color = (0, 200, 0) if success else (0, 100, 200)
        cv2.rectangle(frame, (4, bar_y), (4 + bar_w, bar_y + bar_h),
                      bar_color, -1)
        cv2.rectangle(frame, (4, bar_y), (out_W - 4, bar_y + bar_h),
                      (128, 128, 128), 1)

        # Final frame overlay
        if t == T - 1:
            result_text = "FAILED" if not success else f"SUCCESS (step {success_step})"
            result_color = (0, 0, 255) if not success else (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.6 * scale / 3 + 0.2
            (tw, th), _ = cv2.getTextSize(result_text, font, fs, 2)
            cx, cy = (out_W - tw) // 2, out_H // 2
            cv2.rectangle(frame, (cx - 6, cy - th - 6),
                          (cx + tw + 6, cy + 10), (0, 0, 0), -1)
            cv2.putText(frame, result_text, (cx, cy), font, fs,
                        result_color, 2, cv2.LINE_AA)

        # Side-by-side with wrist camera
        if has_wrist and t < len(wrist_frames):
            wrist = wrist_frames[t]
            if scale > 1:
                wrist = cv2.resize(wrist, (out_W, out_H),
                                   interpolation=cv2.INTER_NEAREST)
            elif wrist.shape[:2] != (out_H, out_W):
                wrist = cv2.resize(wrist, (out_W, out_H))
            # Add "Wrist" label
            wrist = add_text_overlay(wrist, "Wrist Camera",
                                     position=(4, 16),
                                     font_scale=0.35 * scale / 3 + 0.1)
            frame = np.concatenate([frame, wrist], axis=1)

        annotated.append(frame)

    imageio.mimsave(str(out_path), annotated, fps=fps)


# =========================================================================
# Argument parser
# =========================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Failure analysis evaluation (Task A or B, Mode 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Policy checkpoint path.")
    parser.add_argument("--task", type=str, required=True, choices=["A", "B"],
                        help="Task to evaluate: A or B.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--run_id", type=str, default="run0")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--distance_threshold", type=float, default=0.03)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--render_success_videos", action="store_true")

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

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    if args.seed is None:
        args.seed = hash(args.run_id) % (2**31)
    return args


# =========================================================================
# Main
# =========================================================================
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
        update_target_markers = _alt_mod.update_target_markers

        set_seed(args.seed)
        device = args.device if args.device else (
            "cuda" if torch.cuda.is_available() else "cpu")

        task_label = args.task
        print(f"\n{'='*60}")
        print(f"  Failure Analysis — Task {task_label} ({args.run_id})")
        print(f"  Policy: {args.policy}")
        print(f"  Episodes: {args.num_episodes}")
        print(f"{'='*60}\n")

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

        # Fill both slots with the same policy
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

        first_place_xy = tester._sample_new_place_target()
        tester.current_place_xy = first_place_xy
        marker_xy = tuple(args.red_region_center_xy)
        update_target_markers(
            place_markers, goal_markers,
            marker_xy, tuple(goal_xy), marker_z, env,
        )

        # Output
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_dir = out_dir / f"failure_videos_{task_label}_{args.run_id}"
        video_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / f"failure_analysis_{task_label}_{args.run_id}.json"
        npz_path = out_dir / f"failure_analysis_{task_label}_{args.run_id}.npz"

        # ---- Run episodes ----
        results = []
        episode_details = []
        all_episodes_data = {}
        start_time = time.time()

        for ep_idx in range(args.num_episodes):
            # Hard reset
            env.reset()
            pre_position_gripper_down(env)

            if task_label == "A":
                # Object at random position in red region
                rand_xy = tester._sample_taskA_source_target()
                tester.current_place_xy = rand_xy
                update_target_markers(
                    place_markers, goal_markers,
                    marker_xy, tuple(goal_xy), marker_z, env,
                )
                obj_pose = torch.tensor(
                    [rand_xy[0], rand_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                teleport_object_to_pose(env, obj_pose, name="object")
                source_xy = rand_xy
            else:
                # Object at goal position
                new_place_xy = tester._sample_new_place_target()
                tester.current_place_xy = new_place_xy
                update_target_markers(
                    place_markers, goal_markers,
                    marker_xy, tuple(goal_xy), marker_z, env,
                )
                obj_pose = torch.tensor(
                    [goal_xy[0], goal_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                teleport_object_to_pose(env, obj_pose, name="object")
                source_xy = goal_xy

            # Settle
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[0, :7] = ee_hold[0, :7]
            hold_action[0, 7] = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy.reset()

            print(f"\n  [Task {task_label}] Ep {ep_idx + 1}/{args.num_episodes}  "
                  f"obj=[{source_xy[0]:.3f}, {source_xy[1]:.3f}]")

            # Run task
            if task_label == "A":
                ep_data, success = tester.run_task_A()
            else:
                ep_data, success = tester.run_task_B()

            results.append(success)
            rate = sum(results) / len(results) * 100
            print(f"    {'SUCCESS' if success else 'FAILED'}  "
                  f"(running: {sum(results)}/{len(results)} = {rate:.1f}%)")

            # Store details
            T = len(ep_data.get("images", []))
            detail = {
                "episode_index": ep_idx,
                "success": bool(success),
                "success_step": ep_data.get("success_step"),
                "total_steps": T,
            }
            if "obj_pose" in ep_data and len(ep_data["obj_pose"]) > 0:
                detail["initial_obj_position"] = ep_data["obj_pose"][0][:3].tolist()
                detail["final_obj_position"] = ep_data["obj_pose"][-1][:3].tolist()
                final_xy = ep_data["obj_pose"][-1][:2]
                detail["final_dist_to_goal"] = float(
                    np.linalg.norm(final_xy - goal_xy))
            episode_details.append(detail)

            # Save episode data for NPZ
            prefix = f"ep_{ep_idx}"
            all_episodes_data[f"{prefix}_images"] = ep_data["images"]
            all_episodes_data[f"{prefix}_ee_pose"] = ep_data["ee_pose"]
            all_episodes_data[f"{prefix}_obj_pose"] = ep_data["obj_pose"]
            all_episodes_data[f"{prefix}_action"] = ep_data["action"]
            all_episodes_data[f"{prefix}_success"] = np.array(success)
            if ep_data.get("wrist_images") is not None:
                all_episodes_data[f"{prefix}_wrist_images"] = ep_data["wrist_images"]

            # Render video for failed episodes
            should_render = (not success) or args.render_success_videos
            if should_render:
                tag = "fail" if not success else "success"
                vid_path = video_dir / f"{tag}_ep{ep_idx:03d}.mp4"
                print(f"    Rendering video -> {vid_path}")

                if task_label == "B":
                    target = (float(tester.current_place_xy[0]),
                              float(tester.current_place_xy[1]))
                else:
                    target = tuple(args.goal_xy)

                wrist_imgs = (list(ep_data["wrist_images"])
                              if ep_data.get("wrist_images") is not None
                              else None)
                render_episode_video(
                    frames=list(ep_data["images"]),
                    out_path=vid_path,
                    ep_index=ep_idx,
                    success=success,
                    success_step=ep_data.get("success_step"),
                    target_xy=target,
                    obj_poses=ep_data["obj_pose"],
                    ee_poses=ep_data["ee_pose"],
                    actions=ep_data["action"],
                    distance_threshold=args.distance_threshold,
                    goal_xy=tuple(args.goal_xy),
                    task_type=task_label,
                    fps=args.video_fps,
                    region_center_xy=(tuple(args.red_region_center_xy)
                                      if task_label == "B" else None),
                    region_size_xy=(tuple(args.red_region_size_xy)
                                    if task_label == "B" else None),
                    wrist_frames=wrist_imgs,
                )

        elapsed = time.time() - start_time

        # ---- Summary ----
        n_success = sum(results)
        n_fail = len(results) - n_success
        rate = n_success / len(results) if results else 0.0

        print(f"\n{'='*60}")
        print(f"  Failure Analysis — Task {task_label} ({args.run_id})")
        print(f"{'='*60}")
        print(f"  Total: {len(results)}  Success: {n_success}  Failed: {n_fail}")
        print(f"  Success rate: {rate:.1%}")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Save stats
        stats = {
            "experiment": f"failure_analysis_task_{task_label}",
            "timestamp": datetime.now().isoformat(),
            "run_id": args.run_id,
            "config": {
                "policy": args.policy,
                "task": task_label,
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": list(args.goal_xy),
                "red_region_center_xy": list(args.red_region_center_xy),
                "red_region_size_xy": list(args.red_region_size_xy),
                "seed": args.seed,
            },
            "summary": {
                "total_episodes": len(results),
                "success_count": n_success,
                "fail_count": n_fail,
                "success_rate": rate,
            },
            "episodes": episode_details,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats saved: {stats_path}")

        # Save NPZ
        all_episodes_data["num_episodes"] = np.array(args.num_episodes)
        np.savez_compressed(str(npz_path), **all_episodes_data)
        print(f"  NPZ saved:  {npz_path}")
        print(f"  Videos:     {video_dir}/")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
