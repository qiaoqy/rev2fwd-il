#!/usr/bin/env python3
"""Step 12: Failure analysis evaluation — Task B only.

Run N independent Task B episodes with proper env.reset() before each episode.
Save all episode data (rollout NPZ) and render each failed episode as an
annotated MP4 video for manual failure analysis.

This script is designed to be run with a SINGLE task type (Task B) so that
two instances can be launched in parallel on different GPUs.

=============================================================================
OUTPUT
=============================================================================
Given --out_dir <dir> and --run_id <id>:
  <dir>/failure_analysis_<id>.json        — Per-episode stats + summary
  <dir>/failure_analysis_<id>.npz         — All episode rollout data
  <dir>/failure_videos_<id>/fail_ep<N>.mp4 — Annotated video per failed episode

=============================================================================
USAGE
=============================================================================
# GPU 0: episodes 0-99
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/12_eval_failure_analysis.py \
    --policy_B data/.../iter10_ckpt_B \
    --num_episodes 100 --run_id gpu0 \
    --out_dir data/pick_place_isaac_lab_simulation/exp4 \
    --headless

# GPU 2: episodes 100-199
CUDA_VISIBLE_DEVICES=2 python scripts/scripts_pick_place/12_eval_failure_analysis.py \
    --policy_B data/.../iter10_ckpt_B \
    --num_episodes 100 --run_id gpu1 \
    --out_dir data/pick_place_isaac_lab_simulation/exp4 \
    --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import imageio
import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task B failure analysis evaluation with video rendering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy_B", type=str, required=True,
                        help="Path to Task B policy checkpoint.")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of Task B episodes to run.")
    parser.add_argument("--run_id", type=str, default="run0",
                        help="Run identifier (for parallel execution).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results.")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--distance_threshold", type=float, default=0.05)
    parser.add_argument("--height_threshold", type=float, default=0.15)
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, 0.0])
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Default: derived from run_id hash.")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--render_success_videos", action="store_true",
                        help="Also render success episodes (default: failures only).")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    if args.seed is None:
        args.seed = hash(args.run_id) % (2**31)
    return args


def add_text_overlay(img: np.ndarray, text: str,
                     position=(4, 16), font_scale=0.45,
                     color=(255, 255, 255), bg_color=(0, 0, 0)) -> np.ndarray:
    """Add text with background rectangle to image."""
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
    """Add multiple lines of annotated text to top-left of image."""
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
    fps: int = 30,
) -> None:
    """Render a single episode as an annotated MP4 video.

    Annotations include:
    - Episode number, step counter, total steps
    - Target XY (red marker position)
    - Object XY position, distance to target
    - Gripper state (open/closed)
    - Success/failure status + progress bar
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(frames)
    H, W = frames[0].shape[:2]

    # Upscale for readability (128→384)
    scale = max(1, 384 // W)
    out_H, out_W = H * scale, W * scale

    annotated_frames = []
    for t in range(T):
        frame = frames[t]
        if scale > 1:
            frame = cv2.resize(frame, (out_W, out_H),
                               interpolation=cv2.INTER_NEAREST)

        # --- Gather info for annotation ---
        obj_xy = obj_poses[t, :2] if t < len(obj_poses) else [0, 0]
        obj_z = obj_poses[t, 2] if t < len(obj_poses) else 0
        ee_xy = ee_poses[t, :2] if t < len(ee_poses) else [0, 0]
        gripper = actions[t, 7] if t < len(actions) else 0
        dist_to_target = np.linalg.norm(np.array(obj_xy) - np.array(target_xy))

        gripper_str = "OPEN" if gripper > 0.5 else "CLOSED"

        lines = [
            f"Ep {ep_index} | Step {t+1}/{T}",
            f"Target: ({target_xy[0]:.3f}, {target_xy[1]:.3f})",
            f"ObjXY: ({obj_xy[0]:.3f}, {obj_xy[1]:.3f})  Z:{obj_z:.3f}",
            f"Dist: {dist_to_target:.4f}  Thr: {distance_threshold}",
            f"Gripper: {gripper_str} ({gripper:.2f})",
        ]

        # Color-code distance
        if dist_to_target < distance_threshold:
            dist_color = (0, 255, 0)  # green
        elif dist_to_target < distance_threshold * 2:
            dist_color = (0, 255, 255)  # yellow
        else:
            dist_color = (255, 255, 255)  # white

        if success and success_step is not None and t + 1 >= success_step:
            lines.append("STATUS: SUCCESS")
            status_color = (0, 255, 0)
        else:
            lines.append("STATUS: RUNNING..." if not success else "STATUS: SUCCESS")
            status_color = (255, 255, 255) if not success else (0, 255, 0)

        frame = add_multi_line_overlay(frame, lines, start_y=2,
                                       line_height=int(14 * scale / 3 + 2),
                                       font_scale=0.35 * scale / 3 + 0.1)

        # Draw a progress bar at the bottom
        bar_h = max(4, scale * 2)
        bar_y = out_H - bar_h - 2
        progress = (t + 1) / T
        bar_w = int(progress * (out_W - 8))
        bar_color = (0, 200, 0) if success else (0, 100, 200)
        cv2.rectangle(frame, (4, bar_y), (4 + bar_w, bar_y + bar_h),
                      bar_color, -1)
        cv2.rectangle(frame, (4, bar_y), (out_W - 4, bar_y + bar_h),
                      (128, 128, 128), 1)

        # Final frame: big result overlay
        if t == T - 1:
            result_text = "FAILED" if not success else f"SUCCESS (step {success_step})"
            result_color = (0, 0, 255) if not success else (0, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.6 * scale / 3 + 0.2
            (tw, th), _ = cv2.getTextSize(result_text, font, fs, 2)
            cx, cy = (out_W - tw) // 2, out_H // 2
            cv2.rectangle(frame, (cx - 6, cy - th - 6), (cx + tw + 6, cy + 10),
                          (0, 0, 0), -1)
            cv2.putText(frame, result_text, (cx, cy), font, fs,
                        result_color, 2, cv2.LINE_AA)

        annotated_frames.append(frame)

    imageio.mimsave(str(out_path), annotated_frames, fps=fps)


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
            pre_position_gripper_down,
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

        # Use a dummy seed first, will reseed after
        set_seed(args.seed)
        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print(f"  Failure Analysis Evaluation — Task B ({args.run_id})")
        print(f"  Policy: {args.policy_B}")
        print(f"  Episodes: {args.num_episodes}")
        print(f"  Seed: {args.seed}")
        print(f"{'='*60}\n")

        # =================================================================
        # Load policy config
        # =================================================================
        config_B = load_policy_config(args.policy_B)

        # =================================================================
        # Create environment
        # =================================================================
        env = make_env_with_camera(
            task_id=args.task, num_envs=1, device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width, image_height=args.image_height,
            episode_length_s=1000.0, disable_terminations=True,
        )

        # =================================================================
        # Load policy (only B needed)
        # =================================================================
        policy_B, preproc_B, postproc_B, _, n_act_B = load_diffusion_policy(
            args.policy_B, device,
            image_height=args.image_height, image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_B.eval()

        # Build tester — we need a dummy policy A, but use B for both slots
        # since we only call run_task_B
        tester = AlternatingTester(
            env=env,
            policy_A=policy_B, preprocessor_A=preproc_B, postprocessor_A=postproc_B,
            policy_B=policy_B, preprocessor_B=preproc_B, postprocessor_B=postproc_B,
            n_action_steps_A=n_act_B, n_action_steps_B=n_act_B,
            goal_xy=tuple(args.goal_xy),
            height_threshold=args.height_threshold,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=config_B["has_wrist"],
            has_wrist_B=config_B["has_wrist"],
            include_obj_pose_A=config_B["include_obj_pose"],
            include_obj_pose_B=config_B["include_obj_pose"],
            include_gripper_A=config_B["include_gripper"],
            include_gripper_B=config_B["include_gripper"],
        )

        goal_xy = np.array(args.goal_xy)
        rng = np.random.default_rng(args.seed)

        # Initial env reset + markers
        env.reset()
        pre_position_gripper_down(env)
        place_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device,
        )
        tester.place_markers = place_markers
        tester.goal_markers = goal_markers
        tester.marker_z = marker_z

        first_place_xy = tester._sample_new_place_target()
        tester.current_place_xy = first_place_xy
        update_target_markers(
            place_markers, goal_markers,
            first_place_xy, tuple(goal_xy), marker_z, env,
        )

        # =================================================================
        # Output paths
        # =================================================================
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_dir = out_dir / f"failure_videos_{args.run_id}"
        video_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / f"failure_analysis_{args.run_id}.json"
        npz_path = out_dir / f"failure_analysis_{args.run_id}.npz"

        # =================================================================
        # Run Task B episodes with full env.reset()
        # =================================================================
        results = []
        episode_details = []
        all_episodes_data = {}  # For NPZ: ep_0_images, ep_0_actions, ...

        start_time = time.time()

        for ep_idx in range(args.num_episodes):
            # ---- Hard reset for Task B ----
            env.reset()
            pre_position_gripper_down(env)
            new_place_xy = tester._sample_new_place_target()
            tester.current_place_xy = new_place_xy
            update_target_markers(
                place_markers, goal_markers,
                new_place_xy, tuple(goal_xy), marker_z, env,
            )
            # Object starts at goal position (Task B picks from goal)
            obj_pose = torch.tensor(
                [goal_xy[0], goal_xy[1], 0.022, 1.0, 0.0, 0.0, 0.0],
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            teleport_object_to_pose(env, obj_pose, name="object")
            ee_hold = get_ee_pose_w(env)
            hold_action = torch.zeros(1, env.action_space.shape[-1], device=device)
            hold_action[:, :7] = ee_hold[:, :7]
            hold_action[:, 7] = 1.0
            for _ in range(10):
                env.step(hold_action)
            tester.current_gripper_state = 1.0
            policy_B.reset()

            # Clear video_frames buffer
            tester.video_frames = []

            print(f"\n  [Task B] Ep {ep_idx + 1}/{args.num_episodes}  "
                  f"obj=goal [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}], "
                  f"target=[{new_place_xy[0]:.3f}, {new_place_xy[1]:.3f}]")

            # ---- Run task ----
            ep_data, success = tester.run_task_B()

            results.append(success)
            status = "SUCCESS" if success else "FAILED"
            rate = sum(results) / len(results) * 100
            print(f"    {status}  (running: {sum(results)}/{len(results)} = {rate:.1f}%)")

            # ---- Store episode data ----
            T = len(ep_data.get("images", []))
            detail = {
                "episode_index": ep_idx,
                "success": success,
                "success_step": ep_data.get("success_step"),
                "total_steps": T,
                "target_xy": [float(new_place_xy[0]), float(new_place_xy[1])],
                "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            }
            if "obj_pose" in ep_data and len(ep_data["obj_pose"]) > 0:
                detail["initial_obj_position"] = ep_data["obj_pose"][0][:3].tolist()
                detail["final_obj_position"] = ep_data["obj_pose"][-1][:3].tolist()
                final_xy = ep_data["obj_pose"][-1][:2]
                detail["final_dist_to_target"] = float(
                    np.linalg.norm(final_xy - np.array(new_place_xy))
                )
            episode_details.append(detail)

            # Save episode data to NPZ dict
            prefix = f"ep_{ep_idx}"
            all_episodes_data[f"{prefix}_images"] = ep_data["images"]
            all_episodes_data[f"{prefix}_ee_pose"] = ep_data["ee_pose"]
            all_episodes_data[f"{prefix}_obj_pose"] = ep_data["obj_pose"]
            all_episodes_data[f"{prefix}_action"] = ep_data["action"]
            all_episodes_data[f"{prefix}_success"] = np.array(success)
            all_episodes_data[f"{prefix}_target_xy"] = np.array(new_place_xy, dtype=np.float32)
            if ep_data.get("wrist_images") is not None:
                all_episodes_data[f"{prefix}_wrist_images"] = ep_data["wrist_images"]

            # ---- Render video for failed episodes (or all if requested) ----
            should_render = (not success) or args.render_success_videos
            if should_render:
                tag = "fail" if not success else "success"
                vid_path = video_dir / f"{tag}_ep{ep_idx:03d}.mp4"
                print(f"    Rendering video → {vid_path}")
                render_episode_video(
                    frames=list(ep_data["images"]),
                    out_path=vid_path,
                    ep_index=ep_idx,
                    success=success,
                    success_step=ep_data.get("success_step"),
                    target_xy=(float(new_place_xy[0]), float(new_place_xy[1])),
                    obj_poses=ep_data["obj_pose"],
                    ee_poses=ep_data["ee_pose"],
                    actions=ep_data["action"],
                    distance_threshold=args.distance_threshold,
                    goal_xy=tuple(args.goal_xy),
                    fps=args.video_fps,
                )

        elapsed = time.time() - start_time

        # =================================================================
        # Summary
        # =================================================================
        n_success = sum(results)
        n_fail = len(results) - n_success
        rate = n_success / len(results) if results else 0.0

        print(f"\n{'='*60}")
        print(f"  Failure Analysis Results ({args.run_id})")
        print(f"{'='*60}")
        print(f"  Total: {len(results)}  Success: {n_success}  Failed: {n_fail}")
        print(f"  Success rate: {rate:.1%}")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Failure breakdown
        fail_details = [d for d in episode_details if not d["success"]]
        if fail_details:
            dists = [d.get("final_dist_to_target", float("inf"))
                     for d in fail_details]
            print(f"\n  Failed episodes distance to target:")
            print(f"    Mean: {np.mean(dists):.4f}")
            print(f"    Min:  {np.min(dists):.4f}")
            print(f"    Max:  {np.max(dists):.4f}")
            print(f"    Median: {np.median(dists):.4f}")

            # Categorize failures
            near_miss = sum(1 for d in dists if d < 0.10)
            far_miss = sum(1 for d in dists if d >= 0.10)
            print(f"    Near-miss (<0.10m): {near_miss}")
            print(f"    Far-miss  (>=0.10m): {far_miss}")

        # Success step stats
        success_details = [d for d in episode_details if d["success"]]
        if success_details:
            steps = [d["success_step"] for d in success_details
                     if d.get("success_step")]
            if steps:
                print(f"\n  Success step statistics:")
                print(f"    Mean: {np.mean(steps):.1f}")
                print(f"    Min:  {np.min(steps)}")
                print(f"    Max:  {np.max(steps)}")

        # =================================================================
        # Save stats JSON
        # =================================================================
        stats = {
            "experiment": "failure_analysis_task_B",
            "timestamp": datetime.now().isoformat(),
            "run_id": args.run_id,
            "config": {
                "policy_B": args.policy_B,
                "num_episodes": args.num_episodes,
                "horizon": args.horizon,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": args.goal_xy,
                "seed": args.seed,
            },
            "summary": {
                "total_episodes": len(results),
                "success_count": n_success,
                "fail_count": n_fail,
                "success_rate": rate,
                "elapsed_seconds": elapsed,
            },
            "episodes": episode_details,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats saved: {stats_path}")

        # =================================================================
        # Save NPZ
        # =================================================================
        all_episodes_data["num_episodes"] = np.array(args.num_episodes)
        np.savez_compressed(str(npz_path), **all_episodes_data)
        print(f"  NPZ saved:  {npz_path}")
        print(f"  Videos:     {video_dir}/")

        n_videos = len(list(video_dir.glob("*.mp4")))
        print(f"  Total videos rendered: {n_videos}")

        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
