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
# Action chunk collector
# =========================================================================
class ActionChunkCollector:
    """Collects predicted action chunks during evaluation.

    Implements the same ``add_frame`` interface expected by
    ``AlternatingTester._run_task`` so it can be plugged in as
    ``action_chunk_visualizer``.
    """

    def __init__(self):
        self.chunks: list[dict] = []  # one entry per inference step
        self._inference_step = 0

    # signature must match ActionChunkVisualizer.add_frame
    def add_frame(
        self,
        ee_pose_raw,
        ee_pose_norm,
        action_chunk_norm,
        action_chunk_raw=None,
        gt_chunk_norm=None,
        gt_chunk_raw=None,
        table_image=None,
        wrist_image=None,
    ):
        self.chunks.append({
            "inference_step": self._inference_step,
            "ee_pose_raw": np.asarray(ee_pose_raw).copy(),
            "action_chunk_norm": np.asarray(action_chunk_norm).copy(),
            "action_chunk_raw": (
                np.asarray(action_chunk_raw).copy()
                if action_chunk_raw is not None
                else np.asarray(action_chunk_norm).copy()
            ),
        })
        self._inference_step += 1

    def reset(self):
        self.chunks.clear()
        self._inference_step = 0


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


def _draw_chunk_plot(action_chunk_raw: np.ndarray,
                     step_in_chunk: int,
                     ee_xyz: np.ndarray,
                     plot_h: int = 120,
                     plot_w: int = 160) -> np.ndarray:
    """Draw a small XYZ plot of the current action chunk.

    Returns an (plot_h, plot_w, 3) uint8 image.
    """
    img = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    chunk_len = len(action_chunk_raw)
    if chunk_len == 0:
        return img

    margin_l, margin_r, margin_t, margin_b = 30, 5, 12, 14
    gw = plot_w - margin_l - margin_r
    gh = plot_h - margin_t - margin_b

    # X, Y, Z channels
    colors = [(0, 0, 255), (0, 200, 0), (255, 100, 0)]  # R, G, B for X, Y, Z
    labels = ["X", "Y", "Z"]

    all_vals = action_chunk_raw.flatten()
    ee_vals = ee_xyz[:3]
    combined = np.concatenate([all_vals, ee_vals])
    vmin, vmax = combined.min() - 0.01, combined.max() + 0.01
    if vmax - vmin < 1e-6:
        vmin -= 0.05
        vmax += 0.05

    def to_px(step, val):
        x = margin_l + int(step / max(chunk_len - 1, 1) * gw)
        y = margin_t + int((1.0 - (val - vmin) / (vmax - vmin)) * gh)
        return (x, y)

    # Grid
    cv2.rectangle(img, (margin_l, margin_t),
                  (margin_l + gw, margin_t + gh), (50, 50, 50), 1)
    # Y-axis labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"{vmax:.2f}", (1, margin_t + 8), font, 0.25, (160, 160, 160), 1)
    cv2.putText(img, f"{vmin:.2f}", (1, margin_t + gh), font, 0.25, (160, 160, 160), 1)

    # Plot each axis
    for dim in range(min(3, action_chunk_raw.shape[1])):
        pts = [to_px(s, action_chunk_raw[s, dim]) for s in range(chunk_len)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], colors[dim], 1, cv2.LINE_AA)
        # Current EE pose marker (horizontal dashed line)
        ee_y = to_px(0, ee_xyz[dim])[1]
        for xx in range(margin_l, margin_l + gw, 6):
            cv2.line(img, (xx, ee_y), (min(xx + 3, margin_l + gw), ee_y),
                     colors[dim], 1)

    # Vertical line at current step
    cur_x = to_px(step_in_chunk, 0)[0]
    cv2.line(img, (cur_x, margin_t), (cur_x, margin_t + gh),
             (255, 255, 0), 1)

    # Legend
    for dim, (lbl, clr) in enumerate(zip(labels, colors)):
        lx = margin_l + 4 + dim * 35
        cv2.putText(img, lbl, (lx, margin_t + gh + 12), font, 0.3, clr, 1)
    cv2.putText(img, "-- EE", (margin_l + 110, margin_t + gh + 12),
                font, 0.25, (160, 160, 160), 1)

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
    n_action_steps: int = 16,
    action_chunks: list[dict] | None = None,
    premove_len: int = 0,
) -> None:
    """Render annotated MP4 for a single episode.

    When *action_chunks* is provided (list of dicts from
    ``ActionChunkCollector``), an extra debug panel is drawn showing
    the predicted action-chunk trajectory and per-step action values.

    *premove_len*: number of leading frames that belong to the pre-move
    phase (before policy inference starts).  They are annotated with a
    cyan "PRE-MOVE" label.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(frames)
    H, W = frames[0].shape[:2]
    scale = max(1, 384 // W)
    out_H, out_W = H * scale, W * scale
    has_wrist = (wrist_frames is not None and len(wrist_frames) > 0)

    # Pre-build a mapping: env step -> (chunk_index, step_in_chunk, chunk_data)
    # Action chunks only cover the policy phase (after premove_len)
    chunk_map: dict[int, tuple[int, int, dict]] = {}  # step -> info
    if action_chunks:
        for ci, cdata in enumerate(action_chunks):
            start_step = premove_len + ci * n_action_steps
            chunk_len = len(cdata["action_chunk_raw"])
            for s in range(chunk_len):
                env_step = start_step + s
                if env_step < T:
                    chunk_map[env_step] = (ci, s, cdata)

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

        ee_xyz = ee_poses[t, :3] if t < len(ee_poses) else np.zeros(3)
        act_xyz = actions[t, :3] if t < len(actions) else np.zeros(3)

        lines = [
            f"Ep {ep_index} | Step {t+1}/{T}  Task {task_type}",
            f"ObjXY: ({obj_xy[0]:.3f}, {obj_xy[1]:.3f})  Z:{obj_z:.3f}",
        ]

        # Pre-move phase annotation
        is_premove = (t < premove_len)
        if is_premove:
            lines.insert(0, f">>> PRE-MOVE  {t+1}/{premove_len} <<<")

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

        # Action chunk debug info (skip during pre-move)
        chunk_info = chunk_map.get(t)
        if is_premove:
            chunk_info = None  # no chunk during pre-move
        elif chunk_info is not None:
            ci, si, cdata = chunk_info
            chunk_len = len(cdata["action_chunk_raw"])
            is_boundary = (si == 0)
            marker = " <<INFER>>" if is_boundary else ""
            lines.append(f"Chunk {ci} Step {si+1}/{chunk_len}{marker}")
        else:
            # Fallback: compute from n_action_steps
            pol_t = t - premove_len
            ci = pol_t // n_action_steps
            si = pol_t % n_action_steps
            is_boundary = (si == 0)
            marker = " <<INFER>>" if is_boundary else ""
            lines.append(f"Chunk {ci} Step {si+1}/{n_action_steps}{marker}")

        # Show action vs EE pose
        lines.append(f"Act: ({act_xyz[0]:.3f},{act_xyz[1]:.3f},{act_xyz[2]:.3f})")
        lines.append(f"EE:  ({ee_xyz[0]:.3f},{ee_xyz[1]:.3f},{ee_xyz[2]:.3f})")
        delta = np.linalg.norm(act_xyz - ee_xyz)
        lines.append(f"|Act-EE|: {delta:.4f}")

        if success and success_step is not None and t + 1 >= success_step:
            lines.append("STATUS: SUCCESS")
        else:
            lines.append("STATUS: RUNNING..." if not success else "STATUS: SUCCESS")

        frame = add_multi_line_overlay(frame, lines, start_y=2,
                                       line_height=int(14 * scale / 3 + 2),
                                       font_scale=0.35 * scale / 3 + 0.1)

        # Draw chunk XYZ mini-plot in bottom-right corner
        if chunk_info is not None:
            ci, si, cdata = chunk_info
            plot_h, plot_w = 120, 160
            plot_img = _draw_chunk_plot(
                cdata["action_chunk_raw"], si, ee_xyz,
                plot_h=plot_h, plot_w=plot_w,
            )
            # Place in bottom-right of this camera panel
            y0 = max(0, out_H - plot_h - 18)
            x0 = max(0, out_W - plot_w - 4)
            # Semi-transparent overlay
            roi = frame[y0:y0 + plot_h, x0:x0 + plot_w]
            blended = cv2.addWeighted(roi, 0.3, plot_img[:roi.shape[0], :roi.shape[1]], 0.7, 0)
            frame[y0:y0 + plot_h, x0:x0 + plot_w] = blended

        # Chunk boundary flash (yellow border)
        if chunk_info is not None and chunk_info[1] == 0:
            cv2.rectangle(frame, (0, 0), (out_W - 1, out_H - 1),
                          (0, 255, 255), 2)

        # Cyan border for pre-move frames
        if is_premove:
            cv2.rectangle(frame, (0, 0), (out_W - 1, out_H - 1),
                          (255, 255, 0), 2)  # cyan in BGR

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
    parser.add_argument("--pre_move_to_object", action="store_true",
                        help="Move arm above the object before policy inference "
                             "(debug: gives policy a better starting pose).")
    parser.add_argument("--pre_move_height", type=float, default=0.15,
                        help="Height above object for pre-move (default 0.15m).")
    parser.add_argument("--pre_move_steps", type=int, default=80,
                        help="Sim steps for the pre-move phase.")

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

            # ----- Pre-move: move arm above the object -----
            premove_images = []
            premove_wrist = []
            premove_ee = []
            premove_obj = []
            premove_act = []

            if args.pre_move_to_object:
                from rev2fwd_il.sim.scene_api import get_object_pose_w
                GRIPPER_DOWN_QUAT = torch.tensor(
                    [0.0, 1.0, 0.0, 0.0], device=device)

                obj_pos = get_object_pose_w(env)[0].cpu().numpy()
                obj_x, obj_y, obj_z = float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])

                def _make_action(x, y, z, grip):
                    a = torch.zeros(1, env.action_space.shape[-1], device=device)
                    a[0, 0], a[0, 1], a[0, 2] = x, y, z
                    a[0, 3:7] = GRIPPER_DOWN_QUAT
                    a[0, 7] = grip
                    return a

                def _record_and_step(action_t, n_steps):
                    for _ in range(n_steps):
                        tbl, wrs, ee_p, ob_p, gs = tester._get_observation()
                        premove_images.append(tbl)
                        if wrs is not None:
                            premove_wrist.append(wrs)
                        premove_ee.append(ee_p)
                        premove_obj.append(ob_p)
                        premove_act.append(action_t[0].cpu().numpy())
                        env.step(action_t)

                # Phase 1: Approach — move above the object
                hover_z = obj_z + args.pre_move_height
                approach_act = _make_action(obj_x, obj_y, hover_z, 1.0)
                print(f"    [PRE-MOVE] Phase 1: Approach above "
                      f"({obj_x:.3f}, {obj_y:.3f}, {hover_z:.3f})")
                _record_and_step(approach_act, 60)

                # Phase 2: Lower to grasp height (obj_z + 0.02m)
                grasp_z = obj_z + 0.02
                lower_act = _make_action(obj_x, obj_y, grasp_z, 1.0)
                print(f"    [PRE-MOVE] Phase 2: Lower to grasp height z={grasp_z:.3f}")
                _record_and_step(lower_act, 40)

                # Phase 3: Close gripper
                close_act = _make_action(obj_x, obj_y, grasp_z, -1.0)
                print(f"    [PRE-MOVE] Phase 3: Close gripper")
                _record_and_step(close_act, 20)

                # Phase 4: Lift with object
                lift_z = obj_z + args.pre_move_height
                lift_act = _make_action(obj_x, obj_y, lift_z, -1.0)
                print(f"    [PRE-MOVE] Phase 4: Lift to z={lift_z:.3f}")
                _record_and_step(lift_act, 40)

                # Phase 5: Settle at lifted position
                _record_and_step(lift_act, 15)

                ee_now = get_ee_pose_w(env)[0].cpu().numpy()
                ob_now = get_object_pose_w(env)[0].cpu().numpy()
                print(f"    [PRE-MOVE] Done. EE=({ee_now[0]:.3f}, {ee_now[1]:.3f}, "
                      f"{ee_now[2]:.3f})  Obj=({ob_now[0]:.3f}, {ob_now[1]:.3f}, "
                      f"{ob_now[2]:.3f})")
                tester.current_gripper_state = -1.0

            premove_len = len(premove_images)

            # Reset policy action queue (fresh start after pre-move)
            policy.reset()

            # Set up action chunk collector for this episode
            chunk_collector = ActionChunkCollector()
            if task_label == "A":
                tester.action_chunk_visualizer_A = chunk_collector
            else:
                tester.action_chunk_visualizer_B = chunk_collector

            # Run task
            if task_label == "A":
                ep_data, success = tester.run_task_A()
            else:
                ep_data, success = tester.run_task_B()

            # Retrieve collected action chunks
            ep_action_chunks = list(chunk_collector.chunks)

            # Merge pre-move data into episode data
            if premove_len > 0:
                ep_data["images"] = np.concatenate(
                    [np.array(premove_images, dtype=np.uint8),
                     ep_data["images"]], axis=0)
                ep_data["ee_pose"] = np.concatenate(
                    [np.array(premove_ee, dtype=np.float32),
                     ep_data["ee_pose"]], axis=0)
                ep_data["obj_pose"] = np.concatenate(
                    [np.array(premove_obj, dtype=np.float32),
                     ep_data["obj_pose"]], axis=0)
                ep_data["action"] = np.concatenate(
                    [np.array(premove_act, dtype=np.float32),
                     ep_data["action"]], axis=0)
                if premove_wrist and ep_data.get("wrist_images") is not None:
                    ep_data["wrist_images"] = np.concatenate(
                        [np.array(premove_wrist, dtype=np.uint8),
                         ep_data["wrist_images"]], axis=0)
                if ep_data.get("success_step") is not None:
                    ep_data["success_step"] += premove_len

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

            # Save action chunks for this episode
            if ep_action_chunks:
                # Stack chunk raw arrays: (num_chunks, chunk_len, 3)
                raw_stack = np.array(
                    [c["action_chunk_raw"] for c in ep_action_chunks],
                    dtype=np.float32,
                )
                all_episodes_data[f"{prefix}_action_chunks_raw"] = raw_stack

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
                    n_action_steps=n_act,
                    action_chunks=ep_action_chunks if ep_action_chunks else None,
                    premove_len=premove_len,
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
