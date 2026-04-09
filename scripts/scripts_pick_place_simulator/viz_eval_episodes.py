#!/usr/bin/env python3
"""Visualize eval rollout episodes from NPZ files.

Creates per-episode MP4 videos with:
- Table camera view
- Wrist camera view (side-by-side)
- EE + cube XY trajectory overlay on table image
- Text overlay: step/horizon, cube positions, gripper state

Usage:
    python scripts/scripts_pick_place_simulator/viz_eval_episodes.py \
        --npz_A data/pick_place_isaac_lab_simulation/exp38/eval_cyclic_A.npz \
        --npz_B data/pick_place_isaac_lab_simulation/exp38/eval_cyclic_B.npz \
        --out_dir data/pick_place_isaac_lab_simulation/exp38/eval_debug_videos \
        --fps 30
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_A", type=str, default=None)
    p.add_argument("--npz_B", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_episodes", type=int, default=10)
    p.add_argument("--speedup", type=int, default=3,
                   help="Skip every N frames for faster preview")
    return p.parse_args()


# ── coord transforms ─────────────────────────────────────────────────

# Approximate mapping from world XY → pixel in 128×128 table camera image.
# Table cam: eye=(1.6, 0.7, 0.8), target≈(0.4, 0.0, 0.2)
# FOV covers roughly x∈[0.2,0.8], y∈[-0.4,0.4] on the table.
# These are rough — enough for trajectory overlay debugging.
TABLE_X_RANGE = (0.15, 0.85)  # world X
TABLE_Y_RANGE = (-0.45, 0.45)  # world Y
IMG_W, IMG_H = 128, 128


def world_to_pixel(x, y, img_w=IMG_W, img_h=IMG_H):
    """Map world XY → pixel (col, row) on the table camera image."""
    # X → vertical (row): higher X = farther from camera = top of image
    u = 1.0 - (x - TABLE_X_RANGE[0]) / (TABLE_X_RANGE[1] - TABLE_X_RANGE[0])
    # Y → horizontal (col): more negative Y = right of image
    v = 1.0 - (y - TABLE_Y_RANGE[0]) / (TABLE_Y_RANGE[1] - TABLE_Y_RANGE[0])
    col = int(np.clip(v * img_w, 0, img_w - 1))
    row = int(np.clip(u * img_h, 0, img_h - 1))
    return col, row


# ── colors ────────────────────────────────────────────────────────────
COLOR_EE = (0, 255, 0)       # green - end-effector
COLOR_LARGE = (0, 215, 255)  # yellow (BGR)
COLOR_MED = (255, 100, 50)   # blue (BGR)
COLOR_SMALL = (200, 50, 200) # purple (BGR)
COLOR_ACTION = (0, 0, 255)   # red
COLOR_GOAL = (0, 255, 128)   # green marker region
COLOR_RED_REGION = (0, 0, 200)  # red region

GOAL_XY = np.array([0.5, -0.2])
RED_CENTER = np.array([0.5, 0.2])
RED_SIZE = np.array([0.30, 0.30])


def draw_regions(canvas, scale):
    """Draw goal (green) and red region markers on the canvas."""
    h, w = canvas.shape[:2]

    # Green goal marker (small circle)
    gc, gr = world_to_pixel(GOAL_XY[0], GOAL_XY[1], w, h)
    cv2.circle(canvas, (gc, gr), int(4 * scale), COLOR_GOAL, 1)
    cv2.putText(canvas, "G", (gc + int(5*scale), gr - int(3*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, COLOR_GOAL, 1)

    # Red region rectangle
    x1, y1 = RED_CENTER[0] - RED_SIZE[0] / 2, RED_CENTER[1] - RED_SIZE[1] / 2
    x2, y2 = RED_CENTER[0] + RED_SIZE[0] / 2, RED_CENTER[1] + RED_SIZE[1] / 2
    c1, r1 = world_to_pixel(x1, y1, w, h)
    c2, r2 = world_to_pixel(x2, y2, w, h)
    cv2.rectangle(canvas, (min(c1, c2), min(r1, r2)),
                  (max(c1, c2), max(r1, r2)), COLOR_RED_REGION, 1)


def render_episode(ep, task_label: str, out_path: str, fps: int,
                   speedup: int):
    """Render a single episode to an MP4 video."""
    images = ep["images"]           # (T, H, W, 3)
    wrist = ep.get("wrist_images")  # (T, H, W, 3) or None
    ee_poses = ep["ee_pose"]        # (T, 7)
    actions = ep["action"]          # (T, 8)
    obj_poses = ep["obj_pose"]      # (T, 7)  cube_small

    cube_l = ep.get("cube_large_pose")   # (T, 7); optional
    cube_m = ep.get("cube_medium_pose")  # (T, 7)
    cube_s = ep.get("cube_small_pose")   # (T, 7)

    T = len(images)
    success = bool(ep.get("success", False))
    success_step = ep.get("success_step", None)

    # Upscale for readability
    SCALE = 3
    h, w = images.shape[1] * SCALE, images.shape[2] * SCALE
    has_wrist = wrist is not None and len(wrist) > 0

    # Canvas: table | wrist | info panel
    panel_w = int(w * 0.6)
    total_w = w + (w if has_wrist else 0) + panel_w
    total_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (total_w, total_h))

    # Collect all EE xy for trail
    ee_trail = []
    cube_trails = {"large": [], "medium": [], "small": []}

    indices = list(range(0, T, speedup))
    if indices[-1] != T - 1:
        indices.append(T - 1)

    for frame_idx, t in enumerate(indices):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Table camera
        table_img = cv2.resize(images[t], (w, h), interpolation=cv2.INTER_NEAREST)
        table_img = cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR)

        # Draw regions
        draw_regions(table_img, SCALE)

        # EE trail
        ee_xy = ee_poses[t, :2]
        ee_trail.append(ee_xy)
        for i in range(1, len(ee_trail)):
            p1 = world_to_pixel(ee_trail[i-1][0], ee_trail[i-1][1], w, h)
            p2 = world_to_pixel(ee_trail[i][0], ee_trail[i][1], w, h)
            cv2.line(table_img, p1, p2, COLOR_EE, 1)
        # Current EE
        ec, er = world_to_pixel(ee_xy[0], ee_xy[1], w, h)
        cv2.circle(table_img, (ec, er), int(3 * SCALE), COLOR_EE, -1)

        # Cube markers
        if cube_l is not None:
            cx, cy = cube_l[t, 0], cube_l[t, 1]
            cube_trails["large"].append((cx, cy))
            pc, pr = world_to_pixel(cx, cy, w, h)
            cv2.circle(table_img, (pc, pr), int(2 * SCALE), COLOR_LARGE, -1)

        if cube_m is not None:
            cx, cy = cube_m[t, 0], cube_m[t, 1]
            cube_trails["medium"].append((cx, cy))
            pc, pr = world_to_pixel(cx, cy, w, h)
            cv2.circle(table_img, (pc, pr), int(2 * SCALE), COLOR_MED, -1)

        if cube_s is not None:
            cx, cy = cube_s[t, 0], cube_s[t, 1]
            cube_trails["small"].append((cx, cy))
            pc, pr = world_to_pixel(cx, cy, w, h)
            cv2.circle(table_img, (pc, pr), int(2 * SCALE), COLOR_SMALL, -1)

        # Action arrow
        act_xy = actions[t, :2]
        a_c, a_r = world_to_pixel(act_xy[0], act_xy[1], w, h)
        cv2.arrowedLine(table_img, (ec, er), (a_c, a_r), COLOR_ACTION, 1,
                        tipLength=0.3)

        canvas[:h, :w] = table_img

        # Wrist camera
        col_offset = w
        if has_wrist:
            wrist_img = cv2.resize(wrist[t], (w, h),
                                   interpolation=cv2.INTER_NEAREST)
            wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            canvas[:h, col_offset:col_offset + w] = wrist_img
            col_offset += w

        # Info panel
        panel = canvas[:h, col_offset:col_offset + panel_w]
        y_pos = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.4
        c_white = (255, 255, 255)
        c_red = (0, 0, 255)
        c_green = (0, 255, 0)

        def put(text, color=c_white):
            nonlocal y_pos
            cv2.putText(panel, text, (5, y_pos), font, fs, color, 1)
            y_pos += 18

        put(f"{task_label}", (255, 200, 100))
        put(f"Step: {t}/{T}  ({t*100//T}%)")
        outcome = "SUCCESS" if success else "FAILED"
        put(f"Result: {outcome}", c_green if success else c_red)
        if success_step:
            put(f"Success at step: {success_step}")
        put("")
        put(f"EE: ({ee_poses[t,0]:.3f}, {ee_poses[t,1]:.3f}, {ee_poses[t,2]:.3f})")
        put(f"Gripper: {actions[t,7]:.2f}")
        put(f"Act: ({actions[t,0]:.3f}, {actions[t,1]:.3f}, {actions[t,2]:.3f})")
        put("")

        if cube_l is not None:
            lx, ly, lz = cube_l[t, :3]
            put(f"Large:  ({lx:.3f}, {ly:.3f}, {lz:.3f})", COLOR_LARGE)
        if cube_m is not None:
            mx, my, mz = cube_m[t, :3]
            put(f"Medium: ({mx:.3f}, {my:.3f}, {mz:.3f})", COLOR_MED)
        if cube_s is not None:
            sx, sy, sz = cube_s[t, :3]
            put(f"Small:  ({sx:.3f}, {sy:.3f}, {sz:.3f})", COLOR_SMALL)

        put("")
        # Check cube containment
        if cube_l is not None and cube_m is not None and cube_s is not None:
            # Task A check: cubes near goal?
            for name, arr, color in [
                ("L", cube_l, COLOR_LARGE),
                ("M", cube_m, COLOR_MED),
                ("S", cube_s, COLOR_SMALL),
            ]:
                d_goal = np.linalg.norm(arr[t, :2] - GOAL_XY)
                in_red = (abs(arr[t, 0] - RED_CENTER[0]) <= RED_SIZE[0] / 2 and
                          abs(arr[t, 1] - RED_CENTER[1]) <= RED_SIZE[1] / 2)
                put(f"{name}: d_goal={d_goal:.3f} in_red={in_red}", color)

        writer.write(canvas)

    writer.release()
    print(f"  Saved: {out_path} ({len(indices)} frames, {T} total steps)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, npz_path in [("Task_A", args.npz_A), ("Task_B", args.npz_B)]:
        if npz_path is None:
            continue
        print(f"\nLoading {label}: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        episodes = data["episodes"]
        print(f"  {len(episodes)} episodes loaded")

        for i, ep in enumerate(episodes[:args.max_episodes]):
            ep_dict = ep if isinstance(ep, dict) else ep.item()
            success = ep_dict.get("success", False)
            tag = "succ" if success else "fail"
            out_path = out_dir / f"{label}_ep{i}_{tag}.mp4"
            print(f"  Rendering {label} episode {i} ({tag})...")
            render_episode(ep_dict, f"{label} Ep{i} [{tag.upper()}]",
                           str(out_path), args.fps, args.speedup)

    print(f"\nAll videos saved to: {out_dir}")


if __name__ == "__main__":
    main()
