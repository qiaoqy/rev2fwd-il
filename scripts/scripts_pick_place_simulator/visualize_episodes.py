#!/usr/bin/env python3
"""Visualize episodes from NPZ files as MP4 videos.

Renders table + wrist camera images side-by-side with text overlay showing:
  - Episode index, timestep, total frames
  - Success / Failure status
  - EE position (x, y, z)
  - Gripper state (open / closed)
  - FSM state name (if available)

Usage:
    python scripts/scripts_pick_place_simulator/visualize_episodes.py \
        --input data/pick_place_isaac_lab_simulation/exp37/task_B_cube.npz \
        --out_dir data/pick_place_isaac_lab_simulation/exp37/logs/phase0_demo_videos \
        --episodes 0 1 2 3 4 \
        --fps 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# FSM state names matching ExpertState IntEnum
_FSM_STATE_NAMES = {
    0: "REST",
    1: "APPROACH_OBJ",
    2: "GO_ABOVE_OBJ",
    3: "CORRECT_OBJ",
    4: "ALIGN_TO_OBJ",
    5: "GO_TO_OBJ",
    6: "CLOSE",
    7: "LIFT_ALIGN",
    8: "LIFT_CORRECT",
    9: "LIFT_ABOVE",
    10: "LIFT_APPROACH",
    11: "APPROACH_PLACE",
    12: "GO_ABOVE_PLACE",
    13: "CORRECT_PLACE",
    14: "ALIGN_TO_PLACE",
    15: "GO_TO_PLACE",
    16: "LOWER_TO_RELEASE",
    17: "OPEN",
    18: "DEPART_ALIGN",
    19: "DEPART_CORRECT",
    20: "DEPART_ABOVE",
    21: "DEPART_APPROACH",
    22: "RETURN_REST",
    23: "DONE",
}


def _get_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize NPZ episodes as MP4 videos")
    parser.add_argument("--input", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode indices to visualize (default: first 5)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=0, help="0 = full episode")
    parser.add_argument("--show_wrist", action="store_true", default=True,
                        help="Show wrist camera side-by-side")
    parser.add_argument("--no_wrist", dest="show_wrist", action="store_false")
    return parser.parse_args()


def create_episode_video(
    ep: dict,
    ep_idx: int,
    out_path: str,
    fps: int = 20,
    max_frames: int = 0,
    show_wrist: bool = True,
):
    """Create MP4 video from a single episode with text overlay."""
    images = ep["images"]
    wrist = ep.get("wrist_images", None) if show_wrist else None
    T_full = len(images)
    T = T_full if max_frames <= 0 else min(T_full, max_frames)

    H, W = images.shape[1], images.shape[2]
    has_wrist = wrist is not None and len(wrist) >= T

    gap = 4
    img_w = W * 2 + gap if has_wrist else W
    bar_h = 48  # space for 3 lines of text
    total_h = H + bar_h
    total_w = img_w
    # Ensure even dimensions for H.264
    total_h += total_h % 2
    total_w += total_w % 2

    success = ep.get("success", False)
    ee_pose = ep.get("ee_pose", None)
    action = ep.get("action", None)
    fsm_state = ep.get("fsm_state", None)

    font = _get_font(12)

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Camera images
        canvas[bar_h:bar_h + H, :W] = images[t]
        if has_wrist:
            canvas[bar_h:bar_h + H, W + gap:W + gap + W] = wrist[t]

        # Status bar background
        bar_color = (0, 100, 0) if success else (120, 0, 0)
        canvas[:bar_h, :] = bar_color

        # Build text lines
        line1 = f"Ep{ep_idx}  t={t}/{T_full}"
        line1 += "  SUCCESS" if success else "  FAILURE"

        line2 = ""
        if ee_pose is not None and t < len(ee_pose):
            xyz = ee_pose[t, :3]
            line2 += f"EE=({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"

        if action is not None and t < len(action) and action.shape[-1] > 7:
            grip = action[t, 7]
            grip_str = "OPEN" if grip > 0 else "CLOSED"
            line2 += f"  Gripper: {grip_str}"

        line3 = ""
        if fsm_state is not None and t < len(fsm_state):
            state_id = int(fsm_state[t])
            state_name = _FSM_STATE_NAMES.get(state_id, f"STATE_{state_id}")
            line3 = f"FSM: {state_name} ({state_id})"

        # Render text onto bar using PIL
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        draw.text((4, 2), line1, fill=(255, 255, 255), font=font)
        draw.text((4, 16), line2, fill=(220, 220, 220), font=font)
        if line3:
            draw.text((4, 30), line3, fill=(180, 255, 180), font=font)
        canvas = np.array(pil_img)

        writer.append_data(canvas)

    writer.close()
    print(f"  Saved: {out_path} ({T} frames, {T/fps:.1f}s)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading episodes from {args.input}...")
    with np.load(args.input, allow_pickle=True) as data:
        episodes = list(data["episodes"])

    n_total = len(episodes)
    n_success = sum(1 for ep in episodes if ep.get("success", False))
    print(f"  {n_total} episodes ({n_success} success, {n_total - n_success} failure)")

    # Select episodes
    if args.episodes is not None:
        indices = args.episodes
    else:
        indices = list(range(min(5, n_total)))

    print(f"Rendering {len(indices)} episodes...")
    for ep_idx in indices:
        if ep_idx >= n_total:
            print(f"  Skipping ep{ep_idx} (only {n_total} episodes)")
            continue

        ep = episodes[ep_idx]
        success_tag = "success" if ep.get("success", False) else "failure"
        T = len(ep["images"])
        out_path = str(out_dir / f"ep{ep_idx}_{success_tag}_T{T}.mp4")

        create_episode_video(
            ep, ep_idx, out_path,
            fps=args.fps, max_frames=args.max_frames,
            show_wrist=args.show_wrist,
        )

    print(f"\nDone! Videos saved to {out_dir}")


if __name__ == "__main__":
    main()
