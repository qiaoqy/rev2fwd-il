#!/usr/bin/env python3
"""Script 1: Visualize Inovo (RoboKit) Data.

Creates visualization videos showing:
  - Side-by-side primary + gripper camera RGB
  - XYZ trajectory curves for observation and action velocity
  - Gripper state and force/torque overlays

=============================================================================
USAGE
=============================================================================
# Visualize first 5 episodes
python scripts/scripts_task_inovo/1_visualize_inovo_data.py \
    --data_dir data/inovo_data/0209_tower_boby \
    --out_dir data/inovo_data/viz_videos \
    --num_episodes 5 --fps 30

# Visualize a single episode by name
python scripts/scripts_task_inovo/1_visualize_inovo_data.py \
    --data_dir data/inovo_data/0209_tower_boby \
    --episode 2026_02_09-17_54_42 \
    --out_dir data/inovo_data/viz_videos

# Visualize reversed data
python scripts/scripts_task_inovo/1_visualize_inovo_data.py \
    --data_dir data/inovo_data/tower_boby_A \
    --out_dir data/inovo_data/viz_videos_reversed \
    --num_episodes 5

# Skip trajectory curves (faster)
python scripts/scripts_task_inovo/1_visualize_inovo_data.py \
    --data_dir data/inovo_data/0209_tower_boby \
    --out_dir data/inovo_data/viz_videos \
    --no_curves --num_episodes 3
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# =============================================================================
# RoboKit NPZ Decoder (shared with script 0)
# =============================================================================

def load_robokit_frame(path: str | Path) -> dict:
    """Load and decode a single RoboKit NPZ frame."""
    f = np.load(str(path), allow_pickle=True)

    primary_rgb = cv2.imdecode(
        np.frombuffer(f["primary_rgb"].item(), np.uint8), cv2.IMREAD_COLOR
    )
    if primary_rgb is not None:
        primary_rgb = cv2.cvtColor(primary_rgb, cv2.COLOR_BGR2RGB)

    gripper_rgb = cv2.imdecode(
        np.frombuffer(f["gripper_rgb"].item(), np.uint8), cv2.IMREAD_COLOR
    )
    if gripper_rgb is not None:
        gripper_rgb = cv2.cvtColor(gripper_rgb, cv2.COLOR_BGR2RGB)

    robot_obs = pickle.loads(f["robot_obs"].item())
    actions = pickle.loads(f["actions"].item())
    force_torque = pickle.loads(f["force_torque"].item())

    return {
        "primary_rgb": primary_rgb,
        "gripper_rgb": gripper_rgb,
        "robot_obs": np.asarray(robot_obs, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "force_torque": np.asarray(force_torque, dtype=np.float64),
    }


def discover_episodes(data_dir: Path) -> list[dict]:
    """Discover episode directories."""
    episodes = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name in ("extracted", "__pycache__"):
            continue
        npz_files = sorted(entry.glob("*.npz"))
        if len(npz_files) == 0:
            continue
        episodes.append({
            "name": entry.name,
            "path": entry,
            "npz_files": npz_files,
        })
    return episodes


# =============================================================================
# Load Full Episode Data
# =============================================================================

def load_episode_data(npz_files: list[Path]) -> dict:
    """Load all frames of an episode into arrays.

    Returns:
        Dictionary with:
        - primary_images: list of (H, W, 3) uint8
        - gripper_images: list of (H, W, 3) uint8
        - tcp_pos: (T, 3) float64
        - tcp_ori: (T, 3) float64
        - gripper_width: (T,) float64
        - gripper_action: (T,) float64
        - actions_xyz: (T, 3) float64
        - actions_rpy: (T, 3) float64
        - actions_gripper: (T,) float64
        - force_torque: (T, 6) float64
    """
    primary_images = []
    gripper_images = []
    tcp_pos_list = []
    tcp_ori_list = []
    gripper_width_list = []
    gripper_action_list = []
    actions_xyz_list = []
    actions_rpy_list = []
    actions_gripper_list = []
    force_torque_list = []

    for npz_path in tqdm(npz_files, desc="Loading frames", leave=False):
        frame = load_robokit_frame(npz_path)
        primary_images.append(frame["primary_rgb"])
        gripper_images.append(frame["gripper_rgb"])
        tcp_pos_list.append(frame["robot_obs"][:3])
        tcp_ori_list.append(frame["robot_obs"][3:6])
        gripper_width_list.append(frame["robot_obs"][6])
        gripper_action_list.append(frame["robot_obs"][13])
        actions_xyz_list.append(frame["actions"][:3])
        actions_rpy_list.append(frame["actions"][3:6])
        actions_gripper_list.append(frame["actions"][6])
        force_torque_list.append(frame["force_torque"])

    return {
        "primary_images": primary_images,
        "gripper_images": gripper_images,
        "tcp_pos": np.array(tcp_pos_list),
        "tcp_ori": np.array(tcp_ori_list),
        "gripper_width": np.array(gripper_width_list),
        "gripper_action": np.array(gripper_action_list),
        "actions_xyz": np.array(actions_xyz_list),
        "actions_rpy": np.array(actions_rpy_list),
        "actions_gripper": np.array(actions_gripper_list),
        "force_torque": np.array(force_torque_list),
    }


# =============================================================================
# Visualization Helpers
# =============================================================================

def resize_image(img: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)


def add_text_overlay(img: np.ndarray, text: str, position: tuple = (10, 30),
                     font_scale: float = 0.6, color: tuple = (255, 255, 255),
                     bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """Add text with background rectangle."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def create_curve_frame(
    tcp_pos: np.ndarray,       # (T, 3)
    tcp_ori: np.ndarray,       # (T, 3)
    actions_xyz: np.ndarray,   # (T, 3)
    actions_rpy: np.ndarray,   # (T, 3)
    gripper_width: np.ndarray, # (T,)
    gripper_action: np.ndarray, # (T,)
    force_torque: np.ndarray,  # (T, 6)
    current_t: int,
    target_width: int = 960,
    target_height: int = 480,
) -> np.ndarray:
    """Create matplotlib figure with trajectory curves.

    Layout:
      Row 1: TCP Position (XYZ)    |  Action Velocity (XYZ)
      Row 2: Gripper               |  Force/Torque
    """
    fig, axes = plt.subplots(2, 2, figsize=(target_width / 100, target_height / 100), dpi=100)
    T = len(tcp_pos)
    t_axis = np.arange(T)

    # --- TCP Position ---
    ax = axes[0, 0]
    for i, (label, color) in enumerate(zip(["x", "y", "z"], ["r", "g", "b"])):
        ax.plot(t_axis, tcp_pos[:, i], color=color, label=label, linewidth=0.8)
    ax.axvline(current_t, color="black", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title("TCP Position (m)", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # --- Action Velocity XYZ ---
    ax = axes[0, 1]
    for i, (label, color) in enumerate(zip(["vx", "vy", "vz"], ["r", "g", "b"])):
        ax.plot(t_axis, actions_xyz[:, i], color=color, label=label, linewidth=0.8)
    ax.axvline(current_t, color="black", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title("Action Velocity (m/s)", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # --- Gripper ---
    ax = axes[1, 0]
    ax.plot(t_axis, gripper_width, color="purple", label="width", linewidth=0.8)
    ax.plot(t_axis, gripper_action, color="orange", label="action", linewidth=0.8, linestyle="--")
    ax.axvline(current_t, color="black", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title("Gripper State", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    # --- Force/Torque ---
    ax = axes[1, 1]
    ft_labels = ["fx", "fy", "fz", "mx", "my", "mz"]
    ft_colors = ["r", "g", "b", "darkred", "darkgreen", "darkblue"]
    for i in range(6):
        ax.plot(t_axis, force_torque[:, i], color=ft_colors[i], label=ft_labels[i],
                linewidth=0.5, alpha=0.7)
    ax.axvline(current_t, color="black", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title("Force/Torque", fontsize=8)
    ax.legend(fontsize=5, loc="upper right", ncol=2)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.5)

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    curve_img = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
    plt.close(fig)

    return curve_img


# =============================================================================
# Video Creation
# =============================================================================

def create_visualization_video(
    episode_name: str,
    npz_files: list[Path],
    output_path: Path,
    fps: int = 30,
    include_curves: bool = True,
    target_height: int = 360,
) -> None:
    """Create visualization video for one episode.

    Args:
        episode_name: Episode directory name.
        npz_files: Sorted list of .npz file paths.
        output_path: Output .mp4 path.
        fps: Video frame rate.
        include_curves: Whether to include trajectory curve subplot.
        target_height: Target height for camera images.
    """
    print(f"Loading episode {episode_name} ({len(npz_files)} frames) ...")
    ep_data = load_episode_data(npz_files)
    T = len(ep_data["primary_images"])

    print(f"Rendering video to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    for t in tqdm(range(T), desc="Composing frames"):
        # Resize camera images
        primary = ep_data["primary_images"][t]
        gripper = ep_data["gripper_images"][t]

        if primary is None:
            primary = np.zeros((target_height, int(target_height * 848 / 480), 3), dtype=np.uint8)
        else:
            primary = resize_image(primary, target_height)

        if gripper is None:
            gripper = np.zeros_like(primary)
        else:
            gripper = resize_image(gripper, target_height)

        # Add overlays
        tcp = ep_data["tcp_pos"][t]
        gw = ep_data["gripper_width"][t]
        ga = ep_data["gripper_action"][t]
        primary = add_text_overlay(primary, f"Primary | t={t}/{T}", (10, 20), font_scale=0.5)
        primary = add_text_overlay(
            primary,
            f"TCP: ({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f})",
            (10, 42), font_scale=0.45,
        )
        primary = add_text_overlay(
            primary,
            f"Gripper: w={gw:.2f} act={int(ga)}",
            (10, 62), font_scale=0.45,
        )
        gripper = add_text_overlay(gripper, "Gripper Cam", (10, 20), font_scale=0.5)

        # Match widths
        if primary.shape[1] != gripper.shape[1]:
            target_w = min(primary.shape[1], gripper.shape[1])
            primary = cv2.resize(primary, (target_w, target_height))
            gripper = cv2.resize(gripper, (target_w, target_height))

        # Compose: cameras side by side
        cam_row = np.concatenate([primary, gripper], axis=1)

        if include_curves:
            curve_img = create_curve_frame(
                ep_data["tcp_pos"],
                ep_data["tcp_ori"],
                ep_data["actions_xyz"],
                ep_data["actions_rpy"],
                ep_data["gripper_width"],
                ep_data["gripper_action"],
                ep_data["force_torque"],
                current_t=t,
                target_width=cam_row.shape[1],
                target_height=target_height,
            )
            # Resize curve image to match cam_row width
            if curve_img.shape[1] != cam_row.shape[1]:
                curve_img = cv2.resize(
                    curve_img,
                    (cam_row.shape[1], curve_img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            frame = np.concatenate([cam_row, curve_img], axis=0)
        else:
            frame = cam_row

        # Ensure even dimensions for H.264
        h, w = frame.shape[:2]
        if h % 2 != 0:
            frame = frame[:h - 1]
        if frame.shape[1] % 2 != 0:
            frame = frame[:, :frame.shape[1] - 1]

        writer.append_data(frame)

    writer.close()
    print(f"  Saved: {output_path} ({T} frames, {T/fps:.1f}s)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Inovo (RoboKit) data as videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory containing episode sub-directories.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for videos. Default: {data_dir}/viz_videos.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Specific episode directory name to visualize.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=-1,
        help="Number of episodes to visualize (-1 = all). Default: -1.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS. Default: 30.",
    )
    parser.add_argument(
        "--no_curves",
        action="store_true",
        help="Disable trajectory curve visualization (faster).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Target height for camera images in video. Default: 360.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "viz_videos"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    # Discover episodes
    episodes = discover_episodes(data_dir)
    if not episodes:
        print(f"ERROR: No episodes found in {data_dir}")
        return

    # Filter by specific episode name
    if args.episode:
        episodes = [ep for ep in episodes if ep["name"] == args.episode]
        if not episodes:
            print(f"ERROR: Episode '{args.episode}' not found.")
            return

    # Limit number
    if args.num_episodes > 0:
        episodes = episodes[:args.num_episodes]

    print(f"\n{'='*60}")
    print(f"  Inovo Data Visualization")
    print(f"{'='*60}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  FPS: {args.fps}")
    print(f"  Include curves: {not args.no_curves}")
    print(f"{'='*60}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, ep in enumerate(episodes):
        print(f"\n[{i+1}/{len(episodes)}] Episode: {ep['name']}")
        output_path = out_dir / f"{ep['name']}.mp4"
        create_visualization_video(
            episode_name=ep["name"],
            npz_files=ep["npz_files"],
            output_path=output_path,
            fps=args.fps,
            include_curves=not args.no_curves,
            target_height=args.height,
        )

    print(f"\nDone! {len(episodes)} videos saved to {out_dir}")


if __name__ == "__main__":
    main()
