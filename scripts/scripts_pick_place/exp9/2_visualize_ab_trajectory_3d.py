#!/usr/bin/env python3
"""Visualize the first trajectory in 3D for two datasets (Task A and Task B).

This script mirrors the style of scripts_pick_place/2_inspect_data.py and creates
three 3D trajectory plots:
1) Task A trajectory in XYZ space
2) Task B trajectory in XYZ space
3) Task A / Task B overlay with transparency

Usage examples:
python scripts/scripts_pick_place/exp9/2_visualize_ab_trajectory_3d.py

python scripts/scripts_pick_place/exp9/2_visualize_ab_trajectory_3d.py \
    --episode_a 0 --episode_b 0
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Task A / Task B first trajectory in 3D and their transparent overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_a",
        type=str,
        default="data/pick_place_isaac_lab_simulation/exp9/replay_taskA_from_iter0_ep0.npz",
        help="Path to Task A dataset NPZ.",
    )
    parser.add_argument(
        "--dataset_b",
        type=str,
        default="data/pick_place_isaac_lab_simulation/exp9/replay_taskB_from_iter0_ep0.npz",
        help="Path to Task B dataset NPZ.",
    )
    parser.add_argument(
        "--episode_a",
        type=int,
        default=0,
        help="Episode index for Task A (default: 0).",
    )
    parser.add_argument(
        "--episode_b",
        type=int,
        default=0,
        help="Episode index for Task B (default: 0).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="replay_iter0_ep0_3d",
        help="Optional tag for output folder name.",
    )
    return parser.parse_args()


def load_episode_data(npz_path: str, episode_idx: int) -> dict:
    """Load one episode dict from Episode-format NPZ.

    Expected format: {'episodes': [episode_dict, ...]}
    """
    data = np.load(npz_path, allow_pickle=True)

    if "episodes" not in data.files:
        raise ValueError(f"Invalid dataset format in {npz_path}. Keys: {data.files}")

    episodes = data["episodes"]
    if episode_idx < 0 or episode_idx >= len(episodes):
        raise ValueError(f"Episode {episode_idx} out of range. Total: {len(episodes)}")

    episode = episodes[episode_idx]
    if not isinstance(episode, dict):
        raise ValueError(f"Episode {episode_idx} is not a dict. Type: {type(episode)}")

    return episode


def extract_xyz_trajectory(episode: dict) -> np.ndarray:
    """Extract XYZ trajectory with shape (T, 3)."""
    if "ee_pose" in episode:
        ee_pose = np.asarray(episode["ee_pose"])
        if ee_pose.ndim != 2 or ee_pose.shape[1] < 3:
            raise ValueError(f"Invalid ee_pose shape: {ee_pose.shape}")
        return ee_pose[:, :3]

    if "obs" in episode:
        obs = np.asarray(episode["obs"])
        if obs.ndim != 2 or obs.shape[1] < 3:
            raise ValueError(f"Invalid obs shape: {obs.shape}")
        return obs[:, :3]

    raise ValueError("No 'ee_pose' or 'obs' key found for XYZ extraction.")


def set_axes_equal(ax: plt.Axes, xyz: np.ndarray) -> None:
    """Set equal scale for all 3 axes for a correct geometric view."""
    x_vals, y_vals, z_vals = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    z_min, z_max = float(z_vals.min()), float(z_vals.max())

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    half_range = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    half_range = max(half_range, 1e-4)

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)


def get_equal_time_indices(length: int, num_points: int = 5) -> np.ndarray:
    """Get indices for evenly spaced temporal markers along a trajectory."""
    if length <= 0:
        raise ValueError("Trajectory length must be positive.")
    if length == 1:
        return np.array([0], dtype=int)

    idx = np.linspace(0, length - 1, num_points)
    idx = np.round(idx).astype(int)
    # Keep order while removing duplicates for short trajectories.
    return np.unique(idx)


def annotate_time_markers(
    ax: plt.Axes,
    xyz: np.ndarray,
    mark_idx: np.ndarray,
    text_color: str,
    offset_mode: str,
    fontsize: int = 10,
) -> None:
    """Add numbered labels (1, 2, 3, ...) to time marker points."""
    # Offset labels slightly from marker points to reduce overlap.
    span = np.ptp(xyz, axis=0)
    span = np.where(span > 1e-8, span, 1.0)
    dx = 0.02 * float(span[0])
    dy = 0.02 * float(span[1])
    dz = 0.03 * float(span[2])

    if offset_mode == "left_up":
        x_off, y_off = -dx, dy
    elif offset_mode == "right_down":
        x_off, y_off = dx, -dy
    else:
        x_off, y_off = dx, dy

    for label_num, idx in enumerate(mark_idx, start=1):
        text_artist = ax.text(
            xyz[idx, 0] + x_off,
            xyz[idx, 1] + y_off,
            xyz[idx, 2] + dz,
            str(label_num),
            color=text_color,
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=120,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.35},
        )
        text_artist.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.2, foreground="white"),
                path_effects.Normal(),
            ]
        )


def plot_3d_trajectory_on_axis(
    ax: plt.Axes,
    xyz: np.ndarray,
    title: str,
    color: str,
    marker_shape: str,
    marker_label: str,
    time_label_color: str,
    time_label_offset_mode: str,
) -> None:
    """Draw one 3D trajectory on a provided axis."""

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2.0)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color="green", s=45, label="start")
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color="red", s=45, label="end")

    mark_idx = get_equal_time_indices(len(xyz), num_points=5)
    ax.scatter(
        xyz[mark_idx, 0],
        xyz[mark_idx, 1],
        xyz[mark_idx, 2],
        color=color,
        marker=marker_shape,
        s=60,
        alpha=0.95,
        label=marker_label,
    )
    annotate_time_markers(
        ax=ax,
        xyz=xyz,
        mark_idx=mark_idx,
        text_color=time_label_color,
        offset_mode=time_label_offset_mode,
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    set_axes_equal(ax, xyz)


def save_three_panel_figure(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    output_path: Path,
    episode_a: int,
    episode_b: int,
) -> None:
    """Save one canvas with 3 panels: A, B, and transparent overlay (A+B)."""
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    plot_3d_trajectory_on_axis(
        ax=ax1,
        xyz=xyz_a,
        title=f"Task A 3D Trajectory (episode {episode_a})",
        color="tab:blue",
        marker_shape="^",
        marker_label="A time markers",
        time_label_color="black",
        time_label_offset_mode="left_up",
    )
    plot_3d_trajectory_on_axis(
        ax=ax2,
        xyz=xyz_b,
        title=f"Task B 3D Trajectory (episode {episode_b})",
        color="tab:orange",
        marker_shape="o",
        marker_label="B time markers",
        time_label_color="red",
        time_label_offset_mode="right_down",
    )
    ax3.plot(
        xyz_a[:, 0], xyz_a[:, 1], xyz_a[:, 2], color="tab:blue", linewidth=2.0, alpha=0.45, label="A"
    )
    ax3.plot(
        xyz_b[:, 0], xyz_b[:, 1], xyz_b[:, 2], color="tab:orange", linewidth=2.0, alpha=0.45, label="B"
    )
    ax3.scatter(xyz_a[0, 0], xyz_a[0, 1], xyz_a[0, 2], color="tab:blue", s=30, alpha=0.8)
    ax3.scatter(xyz_b[0, 0], xyz_b[0, 1], xyz_b[0, 2], color="tab:orange", s=30, alpha=0.8)
    ax3.scatter(xyz_a[-1, 0], xyz_a[-1, 1], xyz_a[-1, 2], color="tab:blue", marker="x", s=55, alpha=0.9)
    ax3.scatter(xyz_b[-1, 0], xyz_b[-1, 1], xyz_b[-1, 2], color="tab:orange", marker="x", s=55, alpha=0.9)

    idx_a = get_equal_time_indices(len(xyz_a), num_points=5)
    idx_b = get_equal_time_indices(len(xyz_b), num_points=5)
    ax3.scatter(
        xyz_a[idx_a, 0],
        xyz_a[idx_a, 1],
        xyz_a[idx_a, 2],
        color="tab:blue",
        marker="^",
        s=62,
        alpha=0.95,
        label="A time markers",
    )
    annotate_time_markers(ax3, xyz_a, idx_a, text_color="black", offset_mode="left_up")
    ax3.scatter(
        xyz_b[idx_b, 0],
        xyz_b[idx_b, 1],
        xyz_b[idx_b, 2],
        color="tab:orange",
        marker="o",
        s=52,
        alpha=0.95,
        label="B time markers",
    )
    annotate_time_markers(ax3, xyz_b, idx_b, text_color="red", offset_mode="right_down")
    ax3.set_title("3D Overlay (A and B)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend(loc="best")
    set_axes_equal(ax3, np.vstack([xyz_a, xyz_b]))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    dataset_a = Path(args.dataset_a)
    dataset_b = Path(args.dataset_b)

    if not dataset_a.exists():
        raise FileNotFoundError(f"Task A dataset not found: {dataset_a}")
    if not dataset_b.exists():
        raise FileNotFoundError(f"Task B dataset not found: {dataset_b}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("data") / "pick_place_isaac_lab_simulation" / "exp9" / f"viz_ab_3d_{args.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Loading Task A: {dataset_a}")
    print(f"Loading Task B: {dataset_b}")

    ep_a = load_episode_data(str(dataset_a), args.episode_a)
    ep_b = load_episode_data(str(dataset_b), args.episode_b)

    xyz_a = extract_xyz_trajectory(ep_a)
    xyz_b = extract_xyz_trajectory(ep_b)

    if len(xyz_a) == 0 or len(xyz_b) == 0:
        raise ValueError("One of the trajectories is empty.")

    print(f"Task A trajectory length: {len(xyz_a)}")
    print(f"Task B trajectory length: {len(xyz_b)}")

    combined_path = output_dir / "trajectory_A_B_overlay_3d.png"

    save_three_panel_figure(
        xyz_a=xyz_a,
        xyz_b=xyz_b,
        output_path=combined_path,
        episode_a=args.episode_a,
        episode_b=args.episode_b,
    )

    print("Saved figure:")
    print(f"  - {combined_path}")


if __name__ == "__main__":
    main()
