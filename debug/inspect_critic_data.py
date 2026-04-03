#!/usr/bin/env python3
"""Phase 2: Inspect and visualize critic data with Bellman return labels.

Generates for each inspected episode:
1. Side-by-side camera video (table + wrist)
2. Bellman value curve plot
3. Combined value curve + camera frame overview image

Also generates aggregate plots across all episodes.

Usage:
    python debug/inspect_critic_data.py \
        --dataset debug/data/critic_A_train.npz \
        --num_episodes 5 --fps 20

    # Inspect test set
    python debug/inspect_critic_data.py \
        --dataset debug/data/critic_A_test.npz \
        --num_episodes 3
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect critic data with Bellman return labels")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to critic_*_train.npz or critic_*_test.npz")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to visualize (default: 5)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video FPS (default: 20)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Defaults to debug/data/inspect_critic_<timestamp>")
    return parser.parse_args()


def load_labeled_episodes(path: str) -> list[dict]:
    """Load labeled episodes from NPZ file."""
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    return episodes


def _write_video_h264(frames: list[np.ndarray], out_path: str, fps: int = 20):
    """Write frames as H.264 MP4 using imageio-ffmpeg (VS Code compatible)."""
    import imageio

    h, w = frames[0].shape[:2]
    h = h if h % 2 == 0 else h + 1
    w = w if w % 2 == 0 else w + 1

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        if frame.shape[0] != h or frame.shape[1] != w:
            padded = np.full((h, w, 3), 255, dtype=np.uint8)
            padded[:frame.shape[0], :frame.shape[1]] = frame
            frame = padded
        writer.append_data(frame)
    writer.close()


def _render_value_curve_frame(
    bv: np.ndarray,
    t: int,
    success_step: int | None,
    plot_w: int,
    plot_h: int,
) -> np.ndarray:
    """Render a single frame of the real-time value curve as an RGB image.

    Shows the full value trajectory in light grey, the portion up to t
    in blue, a red dot at the current timestep, and a green dashed line
    at success_step.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(plot_w / dpi, plot_h / dpi), dpi=dpi)
    T = len(bv)
    ts = np.arange(T)

    # Full trajectory (ghost)
    ax.plot(ts, bv, color="lightgrey", linewidth=1)
    # Trajectory up to current t
    ax.plot(ts[: t + 1], bv[: t + 1], "b-", linewidth=1.5)
    # Current point
    ax.plot(t, bv[t], "ro", markersize=5)
    # Annotation
    ax.text(
        t, bv[t] + 0.06, f"{bv[t]:.4f}",
        fontsize=8, ha="center", color="red",
        clip_on=True,
    )

    if success_step is not None and success_step < T:
        ax.axvline(x=success_step, color="green", linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("V(t)", fontsize=8)
    ax.set_title(f"Bellman Value  t={t}/{T}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    # Convert matplotlib figure to numpy RGB array
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()  # drop alpha
    plt.close(fig)

    # Resize to exact target if needed (matplotlib rounding)
    if img.shape[0] != plot_h or img.shape[1] != plot_w:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((plot_w, plot_h), Image.LANCZOS))

    return img  # (plot_h, plot_w, 3) uint8


def create_episode_video(ep: dict, out_path: str, ep_idx: int = 0,
                         fps: int = 20, max_frames: int = 1000):
    """Create a video with camera images on top and real-time value curve below.

    Layout (vertically stacked):
        ┌──────────────┬──────────────┐
        │  Table cam   │  Wrist cam   │  (or just table cam)
        ├──────────────┴──────────────┤
        │     Real-time value curve   │
        └─────────────────────────────┘
    """
    import imageio

    images = ep["images"]
    wrist = ep.get("wrist_images", None)
    bv = ep.get("bellman_value", None)
    success = ep.get("success", False)
    success_step = ep.get("success_step", None)
    T_full = len(images)
    T = min(T_full, max_frames)

    H, W = images.shape[1], images.shape[2]

    # Camera row width
    if wrist is not None:
        cam_w = W * 2 + 10
    else:
        cam_w = W

    # Value curve plot dimensions (match camera row width)
    plot_w = cam_w
    plot_h = max(int(H * 0.75), 96)  # 75% of camera height, at least 96px

    # Ensure even dimensions for H.264
    total_h = H + plot_h
    total_h = total_h if total_h % 2 == 0 else total_h + 1
    total_w = cam_w if cam_w % 2 == 0 else cam_w + 1

    if T < T_full:
        print(f"    Video: using first {T}/{T_full} frames")

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # Top row: camera images
        canvas[:H, :W] = images[t]
        if wrist is not None:
            canvas[:H, W + 10: W + 10 + W] = wrist[t]

        # Bottom row: value curve
        if bv is not None:
            curve_img = _render_value_curve_frame(bv, t, success_step, plot_w, plot_h)
            canvas[H: H + plot_h, :plot_w] = curve_img

        writer.append_data(canvas)

    writer.close()
    print(f"    Video saved: {out_path}  ({T} frames, {total_w}x{total_h})")


def plot_bellman_value_curve(ep: dict, ep_idx: int, out_path: str):
    """Plot Bellman value curve for a single episode.

    Shows the value curve with key annotations:
    - success_step marker
    - V(0) and V(success_step) labels
    - EE pose XYZ overlay (secondary axis)
    """
    bv = ep["bellman_value"]
    T = len(bv)
    success = ep.get("success", False)
    success_step = ep.get("success_step", None)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    fig.suptitle(
        f"Episode {ep_idx} — T={T}, success={success}"
        + (f", success_step={success_step}" if success_step is not None else ""),
        fontsize=13,
    )

    timesteps = np.arange(T)

    # --- Top plot: Bellman value curve ---
    ax = axes[0]
    ax.plot(timesteps, bv, "b-", linewidth=1.5, label="Bellman value V(t)")
    ax.set_ylabel("Value V(t)")
    ax.set_xlabel("Timestep")
    ax.set_title("Bellman Return (γ-discounted)")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    if success and success_step is not None:
        ss = min(success_step, T - 1)
        ax.axvline(x=ss, color="green", linestyle="--", alpha=0.7, label=f"success_step={ss}")
        ax.plot(ss, bv[ss], "go", markersize=8)
        ax.annotate(f"V({ss})={bv[ss]:.4f}", xy=(ss, bv[ss]),
                    xytext=(ss + T * 0.02, bv[ss] - 0.1), fontsize=9)
        ax.annotate(f"V(0)={bv[0]:.6f}", xy=(0, bv[0]),
                    xytext=(T * 0.05, bv[0] + 0.1), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="blue"))
    ax.legend(loc="upper left", fontsize=9)

    # --- Bottom plot: EE pose XYZ ---
    ax2 = axes[1]
    ee = ep["ee_pose"]
    for dim, (color, label) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        ax2.plot(timesteps, ee[:, dim], color=color, alpha=0.7, linewidth=1, label=label)
    ax2.set_ylabel("EE Position")
    ax2.set_xlabel("Timestep")
    ax2.set_title("End-Effector XYZ Trajectory")
    ax2.set_xlim(0, T)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    if success and success_step is not None:
        ss = min(success_step, T - 1)
        ax2.axvline(x=ss, color="green", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Value curve saved: {out_path}")


def plot_overview_with_frames(ep: dict, ep_idx: int, out_path: str):
    """Create an overview image: value curve + sampled camera frames."""
    bv = ep["bellman_value"]
    T = len(bv)
    success = ep.get("success", False)
    success_step = ep.get("success_step", None)
    images = ep["images"]

    # Sample 5 evenly spaced frames
    frame_indices = np.linspace(0, T - 1, 5, dtype=int)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 5, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Top: value curve spanning all columns
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(np.arange(T), bv, "b-", linewidth=1.5, label="V(t)")
    ax_curve.set_ylabel("Value V(t)")
    ax_curve.set_xlabel("Timestep")
    ax_curve.set_xlim(0, T)
    ax_curve.set_ylim(-0.05, 1.1)
    ax_curve.grid(True, alpha=0.3)

    if success and success_step is not None:
        ss = min(success_step, T - 1)
        ax_curve.axvline(x=ss, color="green", linestyle="--", alpha=0.7)

    # Mark sampled frames on the curve
    for fi in frame_indices:
        ax_curve.axvline(x=fi, color="orange", linestyle=":", alpha=0.5)
        ax_curve.plot(fi, bv[fi], "ro", markersize=6)

    ax_curve.set_title(
        f"Episode {ep_idx} — T={T}, success={success}"
        + (f", success_step={success_step}" if success_step is not None else ""),
        fontsize=12,
    )
    ax_curve.legend(loc="upper left")

    # Bottom: sampled camera frames
    for col, fi in enumerate(frame_indices):
        ax_img = fig.add_subplot(gs[1, col])
        ax_img.imshow(images[fi])
        ax_img.set_title(f"t={fi}\nV={bv[fi]:.4f}", fontsize=9)
        ax_img.axis("off")

    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Overview saved: {out_path}")


def plot_aggregate_value_curves(episodes: list[dict], out_path: str, title: str):
    """Plot all episodes' Bellman value curves overlayed on one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=14)

    # Left: success episodes
    ax = axes[0]
    ax.set_title("Successful Episodes")
    n_succ = 0
    for i, ep in enumerate(episodes):
        if ep.get("success", False):
            bv = ep["bellman_value"]
            color = plt.cm.viridis(n_succ / max(1, sum(1 for e in episodes if e.get("success"))))
            ax.plot(np.arange(len(bv)), bv, color=color, alpha=0.5, linewidth=0.8)
            n_succ += 1
    ax.set_xlabel("Timestep")
    ax.set_ylabel("V(t)")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f"n={n_succ}", transform=ax.transAxes, fontsize=11, va="top")

    # Right: failure episodes
    ax = axes[1]
    ax.set_title("Failed Episodes")
    n_fail = 0
    for i, ep in enumerate(episodes):
        if not ep.get("success", False):
            bv = ep["bellman_value"]
            ax.plot(np.arange(len(bv)), bv, "r-", alpha=0.5, linewidth=0.8)
            n_fail += 1
    ax.set_xlabel("Timestep")
    ax.set_ylabel("V(t)")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f"n={n_fail}", transform=ax.transAxes, fontsize=11, va="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregate value curves saved: {out_path}")


def plot_normalized_value_curves(episodes: list[dict], out_path: str, title: str):
    """Plot value curves normalized to [0, 1] timestep range (for length comparison)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title(title, fontsize=13)

    for i, ep in enumerate(episodes):
        if not ep.get("success", False):
            continue
        bv = ep["bellman_value"]
        T = len(bv)
        ss = ep.get("success_step", T - 1)
        ss = min(ss, T - 1)
        # Normalize timestep to [0, 1] up to success_step
        t_norm = np.linspace(0, 1, ss + 1)
        color = plt.cm.tab20(i % 20)
        ax.plot(t_norm, bv[:ss + 1], color=color, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("Normalized Timestep (0 = start, 1 = success)")
    ax.set_ylabel("V(t)")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Normalized value curves saved: {out_path}")


def print_summary_table(episodes: list[dict], label: str):
    """Print a summary table of episodes."""
    print(f"\n  {'='*70}")
    print(f"  {label}: {len(episodes)} episodes")
    print(f"  {'='*70}")
    print(f"  {'Ep':>4} {'T':>6} {'Success':>8} {'SStep':>8} {'V(0)':>10} {'V(ss)':>8} {'V(T-1)':>10}")
    print(f"  {'----':>4} {'------':>6} {'--------':>8} {'--------':>8} {'----------':>10} {'--------':>8} {'----------':>10}")

    for i, ep in enumerate(episodes):
        bv = ep["bellman_value"]
        T = len(bv)
        success = ep.get("success", False)
        ss = ep.get("success_step", None)
        v0 = bv[0]
        vT = bv[-1]
        vss = bv[min(ss, T - 1)] if ss is not None else 0.0
        ss_str = str(ss) if ss is not None else "N/A"
        print(f"  {i:>4} {T:>6} {str(success):>8} {ss_str:>8} {v0:>10.6f} {vss:>8.4f} {vT:>10.6f}")


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stem = dataset_path.stem  # e.g. critic_A_train
        out_dir = dataset_path.parent / f"inspect_{stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Inspecting critic data: {dataset_path.name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    # Load
    print(f"\nLoading episodes...")
    episodes = load_labeled_episodes(str(dataset_path))
    n_success = sum(1 for ep in episodes if ep.get("success", False))
    n_failure = len(episodes) - n_success
    print(f"  {len(episodes)} episodes ({n_success} success, {n_failure} failure)")

    # Verify bellman_value exists
    for i, ep in enumerate(episodes):
        if "bellman_value" not in ep:
            print(f"  ERROR: episode {i} missing 'bellman_value' field!")
            return

    # Print summary table
    print_summary_table(episodes, dataset_path.stem)

    # Aggregate plots (all episodes)
    print(f"\nGenerating aggregate plots...")
    plot_aggregate_value_curves(
        episodes, str(out_dir / "all_value_curves.png"), f"{dataset_path.stem} — All Value Curves"
    )
    plot_normalized_value_curves(
        episodes, str(out_dir / "normalized_value_curves.png"),
        f"{dataset_path.stem} — Normalized Value Curves (success only)",
    )

    # Per-episode visualizations
    n_vis = min(args.num_episodes, len(episodes))
    print(f"\nGenerating per-episode visualizations ({n_vis} episodes)...")

    # Pick a mix: first few success + first few failure
    success_eps = [(i, ep) for i, ep in enumerate(episodes) if ep.get("success", False)]
    failure_eps = [(i, ep) for i, ep in enumerate(episodes) if not ep.get("success", False)]

    # Allocate: try to show at least 1 failure if available
    n_fail_vis = min(max(1, n_vis // 4), len(failure_eps))
    n_succ_vis = min(n_vis - n_fail_vis, len(success_eps))
    if n_fail_vis > len(failure_eps):
        n_fail_vis = len(failure_eps)
        n_succ_vis = min(n_vis - n_fail_vis, len(success_eps))

    vis_eps = success_eps[:n_succ_vis] + failure_eps[:n_fail_vis]

    for idx, (ep_i, ep) in enumerate(vis_eps):
        success_tag = "success" if ep.get("success", False) else "failure"
        print(f"\n  Episode {ep_i} ({success_tag}):")

        ep_dir = out_dir / f"ep{ep_i}_{success_tag}"
        ep_dir.mkdir(exist_ok=True)

        # Value curve plot
        plot_bellman_value_curve(ep, ep_i, str(ep_dir / "value_curve.png"))

        # Overview with sampled frames
        plot_overview_with_frames(ep, ep_i, str(ep_dir / "overview.png"))

        # Episode video with real-time value curve
        create_episode_video(ep, str(ep_dir / "video.mp4"), ep_idx=ep_i, fps=args.fps)

    print(f"\n{'='*60}")
    print(f"Inspection complete! Files saved to {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
