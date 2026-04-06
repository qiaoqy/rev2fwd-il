#!/usr/bin/env python3
"""Filter reversed B rollout data using GAE advantage (frame-level).

Pipeline:
1. Load reversed B data (success only) OR reverse on-the-fly
2. Load trained critic model
3. For each episode: critic inference → V(t) → GAE advantage A(t) → frame keep/drop mask
4. Visualize: video with keep/drop overlay per episode
5. Save filtered episodes (only kept frames)

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug/filter_reversed_B_by_advantage.py \
        --reversed_B_path debug/data/iter3_rollout_B_reversed.npz \
        --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
        --gamma 0.995 --lam 0.95 \
        --smooth_window 51 --drop_threshold 0.0 \
        --min_drop_length 50 --min_keep_length 50 \
        --out_dir debug/data/filtered_reversed_B \
        --device cuda:0

    If --reversed_B_path does not exist, provide --raw_B_dir to merge+reverse automatically:
    CUDA_VISIBLE_DEVICES=0 python debug/filter_reversed_B_by_advantage.py \
        --raw_B_dir debug/data \
        --raw_B_prefix iter3_rollout \
        --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
        --out_dir debug/data/filtered_reversed_B \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rev2fwd_il.data.advantage_estimation import (
    compute_gae_from_values,
    compute_frame_filter_mask,
    filter_episode_frames,
)
from debug.eval_critic_visual import (
    load_critic_model,
    build_state,
    predict_episode_values,
)


# ============================================================
# Data loading helpers
# ============================================================

def _get_reverse_episode_fn():
    """Dynamically import reverse_episode from 2_reverse_to_task_A.py."""
    spec = importlib.util.spec_from_file_location(
        "reverse_to_task_A",
        str(Path(__file__).resolve().parent.parent
            / "scripts" / "scripts_pick_place_simulator" / "2_reverse_to_task_A.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reverse_episode


def load_reversed_B_episodes(
    reversed_B_path: str | None = None,
    raw_B_dir: str | None = None,
    raw_B_prefix: str = "iter3_rollout",
    success_only: bool = True,
) -> list[dict]:
    """Load reversed B episodes, or merge+reverse from raw partitions.

    Args:
        reversed_B_path: Path to pre-reversed .npz file. Used if exists.
        raw_B_dir: Directory containing B partition files (p0, p1, ...).
        raw_B_prefix: Filename prefix for partitions.
        success_only: If True, only keep episodes where original B was successful.

    Returns:
        List of reversed episode dicts.
    """
    if reversed_B_path and Path(reversed_B_path).exists():
        print(f"Loading reversed B data from {reversed_B_path}")
        with np.load(reversed_B_path, allow_pickle=True) as data:
            episodes = list(data["episodes"])
        if success_only:
            episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"  {len(episodes)} episodes" + (" (success only)" if success_only else ""))
        return episodes

    if raw_B_dir is None:
        raise FileNotFoundError(
            f"reversed_B_path={reversed_B_path} not found and raw_B_dir not specified."
        )

    # Merge partitions
    raw_dir = Path(raw_B_dir)
    all_episodes = []
    part_idx = 0
    while True:
        p = raw_dir / f"{raw_B_prefix}_B_p{part_idx}.npz"
        if not p.exists():
            break
        with np.load(p, allow_pickle=True) as data:
            eps = list(data["episodes"])
        print(f"  Loaded {p.name}: {len(eps)} episodes")
        all_episodes.extend(eps)
        part_idx += 1

    if part_idx == 0:
        raise FileNotFoundError(f"No partition files: {raw_dir}/{raw_B_prefix}_B_p*.npz")

    if success_only:
        all_episodes = [ep for ep in all_episodes if ep.get("success", False)]
    print(f"  Merged: {len(all_episodes)} episodes" + (" (success only)" if success_only else ""))

    # Reverse
    reverse_episode = _get_reverse_episode_fn()
    reversed_eps = []
    for i, ep in enumerate(all_episodes):
        rev = reverse_episode(ep)
        reversed_eps.append(rev)
    print(f"  Reversed: {len(reversed_eps)} episodes")

    return reversed_eps


# ============================================================
# Visualization
# ============================================================

def _render_filter_frame(
    pred_values: np.ndarray,
    advantages: np.ndarray,
    smoothed_adv: np.ndarray,
    keep_mask: np.ndarray,
    t: int,
    plot_w: int,
    plot_h: int,
) -> np.ndarray:
    """Render a single video frame: value curve + advantage + keep/drop coloring."""
    dpi = 100
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(plot_w / dpi, plot_h / dpi), dpi=dpi,
        height_ratios=[1, 1],
    )
    T = len(pred_values)
    ts = np.arange(T)

    # ---- Top: Value curve with keep/drop background ----
    # Background coloring
    for tt in range(T - 1):
        color = (0.85, 1.0, 0.85) if keep_mask[tt] else (1.0, 0.85, 0.85)
        ax1.axvspan(tt, tt + 1, color=color, alpha=0.5)

    # Ghost full trajectory
    ax1.plot(ts, pred_values, color="lightsalmon", linewidth=0.8)
    # Up to current t
    ax1.plot(ts[:t + 1], pred_values[:t + 1], "r-", linewidth=1.5, label="V(t)")
    ax1.plot(t, pred_values[t], "ro", markersize=4)

    status = "KEEP" if keep_mask[t] else "DROP"
    status_color = "green" if keep_mask[t] else "red"
    ax1.set_title(f"Value  t={t}/{T}  [{status}]", fontsize=8, color=status_color)
    ax1.set_xlim(0, T)
    v_lo, v_hi = pred_values.min(), pred_values.max()
    v_margin = max(abs(v_hi - v_lo) * 0.1, 0.05)
    ax1.set_ylim(v_lo - v_margin, v_hi + v_margin)
    ax1.set_ylabel("V(t)", fontsize=7)
    ax1.tick_params(labelsize=5)
    ax1.grid(True, alpha=0.2)

    # ---- Bottom: Advantage curve ----
    for tt in range(T - 1):
        color = (0.85, 1.0, 0.85) if keep_mask[tt] else (1.0, 0.85, 0.85)
        ax2.axvspan(tt, tt + 1, color=color, alpha=0.5)

    ax2.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax2.plot(ts, smoothed_adv, color="lightskyblue", linewidth=0.8)
    ax2.plot(ts[:t + 1], smoothed_adv[:t + 1], "b-", linewidth=1.2, label="Smoothed A(t)")
    ax2.plot(t, smoothed_adv[t], "bo", markersize=3)

    ax2.set_title(f"Smoothed Advantage  A={smoothed_adv[t]:.4f}", fontsize=8)
    ax2.set_xlim(0, T)
    adv_range = max(abs(smoothed_adv.min()), abs(smoothed_adv.max()), 0.05) * 1.3
    ax2.set_ylim(-adv_range, adv_range)
    ax2.set_xlabel("Timestep", fontsize=7)
    ax2.set_ylabel("A(t)", fontsize=7)
    ax2.tick_params(labelsize=5)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    if img.shape[0] != plot_h or img.shape[1] != plot_w:
        from PIL import Image
        img = np.array(Image.fromarray(img).resize((plot_w, plot_h), Image.LANCZOS))

    return img


def create_filter_video(
    ep: dict,
    pred_values: np.ndarray,
    advantages: np.ndarray,
    smoothed_adv: np.ndarray,
    keep_mask: np.ndarray,
    out_path: str,
    ep_idx: int = 0,
    fps: int = 20,
    max_frames: int = 0,
):
    """Video: camera images with KEEP/DROP border + value/advantage curves underneath."""
    import imageio

    images = ep["images"]
    wrist = ep.get("wrist_images", None)
    T_full = len(images)
    T = T_full if max_frames <= 0 else min(T_full, max_frames)

    H, W = images.shape[1], images.shape[2]
    cam_w = W * 2 + 10 if wrist is not None else W
    plot_w = cam_w
    plot_h = max(int(H * 0.8), 128)
    # Status bar height
    bar_h = 24
    total_h = H + bar_h + plot_h
    total_h = total_h + (total_h % 2)  # ensure even
    total_w = cam_w + (cam_w % 2)

    if T < T_full:
        print(f"    Video: using first {T}/{T_full} frames")

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # Status bar (KEEP=green, DROP=red)
        is_keep = keep_mask[t] if t < len(keep_mask) else True
        bar_color = (0, 180, 0) if is_keep else (200, 0, 0)
        canvas[:bar_h, :, :] = bar_color

        # Status text rendered as colored bar (text via matplotlib is too slow per-frame)
        # The color itself is the indicator

        # Camera images (below status bar)
        y_off = bar_h
        # Add colored border (3px)
        border = 3
        border_color = (0, 200, 0) if is_keep else (220, 0, 0)
        canvas[y_off:y_off + H, :W] = images[t]
        # Top/bottom/left/right border
        canvas[y_off:y_off + border, :W] = border_color
        canvas[y_off + H - border:y_off + H, :W] = border_color
        canvas[y_off:y_off + H, :border] = border_color
        canvas[y_off:y_off + H, W - border:W] = border_color

        if wrist is not None:
            x_off = W + 10
            canvas[y_off:y_off + H, x_off:x_off + W] = wrist[t]
            canvas[y_off:y_off + border, x_off:x_off + W] = border_color
            canvas[y_off + H - border:y_off + H, x_off:x_off + W] = border_color
            canvas[y_off:y_off + H, x_off:x_off + border] = border_color
            canvas[y_off:y_off + H, x_off + W - border:x_off + W] = border_color

        # Plot area
        plot_y = y_off + H
        curve_img = _render_filter_frame(
            pred_values, advantages, smoothed_adv, keep_mask,
            t, plot_w, plot_h,
        )
        canvas[plot_y:plot_y + plot_h, :plot_w] = curve_img

        writer.append_data(canvas)

    writer.close()
    print(f"    Video saved: {out_path}  ({T} frames)")


def plot_filter_overview_2stage(
    ep_full: dict,
    pred_values_full: np.ndarray,
    pred_values_trunc: np.ndarray,
    advantages: np.ndarray,
    smoothed_adv: np.ndarray,
    keep_mask: np.ndarray,
    truncate_t: int,
    ep_idx: int,
    out_path: str,
):
    """2-stage overview: full V(t) with truncation + truncated advantage + keep/drop + frames."""
    T_orig = len(pred_values_full)
    T_trunc = len(pred_values_trunc)
    n_kept = keep_mask.sum()
    images = ep_full["images"]

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 6, height_ratios=[1.2, 1.2, 1.2, 1], hspace=0.45, wspace=0.3)

    # === Row 1: Full V(t) with truncation ===
    ax1 = fig.add_subplot(gs[0, :])
    # Shading: kept region (before truncation + keep_mask), truncated region
    # First shade the truncated tail
    if truncate_t < T_orig:
        ax1.axvspan(truncate_t, T_orig, color=(0.8, 0.8, 0.8, 0.5), label="Truncated (V≥thresh)")
        ax1.axvline(x=truncate_t, color="purple", linewidth=2, linestyle="--", label=f"Truncate t={truncate_t}")
    ts_full = np.arange(T_orig)
    ax1.plot(ts_full, pred_values_full, "r-", linewidth=1.5, alpha=0.9, label="V(t) full")
    ax1.set_ylabel("Value V(t)", fontsize=10)
    ax1.set_xlim(0, T_orig)
    vf_lo, vf_hi = pred_values_full.min(), pred_values_full.max()
    vf_margin = max(abs(vf_hi - vf_lo) * 0.1, 0.05)
    ax1.set_ylim(vf_lo - vf_margin, vf_hi + vf_margin)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(
        f"Episode {ep_idx} — Stage 1: T_orig={T_orig} → truncated at t={truncate_t} "
        f"(dropped {T_orig - truncate_t} frames)",
        fontsize=11,
    )

    # === Row 2: Truncated V(t) with keep/drop bands ===
    ax2 = fig.add_subplot(gs[1, :])
    ts_trunc = np.arange(T_trunc)
    _draw_keep_drop_bands(ax2, keep_mask, T_trunc)
    ax2.plot(ts_trunc, pred_values_trunc, "r-", linewidth=1.5, alpha=0.9, label="V(t) truncated")
    ax2.set_ylabel("Value V(t)", fontsize=10)
    ax2.set_xlim(0, T_trunc)
    vt_lo, vt_hi = pred_values_trunc.min(), pred_values_trunc.max()
    vt_margin = max(abs(vt_hi - vt_lo) * 0.1, 0.05)
    ax2.set_ylim(vt_lo - vt_margin, vt_hi + vt_margin)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_title(
        f"Stage 2: Advantage filter on truncated ({T_trunc} frames) → kept {n_kept}/{T_trunc} "
        f"({n_kept/T_trunc:.1%}), overall {n_kept}/{T_orig} ({n_kept/T_orig:.1%})",
        fontsize=11,
    )

    # === Row 3: Advantage curve (truncated) ===
    ax3 = fig.add_subplot(gs[2, :], sharex=ax2)
    _draw_keep_drop_bands(ax3, keep_mask, T_trunc)
    ax3.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax3.plot(ts_trunc, advantages[:T_trunc], color="lightblue", linewidth=0.5, alpha=0.5, label="Raw A(t)")
    ax3.plot(ts_trunc, smoothed_adv, "b-", linewidth=1.5, alpha=0.9, label="Smoothed A(t)")
    adv_range = max(abs(smoothed_adv.min()), abs(smoothed_adv.max()), 0.05) * 1.3
    ax3.set_ylim(-adv_range, adv_range)
    ax3.set_ylabel("Advantage A(t)", fontsize=10)
    ax3.set_xlabel("Timestep (truncated)", fontsize=10)
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper left", fontsize=8)

    # === Row 4: Sampled frames from truncated portion ===
    frame_indices = np.linspace(0, T_trunc - 1, 6, dtype=int)
    for col, fi in enumerate(frame_indices):
        ax_img = fig.add_subplot(gs[3, col])
        ax_img.imshow(images[fi])
        status = "KEEP" if keep_mask[fi] else "DROP"
        color = "green" if keep_mask[fi] else "red"
        ax_img.set_title(
            f"t={fi} [{status}]\nV={pred_values_trunc[fi]:.3f} A={smoothed_adv[fi]:.4f}",
            fontsize=7, color=color,
        )
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax2.axvline(x=fi, color="orange", linestyle=":", alpha=0.4)
        ax3.axvline(x=fi, color="orange", linestyle=":", alpha=0.4)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Overview saved: {out_path}")


def _draw_keep_drop_bands(ax, keep_mask, T):
    """Draw green/red background bands for keep/drop regions."""
    i = 0
    while i < T:
        val = keep_mask[i]
        j = i
        while j < T and keep_mask[j] == val:
            j += 1
        color = (0.85, 1.0, 0.85, 0.5) if val else (1.0, 0.85, 0.85, 0.5)
        ax.axvspan(i, j, color=color)
        i = j


def plot_aggregate_summary(all_stats: list[dict], out_path: str):
    """Summary plot across all episodes: truncation + advantage filtering stats."""
    n = len(all_stats)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Per-episode frame counts: original → truncated → kept
    ax = axes[0]
    ep_ids = [s["ep_idx"] for s in all_stats]
    orig_lens = [s["original_length"] for s in all_stats]
    trunc_lens = [s["truncated_length"] for s in all_stats]
    kept_lens = [s["kept_length"] for s in all_stats]
    y = range(n)
    ax.barh(y, orig_lens, color="lightgray", alpha=0.7, label="Original")
    ax.barh(y, trunc_lens, color="lightskyblue", alpha=0.7, label="After truncation")
    ax.barh(y, kept_lens, color="limegreen", alpha=0.8, label="After adv filter")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"Ep {i}" for i in ep_ids], fontsize=7)
    ax.set_xlabel("Frames")
    ax.set_title("Frame Count: Original → Truncated → Kept")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # 2. Keep ratio of truncated (stage 2 effectiveness)
    ax = axes[1]
    keep_of_trunc = [s["keep_ratio_of_truncated"] for s in all_stats]
    keep_of_orig = [s["keep_ratio_of_original"] for s in all_stats]
    ax.hist(keep_of_trunc, bins=min(20, n), alpha=0.7, color="steelblue", edgecolor="white", label="% of truncated")
    ax.hist(keep_of_orig, bins=min(20, n), alpha=0.5, color="coral", edgecolor="white", label="% of original")
    ax.axvline(x=0.5, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Keep Ratio")
    ax.set_title("Keep Ratio Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # 3. Scatter: truncate_t vs kept_length
    ax = axes[2]
    trunc_ts = [s["truncate_t"] for s in all_stats]
    mean_vals = [s["mean_value_trunc"] for s in all_stats]
    ax.scatter(trunc_ts, kept_lens, c=mean_vals, cmap="RdYlGn", s=40, alpha=0.7, edgecolors="k", linewidth=0.5)
    ax.set_xlabel("Truncation Point t")
    ax.set_ylabel("Kept Frames")
    ax.set_title("Truncation Point vs Kept Frames (color=mean V)")
    ax.grid(True, alpha=0.2)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Mean Value (truncated)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregate summary saved: {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter reversed B rollout frames using GAE advantage"
    )
    # Data
    parser.add_argument("--reversed_B_path", type=str, default=None,
                        help="Path to pre-reversed B .npz file")
    parser.add_argument("--raw_B_dir", type=str, default=None,
                        help="Directory with B partition files (used if reversed_B_path missing)")
    parser.add_argument("--raw_B_prefix", type=str, default="iter3_rollout")

    # Critic
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to critic checkpoint.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for critic inference")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)

    # GAE params
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--terminal_bootstrap", action="store_true", default=True,
                        help="Bootstrap terminal value (default: True)")
    parser.add_argument("--no_terminal_bootstrap", dest="terminal_bootstrap", action="store_false")

    # Stage 1: Value truncation
    parser.add_argument("--value_truncate", type=float, default=None,
                        help="Truncate episode at first timestep where V(t) >= this value. Omit to disable.")
    parser.add_argument("--truncate_margin", type=int, default=30,
                        help="Keep this many extra frames after the truncation point")
    parser.add_argument("--step_reward", type=float, default=0.0,
                        help="Per-step reward for GAE TD error: r_bar(t). "
                             "For MC value normalized to [-1,0): use -1/(2*R_success), e.g. -1/6000. "
                             "0 = sparse reward (Exp36 default).")

    # Stage 2: Advantage filter params
    parser.add_argument("--smooth_window", type=int, default=16,
                        help="Moving average window for advantage smoothing")
    parser.add_argument("--drop_threshold", type=float, default=0.02,
                        help="Smoothed advantage below this → drop candidate")
    parser.add_argument("--min_drop_length", type=int, default=15,
                        help="Drop segments shorter than this are converted back to keep")
    parser.add_argument("--min_keep_length", type=int, default=15,
                        help="Keep segments shorter than this (sandwiched by drops) become drop")

    # Output
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_video_frames", type=int, default=0,
                        help="0 = full episode length")
    parser.add_argument("--num_vis_episodes", type=int, default=5,
                        help="Number of episodes to visualize (videos + overview)")

    return parser.parse_args()


def _smooth_advantages(advantages: np.ndarray, smooth_window: int) -> np.ndarray:
    """Centered moving average smoothing (same logic as compute_frame_filter_mask)."""
    T = len(advantages)
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    pad = smooth_window // 2
    padded = np.pad(advantages, (pad, pad), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[:T]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gamma == 1.0 and args.step_reward != 0.0 and args.terminal_bootstrap:
        print(
            "WARNING: gamma=1 with non-zero step_reward is usually paired with "
            "--no_terminal_bootstrap for MC-return consistency."
        )

    print(f"{'=' * 60}")
    print(f"GAE Advantage Frame-Level Filter for Reversed B Rollout")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Stage 1: Truncate at V(t) >= {args.value_truncate}" if args.value_truncate is not None else "  Stage 1: No truncation")
    print(f"  Stage 2: GAE γ={args.gamma}, λ={args.lam}, bootstrap={args.terminal_bootstrap}, step_reward={args.step_reward}")
    print(f"  Filter: window={args.smooth_window}, threshold={args.drop_threshold}, "
          f"min_drop={args.min_drop_length}, min_keep={args.min_keep_length}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # --- 1. Load data (success B only, reversed) ---
    print(f"\n[Step 1] Loading reversed B episodes (success only)...")
    episodes = load_reversed_B_episodes(
        reversed_B_path=args.reversed_B_path,
        raw_B_dir=args.raw_B_dir,
        raw_B_prefix=args.raw_B_prefix,
        success_only=True,
    )
    print(f"  Loaded {len(episodes)} reversed B episodes (all originally successful)")

    # --- 2. Load critic ---
    print(f"\n[Step 2] Loading critic model...")
    model = load_critic_model(args.checkpoint, args.device)

    # --- 3. Per-episode: inference → GAE → filter ---
    print(f"\n[Step 3] Running per-episode inference + advantage filtering...")
    filtered_episodes = []
    all_stats = []

    # For visualization
    vis_data = []

    for i, ep in enumerate(episodes):
        T_orig = len(ep["action"])
        print(f"\n  Episode {i}/{len(episodes)} (T_orig={T_orig})...")

        # Critic inference (full episode)
        pred_values_full = predict_episode_values(
            model, ep,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            device=args.device,
            batch_size=args.batch_size,
        )

        # --- Stage 1: Truncate at value threshold ---
        truncate_t = T_orig
        if args.value_truncate is not None:
            above_thresh = np.where(pred_values_full >= args.value_truncate)[0]
            if len(above_thresh) > 0:
                truncate_t = int(above_thresh[0]) + args.truncate_margin
                # Clamp to episode length, ensure at least some frames remain
                truncate_t = min(truncate_t, T_orig)
                truncate_t = max(truncate_t, 10)

        if truncate_t < T_orig:
            # Truncate episode arrays
            ep_trunc = {}
            for k, v in ep.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == T_orig:
                    ep_trunc[k] = v[:truncate_t]
                else:
                    ep_trunc[k] = v
            pred_values = pred_values_full[:truncate_t]
            print(f"    Stage 1: Truncated at t={truncate_t} (V={pred_values_full[truncate_t]:.4f}), "
                  f"dropped {T_orig - truncate_t} frames")
        else:
            ep_trunc = ep
            pred_values = pred_values_full
            print(f"    Stage 1: No truncation (V never reached {args.value_truncate})")

        T = len(pred_values)

        # --- Stage 2: GAE + advantage filter on truncated data ---
        # Build per-step reward array: r_bar(t) = step_reward for all t
        rewards = np.full(T, args.step_reward, dtype=np.float32) if args.step_reward != 0.0 else None
        advantages = compute_gae_from_values(
            pred_values,
            gamma=args.gamma,
            lam=args.lam,
            rewards=rewards,
            terminal_bootstrap=args.terminal_bootstrap,
        )

        smoothed_adv = _smooth_advantages(advantages, args.smooth_window)

        keep_mask = compute_frame_filter_mask(
            advantages,
            smooth_window=args.smooth_window,
            drop_threshold=args.drop_threshold,
            min_drop_length=args.min_drop_length,
            min_keep_length=args.min_keep_length,
        )

        n_kept = keep_mask.sum()
        keep_ratio_of_trunc = n_kept / T
        keep_ratio_of_orig = n_kept / T_orig
        print(f"    V(trunc): mean={pred_values.mean():.4f}, max={pred_values.max():.4f}")
        print(f"    A(trunc): mean={advantages.mean():.4f}, std={advantages.std():.4f}")
        print(f"    Stage 2: kept {n_kept}/{T} truncated frames ({keep_ratio_of_trunc:.1%}), "
              f"{n_kept}/{T_orig} original ({keep_ratio_of_orig:.1%})")

        # Filter the truncated episode
        filtered_ep = filter_episode_frames(ep_trunc, keep_mask)
        filtered_episodes.append(filtered_ep)

        # Stats
        stats = {
            "ep_idx": i,
            "original_length": T_orig,
            "truncated_length": T,
            "truncate_t": truncate_t,
            "kept_length": int(n_kept),
            "keep_ratio_of_truncated": float(keep_ratio_of_trunc),
            "keep_ratio_of_original": float(keep_ratio_of_orig),
            "mean_value_full": float(pred_values_full.mean()),
            "max_value_full": float(pred_values_full.max()),
            "mean_value_trunc": float(pred_values.mean()),
            "mean_adv": float(advantages.mean()),
            "std_adv": float(advantages.std()),
            "positive_adv_ratio": float((advantages > 0).mean()),
        }
        all_stats.append(stats)

        # Store for visualization
        if len(vis_data) < args.num_vis_episodes:
            vis_data.append({
                "ep_idx": i,
                "episode": ep,  # original full episode for viz
                "ep_trunc": ep_trunc,
                "pred_values_full": pred_values_full,
                "pred_values": pred_values,
                "advantages": advantages,
                "smoothed_adv": smoothed_adv,
                "keep_mask": keep_mask,
                "truncate_t": truncate_t,
            })

    # --- 4. Save filtered data ---
    print(f"\n[Step 4] Saving filtered data...")
    np.savez_compressed(
        out_dir / "filtered_episodes.npz",
        episodes=np.array(filtered_episodes, dtype=object),
    )
    mb = (out_dir / "filtered_episodes.npz").stat().st_size / (1024 * 1024)
    print(f"  Saved: filtered_episodes.npz ({mb:.1f} MB)")

    # Save stats
    # Convert stats to JSON-serializable (no numpy)
    with open(out_dir / "filter_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Saved: filter_stats.json")

    # Save filter config
    config = {
        "value_truncate": args.value_truncate,
        "gamma": args.gamma,
        "lam": args.lam,
        "step_reward": args.step_reward,
        "terminal_bootstrap": args.terminal_bootstrap,
        "smooth_window": args.smooth_window,
        "drop_threshold": args.drop_threshold,
        "min_drop_length": args.min_drop_length,
        "min_keep_length": args.min_keep_length,
        "checkpoint": args.checkpoint,
        "num_episodes": len(episodes),
        "num_filtered_episodes": len(filtered_episodes),
    }
    with open(out_dir / "filter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- 5. Visualization ---
    print(f"\n[Step 5] Generating visualizations for {len(vis_data)} episodes...")
    print(f"  (Skipping slow per-frame video; use visualize_advantage_filter.py for fast videos)")
    for vd in vis_data:
        ep_dir = out_dir / f"ep{vd['ep_idx']}"
        ep_dir.mkdir(exist_ok=True)

        # Overview plot (2-stage: full V(t) + truncation + advantage filter)
        plot_filter_overview_2stage(
            vd["episode"], vd["pred_values_full"], vd["pred_values"],
            vd["advantages"], vd["smoothed_adv"], vd["keep_mask"],
            vd["truncate_t"], vd["ep_idx"], str(ep_dir / "overview.png"),
        )

    # Aggregate summary
    plot_aggregate_summary(all_stats, str(out_dir / "aggregate_summary.png"))

    # --- Summary ---
    total_orig = sum(s["original_length"] for s in all_stats)
    total_trunc = sum(s["truncated_length"] for s in all_stats)
    total_kept = sum(s["kept_length"] for s in all_stats)
    mean_keep_of_trunc = np.mean([s["keep_ratio_of_truncated"] for s in all_stats])
    mean_keep_of_orig = np.mean([s["keep_ratio_of_original"] for s in all_stats])

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Stage 1 (truncate at V>={args.value_truncate}): {total_orig} → {total_trunc} frames ({total_trunc/total_orig:.1%})")
    print(f"  Stage 2 (advantage filter): {total_trunc} → {total_kept} frames ({total_kept/total_trunc:.1%} of truncated)")
    print(f"  Overall: {total_orig} → {total_kept} frames ({total_kept/total_orig:.1%} of original)")
    print(f"  Mean keep ratio (of truncated): {mean_keep_of_trunc:.1%}")
    print(f"  Mean keep ratio (of original): {mean_keep_of_orig:.1%}")
    print(f"  Output: {out_dir}")
    print(f"\nPer-episode:")
    for s in all_stats:
        print(f"  Ep {s['ep_idx']:3d}: T_orig={s['original_length']:5d} → trunc={s['truncated_length']:5d} "
              f"→ kept={s['kept_length']:5d} ({s['keep_ratio_of_truncated']:.1%} of trunc, "
              f"{s['keep_ratio_of_original']:.1%} of orig), mean_V={s['mean_value_trunc']:.3f}")


if __name__ == "__main__":
    main()
