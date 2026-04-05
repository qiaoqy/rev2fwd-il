#!/usr/bin/env python3
"""Fast visualization of advantage-filtered reversed B episodes.

Re-runs critic inference on selected episodes and generates videos with
keep/drop overlay. Uses pre-rendered static curve images (not per-frame matplotlib)
for speed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug/visualize_advantage_filter.py \
        --reversed_B_path debug/data/iter3_rollout_B_reversed.npz \
        --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
        --episode_indices 0,1,2,10,50 \
        --out_dir debug/data/filtered_reversed_B_v1 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rev2fwd_il.data.advantage_estimation import (
    compute_gae_from_values,
    compute_frame_filter_mask,
)
from debug.eval_critic_visual import (
    load_critic_model,
    predict_episode_values,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reversed_B_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episode_indices", type=str, default="0,1,2,10,50",
                        help="Comma-separated episode indices to visualize")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    # Stage 1: Value truncation
    parser.add_argument("--value_truncate", type=float, default=0.99,
                        help="Truncate episode at first timestep where V(t) >= this. 0 = disable.")
    parser.add_argument("--truncate_margin", type=int, default=30,
                        help="Keep this many extra frames after the truncation point")
    # GAE params
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    # Stage 2: Advantage filter params
    parser.add_argument("--smooth_window", type=int, default=16)
    parser.add_argument("--drop_threshold", type=float, default=0.02)
    parser.add_argument("--min_drop_length", type=int, default=15)
    parser.add_argument("--min_keep_length", type=int, default=15)
    return parser.parse_args()


def _smooth_advantages(advantages: np.ndarray, smooth_window: int) -> np.ndarray:
    T = len(advantages)
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    pad = smooth_window // 2
    padded = np.pad(advantages, (pad, pad), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[:T]


def render_static_curves(
    pred_values: np.ndarray,
    smoothed_adv: np.ndarray,
    keep_mask: np.ndarray,
    plot_w: int,
    plot_h: int,
) -> np.ndarray:
    """Pre-render the full curve image once (no per-frame matplotlib).
    
    Returns:
        (img, x_left_px, x_right_px): rendered image and pixel columns 
        corresponding to data x=0 and x=T.
    """
    dpi = 100
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(plot_w / dpi, plot_h / dpi), dpi=dpi,
        height_ratios=[1, 1],
    )
    T = len(pred_values)
    ts = np.arange(T)

    # Draw keep/drop bands
    for ax in [ax1, ax2]:
        i = 0
        while i < T:
            val = keep_mask[i]
            j = i
            while j < T and keep_mask[j] == val:
                j += 1
            color = (0.85, 1.0, 0.85, 0.5) if val else (1.0, 0.85, 0.85, 0.5)
            ax.axvspan(i, j, color=color)
            i = j

    # Value curve
    ax1.plot(ts, pred_values, "r-", linewidth=1.5, alpha=0.9)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-0.1, 1.15)
    ax1.set_ylabel("V(t)", fontsize=7)
    ax1.set_title("Value V(t)", fontsize=8)
    ax1.tick_params(labelsize=5)
    ax1.grid(True, alpha=0.2)

    # Advantage curve
    ax2.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
    ax2.plot(ts, smoothed_adv, "b-", linewidth=1.5, alpha=0.9)
    adv_range = max(abs(smoothed_adv.min()), abs(smoothed_adv.max()), 0.05) * 1.3
    ax2.set_xlim(0, T)
    ax2.set_ylim(-adv_range, adv_range)
    ax2.set_ylabel("A(t)", fontsize=7)
    ax2.set_xlabel("Timestep", fontsize=7)
    ax2.set_title("Smoothed Advantage A(t)", fontsize=8)
    ax2.tick_params(labelsize=5)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    # Get data-area pixel bounds from axes position
    # Use ax2 (bottom axis) — same x-range as ax1
    bbox = ax2.get_position()  # in figure fraction coords
    fig_w_px = fig.get_size_inches()[0] * dpi
    x_left_px = int(bbox.x0 * fig_w_px)
    x_right_px = int(bbox.x1 * fig_w_px)

    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    if img.shape[0] != plot_h or img.shape[1] != plot_w:
        # Scale the pixel bounds proportionally
        scale_x = plot_w / img.shape[1]
        x_left_px = int(x_left_px * scale_x)
        x_right_px = int(x_right_px * scale_x)
        img = np.array(Image.fromarray(img).resize((plot_w, plot_h), Image.LANCZOS))

    return img, x_left_px, x_right_px


def create_fast_filter_video(
    ep: dict,
    pred_values: np.ndarray,
    smoothed_adv: np.ndarray,
    keep_mask: np.ndarray,
    out_path: str,
    ep_idx: int = 0,
    fps: int = 20,
):
    """Fast video: camera + static curve with cursor line overlay."""
    import imageio

    images = ep["images"]
    wrist = ep.get("wrist_images", None)
    T = len(images)

    H, W = images.shape[1], images.shape[2]
    cam_w = W * 2 + 10 if wrist is not None else W
    plot_w = cam_w
    plot_h = max(int(H * 0.8), 128)
    bar_h = 28
    total_h = H + bar_h + plot_h
    total_h = total_h + (total_h % 2)
    total_w = cam_w + (cam_w % 2)

    # Pre-render static curves (returns data-area pixel bounds)
    static_curves, x_left_px, x_right_px = render_static_curves(
        pred_values, smoothed_adv, keep_mask, plot_w, plot_h,
    )
    T_data = len(pred_values)

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
        is_keep = bool(keep_mask[t]) if t < len(keep_mask) else True

        # Status bar
        bar_color = (0, 180, 0) if is_keep else (200, 0, 0)
        canvas[:bar_h, :total_w, :] = bar_color

        # Overlay text on status bar using PIL
        pil_bar = Image.fromarray(canvas[:bar_h, :total_w])
        draw = ImageDraw.Draw(pil_bar)
        status_text = f"  t={t}/{T}  {'KEEP' if is_keep else 'DROP'}  V={pred_values[t]:.3f}  A={smoothed_adv[t]:.4f}"
        draw.text((4, 4), status_text, fill=(255, 255, 255))
        canvas[:bar_h, :total_w] = np.array(pil_bar)

        # Camera images with colored border
        y_off = bar_h
        border = 3
        bc = (0, 200, 0) if is_keep else (220, 0, 0)

        canvas[y_off:y_off + H, :W] = images[t]
        canvas[y_off:y_off + border, :W] = bc
        canvas[y_off + H - border:y_off + H, :W] = bc
        canvas[y_off:y_off + H, :border] = bc
        canvas[y_off:y_off + H, W - border:W] = bc

        if wrist is not None:
            x_off = W + 10
            canvas[y_off:y_off + H, x_off:x_off + W] = wrist[t]
            canvas[y_off:y_off + border, x_off:x_off + W] = bc
            canvas[y_off + H - border:y_off + H, x_off:x_off + W] = bc
            canvas[y_off:y_off + H, x_off:x_off + border] = bc
            canvas[y_off:y_off + H, x_off + W - border:x_off + W] = bc

        # Static curves + moving cursor (aligned to data area)
        plot_y = y_off + H
        curve_frame = static_curves.copy()
        # Map timestep t to pixel x within the data area
        if T_data > 1:
            cursor_x = x_left_px + int(t / (T_data - 1) * (x_right_px - x_left_px))
        else:
            cursor_x = (x_left_px + x_right_px) // 2
        cursor_x = max(0, min(cursor_x, plot_w - 1))
        curve_frame[:, max(0, cursor_x - 1):cursor_x + 2, :] = (0, 0, 0)  # black cursor
        canvas[plot_y:plot_y + plot_h, :plot_w] = curve_frame

        writer.append_data(canvas)

    writer.close()
    print(f"    Video saved: {out_path}  ({T} frames)")


def _draw_keep_drop_bands(ax, keep_mask, T):
    """Draw green/red background bands for keep/drop regions."""
    i = 0
    while i < T:
        val = keep_mask[i]; j = i
        while j < T and keep_mask[j] == val: j += 1
        color = (0.85, 1.0, 0.85, 0.5) if val else (1.0, 0.85, 0.85, 0.5)
        ax.axvspan(i, j, color=color); i = j


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
    if truncate_t < T_orig:
        ax1.axvspan(truncate_t, T_orig, color=(0.8, 0.8, 0.8, 0.5), label="Truncated (V≥thresh)")
        ax1.axvline(x=truncate_t, color="purple", linewidth=2, linestyle="--", label=f"Truncate t={truncate_t}")
    ts_full = np.arange(T_orig)
    ax1.plot(ts_full, pred_values_full, "r-", linewidth=1.5, alpha=0.9, label="V(t) full")
    ax1.axhline(y=0.99, color="purple", linewidth=0.8, linestyle=":", alpha=0.6, label="V=0.99 threshold")
    ax1.set_ylabel("Value V(t)", fontsize=10)
    ax1.set_xlim(0, T_orig)
    ax1.set_ylim(-0.1, 1.15)
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
    ax2.set_ylim(-0.1, max(pred_values_trunc.max() * 1.1, 0.5))
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

    # === Row 4: Sampled frames ===
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
            spine.set_edgecolor(color); spine.set_linewidth(3)
        ax_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax2.axvline(x=fi, color="orange", linestyle=":", alpha=0.4)
        ax3.axvline(x=fi, color="orange", linestyle=":", alpha=0.4)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Overview saved: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(x) for x in args.episode_indices.split(",")]

    # Load data
    print(f"Loading reversed B data from {args.reversed_B_path}")
    with np.load(args.reversed_B_path, allow_pickle=True) as data:
        all_episodes = list(data["episodes"])
    # Success only
    all_episodes = [ep for ep in all_episodes if ep.get("success", False)]
    print(f"  {len(all_episodes)} success episodes")

    # Load critic
    model = load_critic_model(args.checkpoint, args.device)

    for ep_idx in indices:
        if ep_idx >= len(all_episodes):
            print(f"  Skipping ep {ep_idx} (only {len(all_episodes)} episodes)")
            continue

        ep = all_episodes[ep_idx]
        T_orig = len(ep["action"])
        print(f"\n  Episode {ep_idx} (T_orig={T_orig})...")

        # Inference (full episode)
        pred_values_full = predict_episode_values(
            model, ep, horizon=args.horizon, n_obs_steps=args.n_obs_steps,
            device=args.device, batch_size=args.batch_size,
        )

        # Stage 1: Truncate at value threshold
        truncate_t = T_orig
        if args.value_truncate > 0:
            above_thresh = np.where(pred_values_full >= args.value_truncate)[0]
            if len(above_thresh) > 0:
                truncate_t = min(int(above_thresh[0]) + args.truncate_margin, T_orig)
                truncate_t = max(truncate_t, 10)

        if truncate_t < T_orig:
            ep_trunc = {}
            for k, v in ep.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == T_orig:
                    ep_trunc[k] = v[:truncate_t]
                else:
                    ep_trunc[k] = v
            pred_values = pred_values_full[:truncate_t]
            print(f"    Stage 1: Truncated at t={truncate_t}")
        else:
            ep_trunc = ep
            pred_values = pred_values_full
            print(f"    Stage 1: No truncation")

        T = len(pred_values)

        # Stage 2: GAE + filter on truncated data
        advantages = compute_gae_from_values(pred_values, gamma=args.gamma, lam=args.lam, terminal_bootstrap=True)
        smoothed_adv = _smooth_advantages(advantages, args.smooth_window)

        keep_mask = compute_frame_filter_mask(
            advantages,
            smooth_window=args.smooth_window,
            drop_threshold=args.drop_threshold,
            min_drop_length=args.min_drop_length,
            min_keep_length=args.min_keep_length,
        )

        n_kept = keep_mask.sum()
        print(f"    Stage 2: Keep {n_kept}/{T} truncated ({n_kept/T:.1%}), "
              f"{n_kept}/{T_orig} original ({n_kept/T_orig:.1%})")

        ep_dir = out_dir / f"ep{ep_idx}"
        ep_dir.mkdir(exist_ok=True)

        # Overview (2-stage)
        plot_filter_overview_2stage(
            ep, pred_values_full, pred_values, advantages, smoothed_adv,
            keep_mask, truncate_t, ep_idx, str(ep_dir / "overview.png"),
        )

        # Video (fast: on truncated data only)
        create_fast_filter_video(
            ep_trunc, pred_values, smoothed_adv, keep_mask,
            str(ep_dir / "video.mp4"), ep_idx=ep_idx, fps=args.fps,
        )

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
