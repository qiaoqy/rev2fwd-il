#!/usr/bin/env python3
"""Evaluate critic model on reversed B rollout data.

Pipeline:
1. Merge B rollout partitions → iter3_rollout_B_all.npz
2. Reverse all episodes (no filter/speed-adjust) → iter3_rollout_B_reversed.npz
3. Run critic model inference on reversed episodes
4. Visualize 5 episodes (value curve + overview + video)

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug/eval_critic_on_reversed_B.py \
        --data_dir debug/data \
        --checkpoint debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt \
        --num_episodes 5 \
        --out_dir debug/data/eval_critic_reversed_B \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")

import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import reverse_episode from 2_reverse_to_task_A.py (can't import directly due to leading digit)
_reverse_spec = importlib.util.spec_from_file_location(
    "reverse_to_task_A",
    str(Path(__file__).resolve().parent.parent / "scripts" / "scripts_pick_place_simulator" / "2_reverse_to_task_A.py"),
)
_reverse_mod = importlib.util.module_from_spec(_reverse_spec)
_reverse_spec.loader.exec_module(_reverse_mod)
reverse_episode = _reverse_mod.reverse_episode

from debug.eval_critic_visual import (
    load_critic_model,
    build_state,
    predict_episode_values,
    plot_overview_with_frames,
    create_episode_video,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate critic on reversed B rollout data")
    parser.add_argument("--data_dir", type=str, default="debug/data")
    parser.add_argument("--checkpoint", type=str,
                        default="debug/data/critic_A_ddp_v2/checkpoints/final/checkpoint.pt")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_video_frames", type=int, default=0,
                        help="Max video frames (0 = use full episode length)")
    parser.add_argument("--out_dir", type=str, default="debug/data/eval_critic_reversed_B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prefix", type=str, default="iter3_rollout")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Skip merge step if iter3_rollout_B_all.npz already exists")
    parser.add_argument("--skip_reverse", action="store_true",
                        help="Skip reverse step if iter3_rollout_B_reversed.npz already exists")
    return parser.parse_args()


# ============================================================
# Step 1: Merge B rollout partitions
# ============================================================

def merge_partitions(data_dir: Path, prefix: str) -> Path:
    """Merge iter3_rollout_B_p*.npz into iter3_rollout_B_all.npz."""
    out_path = data_dir / f"{prefix}_B_all.npz"

    all_episodes = []
    part_idx = 0
    while True:
        p = data_dir / f"{prefix}_B_p{part_idx}.npz"
        if not p.exists():
            break
        with np.load(p, allow_pickle=True) as data:
            eps = list(data["episodes"])
        print(f"  Loaded {p.name}: {len(eps)} episodes")
        all_episodes.extend(eps)
        part_idx += 1

    if part_idx == 0:
        raise FileNotFoundError(f"No partition files found: {data_dir}/{prefix}_B_p*.npz")

    n_success = sum(1 for ep in all_episodes if ep.get("success", False))
    print(f"  Merged: {len(all_episodes)} episodes ({n_success} success, "
          f"{len(all_episodes) - n_success} failure) from {part_idx} partitions")

    np.savez_compressed(out_path, episodes=np.array(all_episodes, dtype=object))
    mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {out_path} ({mb:.1f} MB)")
    return out_path


# ============================================================
# Step 2: Reverse episodes (no filter, no speed-adjust)
# ============================================================

def reverse_all_episodes(input_path: Path, output_path: Path) -> Path:
    """Time-reverse all episodes (success_only=0, keep all)."""
    with np.load(input_path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"  Loaded {len(episodes)} episodes from {input_path.name}")

    reversed_eps = []
    for i, ep in enumerate(episodes):
        rev = reverse_episode(ep)
        reversed_eps.append(rev)
        if (i + 1) % 20 == 0 or i == 0:
            T = len(rev["images"])
            print(f"    Reversed ep {i}: T={T}, success={ep.get('success', False)}")

    n_success = sum(1 for ep in reversed_eps if ep.get("success", False))
    print(f"  Reversed: {len(reversed_eps)} episodes ({n_success} success)")

    np.savez_compressed(output_path, episodes=np.array(reversed_eps, dtype=object))
    mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({mb:.1f} MB)")
    return output_path


# ============================================================
# Step 3+4: Critic inference + visualization
# ============================================================

def plot_value_curve_no_gt(ep: dict, pred_values: np.ndarray, ep_idx: int, out_path: str):
    """Plot predicted value curve (no GT bellman since this is reversed B data)."""
    import matplotlib.pyplot as plt

    T = len(pred_values)
    success = ep.get("success", False)
    success_step_orig = ep.get("success_step", None)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[2, 1])
    fig.suptitle(
        f"Episode {ep_idx} — T={T}, B_success={success}"
        + (f" (orig_B_success_step={success_step_orig})" if success_step_orig else "")
        + " [Reversed B → Critic A pred]",
        fontsize=13,
    )
    ts = np.arange(T)

    # Top: predicted value curve
    ax = axes[0]
    ax.plot(ts, pred_values, "r-", linewidth=1.5, alpha=0.8, label="Critic Predicted V(t)")
    ax.set_ylabel("Value")
    ax.set_xlabel("Timestep")
    ax.set_title("Critic Value Prediction on Reversed B Rollout")
    ax.set_xlim(0, T)
    ax.set_ylim(-0.1, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Bottom: EE pose XYZ
    ax2 = axes[1]
    ee = ep["ee_pose"]
    for dim, (color, label) in enumerate(zip(["r", "g", "b"], ["X", "Y", "Z"])):
        ax2.plot(ts, ee[:T, dim], color=color, alpha=0.7, linewidth=1, label=label)
    ax2.set_ylabel("EE Position")
    ax2.set_xlabel("Timestep")
    ax2.set_title("End-Effector XYZ Trajectory (Reversed)")
    ax2.set_xlim(0, T)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Value curve saved: {out_path}")


def plot_aggregate_reversed_B(all_results: list[dict], out_path: str):
    """Aggregate plot of all selected episodes' predicted value curves."""
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle("Critic A Prediction on Reversed B Rollout", fontsize=14, y=1.002)

    for i, res in enumerate(all_results):
        ax = axes[i, 0]
        pv = res["pred"]
        T = len(pv)
        ts = np.arange(T)
        success = res["success"]

        ax.plot(ts, pv, "r-", linewidth=1.2, alpha=0.8, label="Critic Pred V(t)")
        ax.set_xlim(0, T)
        ax.set_ylim(-0.1, 1.15)
        ax.grid(True, alpha=0.3)

        tag = "B_success" if success else "B_failure"
        title = f"Ep {res['ep_idx']} ({tag}) — T={T}"
        title += f" | mean_V={pv.mean():.3f}, max_V={pv.max():.3f}"
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Aggregate plot saved: {out_path}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Critic Evaluation on Reversed B Rollout")
    print(f"  Data dir: {data_dir}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # --- Step 1: Merge B partitions ---
    merged_path = data_dir / f"{args.prefix}_B_all.npz"
    if args.skip_merge and merged_path.exists():
        print(f"\n[Step 1] Skipping merge, using existing {merged_path}")
    else:
        print(f"\n[Step 1] Merging B rollout partitions...")
        merge_partitions(data_dir, args.prefix)

    # --- Step 2: Reverse ---
    reversed_path = data_dir / f"{args.prefix}_B_reversed.npz"
    if args.skip_reverse and reversed_path.exists():
        print(f"\n[Step 2] Skipping reverse, using existing {reversed_path}")
    else:
        print(f"\n[Step 2] Reversing B episodes (no filter/speed-adjust)...")
        reverse_all_episodes(merged_path, reversed_path)

    # --- Step 3: Load reversed data + critic model ---
    print(f"\n[Step 3] Loading reversed data and critic model...")
    with np.load(reversed_path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    n_success = sum(1 for ep in episodes if ep.get("success", False))
    n_failure = len(episodes) - n_success
    print(f"  {len(episodes)} reversed episodes ({n_success} B_success, {n_failure} B_failure)")

    model = load_critic_model(args.checkpoint, args.device)

    # Select episodes: mix of success + failure
    success_eps = [(i, ep) for i, ep in enumerate(episodes) if ep.get("success", False)]
    failure_eps = [(i, ep) for i, ep in enumerate(episodes) if not ep.get("success", False)]

    n_vis = min(args.num_episodes, len(episodes))
    n_fail_vis = min(max(1, n_vis // 3), len(failure_eps))
    n_succ_vis = min(n_vis - n_fail_vis, len(success_eps))
    if n_fail_vis > len(failure_eps):
        n_fail_vis = len(failure_eps)
        n_succ_vis = min(n_vis - n_fail_vis, len(success_eps))

    selected = success_eps[:n_succ_vis] + failure_eps[:n_fail_vis]
    print(f"  Selected {len(selected)} episodes for visualization "
          f"({n_succ_vis} success, {n_fail_vis} failure)")

    # --- Step 4: Inference + visualization ---
    print(f"\n[Step 4] Running critic inference and visualization...")
    all_results = []
    for idx, (ep_i, ep) in enumerate(selected):
        success_tag = "B_success" if ep.get("success", False) else "B_failure"
        T = len(ep["action"])
        print(f"\n  [{idx+1}/{len(selected)}] Episode {ep_i} ({success_tag}, T={T})...")

        pred_values = predict_episode_values(
            model, ep,
            horizon=args.horizon,
            n_obs_steps=args.n_obs_steps,
            device=args.device,
            batch_size=args.batch_size,
        )

        print(f"    mean_V={pred_values.mean():.4f}, max_V={pred_values.max():.4f}, "
              f"min_V={pred_values.min():.4f}")

        all_results.append({
            "ep_idx": ep_i,
            "pred": pred_values,
            "success": ep.get("success", False),
        })

        ep_dir = out_dir / f"ep{ep_i}_{success_tag}"
        ep_dir.mkdir(exist_ok=True)

        # Value curve (no GT)
        plot_value_curve_no_gt(ep, pred_values, ep_i, str(ep_dir / "value_curve.png"))

        # Overview with frames (fake bellman_value for compatibility)
        ep_with_bv = dict(ep)
        ep_with_bv["bellman_value"] = pred_values  # use pred as "GT" for overview layout
        plot_overview_with_frames(ep_with_bv, pred_values, ep_i, str(ep_dir / "overview.png"))

        # Video
        ep_for_video = dict(ep)
        ep_for_video["bellman_value"] = pred_values
        max_frames = args.max_video_frames if args.max_video_frames > 0 else len(ep["action"])
        create_episode_video(
            ep_for_video, pred_values, str(ep_dir / "video.mp4"),
            ep_idx=ep_i, fps=args.fps, max_frames=max_frames,
        )

    # Aggregate plot
    print(f"\nGenerating aggregate plot...")
    plot_aggregate_reversed_B(all_results, str(out_dir / "aggregate_pred_vs_gt.png"))

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for res in all_results:
        tag = "B_success" if res["success"] else "B_failure"
        pv = res["pred"]
        print(f"  Ep {res['ep_idx']:3d} ({tag:10s}): "
              f"T={len(pv):5d}, mean_V={pv.mean():.4f}, max_V={pv.max():.4f}")
    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
