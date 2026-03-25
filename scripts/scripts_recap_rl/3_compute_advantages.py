#!/usr/bin/env python3
"""Step 3: Compute per-frame advantages and generate binary indicators.

Loads the trained value function from step 2, runs inference on all frames
in the provided NPZ dataset, and produces an augmented NPZ that includes a
per-frame binary indicator field 'indicators' (List[int], 0 or 1).

Advantage:  A_t = R_t - V(o_t)
Threshold:  ε = percentile_30(A_values across whole dataset)
Indicator:  I_t = 1 if A_t > ε  (positive = this action was above average)

With success rate ~50% and percentile=30, ~70% of frames become positive.

Usage:
    python scripts/scripts_recap_rl/3_compute_advantages.py \\
        --policy data/exp_new/weights/PP_A/.../pretrained_model \\
        --vf_ckpt data/recap_exp/vf_A.pt \\
        --npz_paths data/exp_new/task_A_reversed_100.npz \\
                    data/recap_exp/rollouts_A_200.npz \\
        --out data/recap_exp/advantages_A.npz \\
        --stats_out data/recap_exp/advantage_stats_A.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RECAP advantages and binary indicators (no simulator).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Pretrained DiffusionPolicy checkpoint (for encoder).")
    parser.add_argument("--vf_ckpt", type=str, required=True,
                        help="Value function checkpoint (.pt) from step 2.")
    parser.add_argument("--npz_paths", type=str, nargs="+", required=True,
                        help="Input NPZ files (demos + rollouts).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output NPZ path (episodes with 'indicators' field).")
    parser.add_argument("--stats_out", type=str, default=None,
                        help="Optional JSON path for advantage statistics.")

    parser.add_argument("--percentile", type=float, default=30.0,
                        help="Advantage percentile for threshold (~50%% success → 30).")
    parser.add_argument("--c_fail", type=float, default=1200.0)
    parser.add_argument("--max_ep_len", type=float, default=1200.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import (
        load_episodes_from_npz_list,
        load_vf_checkpoint,
        compute_advantages_and_indicators,
        compute_normalized_returns,
        save_episodes_npz,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"RECAP — Advantage Computation")
    print(f"  Policy:     {args.policy}")
    print(f"  VF ckpt:    {args.vf_ckpt}")
    print(f"  Percentile: {args.percentile}")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n")

    # ---- Load data ----
    episodes = load_episodes_from_npz_list(args.npz_paths)
    n_suc  = sum(1 for ep in episodes if ep["success"])
    n_fail = len(episodes) - n_suc
    total_frames = sum(len(ep["action"]) for ep in episodes)
    print(f"  {len(episodes)} episodes | {n_suc} success / {n_fail} failure | "
          f"{total_frames} total frames")

    # ---- Load value function ----
    print("\nLoading value function...")
    vf_model = load_vf_checkpoint(
        pretrained_dir=args.policy,
        ckpt_path=args.vf_ckpt,
        device=device,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    vf_model.eval()

    # ---- Compute advantages and indicators ----
    start = time.time()
    threshold = compute_advantages_and_indicators(
        episodes=episodes,
        vf_model=vf_model,
        device=device,
        percentile=args.percentile,
        c_fail=args.c_fail,
        max_len=args.max_ep_len,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start

    # ---- Statistics ----
    all_indicators = [I for ep in episodes for I in ep["indicators"]]
    n_pos = sum(all_indicators)
    n_neg = len(all_indicators) - n_pos
    pos_ratio = n_pos / len(all_indicators) if all_indicators else 0.0

    print(f"\n  Frames:   total={len(all_indicators)}"
          f"  positive={n_pos} ({pos_ratio:.1%})"
          f"  negative={n_neg} ({1-pos_ratio:.1%})")
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Time: {elapsed:.1f}s")

    # Per-episode breakdown
    ep_summaries = []
    for ep_idx, ep in enumerate(episodes):
        n_ep_frames = len(ep["indicators"])
        n_ep_pos    = sum(ep["indicators"])
        ep_summaries.append({
            "episode": ep_idx,
            "success": bool(ep["success"]),
            "n_frames": n_ep_frames,
            "n_positive": n_ep_pos,
            "positive_ratio": n_ep_pos / n_ep_frames if n_ep_frames > 0 else 0.0,
        })

    suc_pos  = np.mean([s["positive_ratio"] for s in ep_summaries if s["success"]])
    fail_pos = np.mean([s["positive_ratio"] for s in ep_summaries if not s["success"]])
    print(f"\n  Positive ratio in SUCCESS episodes: {suc_pos:.1%}")
    print(f"  Positive ratio in FAILURE episodes: {fail_pos:.1%}")

    # ---- Save ----
    save_episodes_npz(args.out, episodes)

    if args.stats_out:
        stats = {
            "threshold": threshold,
            "percentile": args.percentile,
            "total_frames": len(all_indicators),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_ratio": pos_ratio,
            "success_ep_positive_ratio": float(suc_pos),
            "failure_ep_positive_ratio": float(fail_pos),
            "c_fail": args.c_fail,
            "max_ep_len": args.max_ep_len,
            "per_episode": ep_summaries,
        }
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved to: {stats_path}")

    print(f"\nDone. Augmented NPZ: {args.out}")


if __name__ == "__main__":
    main()
