#!/usr/bin/env python3
"""Prepare critic training data from rollout episodes.

Supports two input modes:
1. Load rollout partitions from `data_dir` + `prefix`
2. Load a single merged NPZ via `--input_npz`

In both modes the script computes MC returns, performs an 80/20 train/val
split, and saves labeled NPZ files.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rev2fwd_il.data.value_labeling import compute_mc_returns


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare critic data with MC returns")
    parser.add_argument("--input_npz", type=str, default=None,
                        help="Optional merged rollout NPZ file. If provided, load episodes from this file instead of partition files.")
    parser.add_argument("--data_dir", type=str, default="debug/data",
                        help="Directory containing rollout partition files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save critic_{task}_{train,val,split}. Defaults to data_dir.")
    parser.add_argument("--task", type=str, default="A", choices=["A", "B"],
                        help="Task to process (A or B)")
    parser.add_argument("--max_episode_length", type=int, default=3000,
                        help="R_success normalization constant for MC returns")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of episodes for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    parser.add_argument("--prefix", type=str, default="iter3_rollout",
                        help="Filename prefix for partition files")
    return parser.parse_args()


def load_episodes_from_npz(input_path: Path) -> list[dict]:
    """Load episodes from a merged NPZ file."""
    with np.load(input_path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"  Loaded merged file {input_path}: {len(episodes)} episodes")
    return episodes


def load_all_partitions(data_dir: Path, task: str, prefix: str) -> list[dict]:
    """Load all partition files for a given task and merge episodes."""
    all_episodes = []
    partition_idx = 0
    while True:
        path = data_dir / f"{prefix}_{task}_p{partition_idx}.npz"
        if not path.exists():
            break
        with np.load(path, allow_pickle=True) as data:
            episodes = list(data["episodes"])
        print(f"  Loaded {path.name}: {len(episodes)} episodes")
        # Tag each episode with its source partition
        for ep in episodes:
            ep["_source_partition"] = partition_idx
        all_episodes.extend(episodes)
        partition_idx += 1

    if partition_idx == 0:
        raise FileNotFoundError(
            f"No partition files found matching {data_dir}/{prefix}_{task}_p*.npz"
        )
    print(f"  Total: {len(all_episodes)} episodes from {partition_idx} partitions")
    return all_episodes


def add_mc_values(episodes: list[dict], max_episode_length: int) -> None:
    """Compute and attach mc_value to each episode dict in-place."""
    values = compute_mc_returns(episodes, max_episode_length=max_episode_length)
    for ep, v in zip(episodes, values):
        ep["mc_value"] = v  # (T,) float32


def split_episodes(
    episodes: list[dict], train_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split episodes into train/val sets with stratified sampling by success.

    Ensures both train and val sets contain both successful and failed episodes
    (when available) at approximately the requested ratio.
    """
    rng = np.random.RandomState(seed)

    success_eps = [ep for ep in episodes if ep.get("success", False)]
    failure_eps = [ep for ep in episodes if not ep.get("success", False)]

    rng.shuffle(success_eps)
    rng.shuffle(failure_eps)

    n_train_success = max(1, int(len(success_eps) * train_ratio))
    n_train_failure = max(0, int(len(failure_eps) * train_ratio))

    # Handle edge case: if only 1 failure episode, put it in train
    if len(failure_eps) == 1:
        n_train_failure = 1

    train_eps = success_eps[:n_train_success] + failure_eps[:n_train_failure]
    val_eps = success_eps[n_train_success:] + failure_eps[n_train_failure:]

    # Shuffle within each set
    rng.shuffle(train_eps)
    rng.shuffle(val_eps)

    return train_eps, val_eps


def save_labeled_dataset(episodes: list[dict], output_path: Path) -> None:
    """Save episodes with mc_value as compressed NPZ."""
    # Remove internal tags before saving
    clean_eps = []
    for ep in episodes:
        clean_ep = {k: v for k, v in ep.items() if not k.startswith("_")}
        clean_eps.append(clean_ep)

    np.savez_compressed(str(output_path), episodes=np.array(clean_eps, dtype=object))
    print(f"  Saved {len(clean_eps)} episodes to {output_path}")


def compute_split_stats(episodes: list[dict], label: str) -> dict:
    """Compute summary statistics for a set of episodes."""
    n_total = len(episodes)
    n_success = sum(1 for ep in episodes if ep.get("success", False))
    n_failure = n_total - n_success

    lengths = [len(ep["action"]) for ep in episodes]
    total_frames = sum(lengths)

    # MC value stats for successful episodes
    mv_stats = {}
    success_eps = [ep for ep in episodes if ep.get("success", False)]
    if success_eps:
        all_mv = np.concatenate([ep["mc_value"] for ep in success_eps])
        mv_stats = {
            "mc_value_min": float(np.min(all_mv)),
            "mc_value_max": float(np.max(all_mv)),
            "mc_value_mean": float(np.mean(all_mv)),
        }

    return {
        "label": label,
        "num_episodes": n_total,
        "num_success": n_success,
        "num_failure": n_failure,
        "success_rate": n_success / n_total if n_total > 0 else 0,
        "total_frames": total_frames,
        "episode_length_min": min(lengths) if lengths else 0,
        "episode_length_max": max(lengths) if lengths else 0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0,
        **mv_stats,
    }


def verify_mc_returns(episodes: list[dict], max_episode_length: int) -> None:
    """Run verification checks on MC return labels.

    With [-1, 0) normalization:
        Success: V_bar(t) = -remaining / (2R) ∈ (-0.5, 0)
        Failure: V_bar(t) = -0.5 - remaining / (2R) ∈ (-1, -0.5)
        Boundary: -0.5
    """
    print("\n  Verification checks:")
    all_ok = True
    R = max_episode_length

    for i, ep in enumerate(episodes):
        mv = ep["mc_value"]
        T = len(mv)
        success = ep.get("success", False)

        if not success:
            # Failure: all values should be < -0.5
            if np.any(mv >= -0.5 + 1e-6):
                print(f"    FAIL ep{i}: failure episode has mc values >= -0.5")
                all_ok = False
            # V(T-1) should be ≈ -0.5 - 1/(2R)
            expected_last = -0.5 - 1.0 / (2 * R)
            if not np.isclose(mv[-1], expected_last, atol=1e-6):
                print(f"    FAIL ep{i}: V(T-1)={mv[-1]:.6f}, expected {expected_last:.6f}")
                all_ok = False
            continue

        # Success: all values should be in (-0.5, 0)
        if np.any(mv >= 0 + 1e-6):
            print(f"    FAIL ep{i}: success episode has mc values >= 0")
            all_ok = False
        if np.any(mv < -0.5 - 1e-6):
            print(f"    FAIL ep{i}: success episode has mc values < -0.5")
            all_ok = False

        # Values should be monotonically non-decreasing
        diffs = np.diff(mv)
        if np.any(diffs < -1e-6):
            print(f"    FAIL ep{i}: non-monotonic mc values")
            all_ok = False

        # V(T-1) = -1/(2R) ≈ 0
        expected_last = -1.0 / (2 * R)
        if not np.isclose(mv[-1], expected_last, rtol=1e-4):
            print(f"    FAIL ep{i}: V(T-1)={mv[-1]:.6f}, expected {expected_last:.6f}")
            all_ok = False

    status = "PASSED" if all_ok else "FAILED"
    print(f"    All checks: {status}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else data_dir

    print(f"=" * 60)
    print(f"Preparing critic data for Task {args.task}")
    print(f"  data_dir: {data_dir}")
    if args.input_npz is not None:
        print(f"  input_npz: {args.input_npz}")
    print(f"  output_dir: {output_dir}")
    print(f"  max_episode_length: {args.max_episode_length}  train_ratio: {args.train_ratio}  seed: {args.seed}")
    print(f"=" * 60)

    # Step 1: Load all partition files
    print(f"\n[1/4] Loading partitions...")
    if args.input_npz is not None:
        episodes = load_episodes_from_npz(Path(args.input_npz))
    else:
        episodes = load_all_partitions(data_dir, args.task, args.prefix)

    # Step 2: Compute MC returns for all episodes
    print(f"\n[2/4] Computing MC returns (R_success={args.max_episode_length})...")
    add_mc_values(episodes, args.max_episode_length)

    # Verify on all episodes
    verify_mc_returns(episodes, args.max_episode_length)

    # Step 3: Split into train/val
    print(f"\n[3/4] Splitting {len(episodes)} episodes ({args.train_ratio:.0%} train)...")
    train_eps, val_eps = split_episodes(episodes, args.train_ratio, args.seed)

    train_stats = compute_split_stats(train_eps, "train")
    val_stats = compute_split_stats(val_eps, "val")

    print(f"  Train: {train_stats['num_episodes']} eps "
          f"({train_stats['num_success']} success, {train_stats['num_failure']} failure, "
          f"{train_stats['total_frames']} frames)")
    print(f"  Val:   {val_stats['num_episodes']} eps "
          f"({val_stats['num_success']} success, {val_stats['num_failure']} failure, "
          f"{val_stats['total_frames']} frames)")

    # Step 4: Save labeled datasets
    print(f"\n[4/4] Saving labeled datasets...")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"critic_{args.task}_train.npz"
    val_path = output_dir / f"critic_{args.task}_val.npz"
    save_labeled_dataset(train_eps, train_path)
    save_labeled_dataset(val_eps, val_path)

    # Save split metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "train": train_stats,
        "val": val_stats,
    }
    meta_path = output_dir / f"critic_{args.task}_split.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"Done! Output files:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {meta_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
