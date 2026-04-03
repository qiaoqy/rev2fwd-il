#!/usr/bin/env python3
"""Phase 2: Prepare critic training data from rollout episodes.

Loads rollout data from partition files, computes Bellman returns,
splits into 80/20 train/test sets, and saves labeled NPZ files.

Usage:
    python debug/prepare_critic_data.py \
        --data_dir debug/data \
        --task A \
        --gamma 0.99 \
        --train_ratio 0.8

Output:
    debug/data/critic_A_train.npz   (80% episodes with bellman_value)
    debug/data/critic_A_test.npz    (20% episodes with bellman_value)
    debug/data/critic_A_split.json  (split metadata)
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rev2fwd_il.data.value_labeling import compute_bellman_returns


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare critic data with Bellman returns")
    parser.add_argument("--data_dir", type=str, default="debug/data",
                        help="Directory containing rollout partition files")
    parser.add_argument("--task", type=str, default="A", choices=["A", "B"],
                        help="Task to process (A or B)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for Bellman returns")
    parser.add_argument("--success_reward", type=float, default=1.0,
                        help="Reward value at success timestep")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of episodes for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    parser.add_argument("--prefix", type=str, default="iter3_rollout",
                        help="Filename prefix for partition files")
    return parser.parse_args()


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


def add_bellman_values(episodes: list[dict], gamma: float, success_reward: float) -> None:
    """Compute and attach bellman_value to each episode dict in-place."""
    values = compute_bellman_returns(episodes, gamma=gamma, success_reward=success_reward)
    for ep, v in zip(episodes, values):
        ep["bellman_value"] = v  # (T,) float32


def split_episodes(
    episodes: list[dict], train_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split episodes into train/test sets with stratified sampling by success.

    Ensures both train and test sets contain both successful and failed episodes
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
    test_eps = success_eps[n_train_success:] + failure_eps[n_train_failure:]

    # Shuffle within each set
    rng.shuffle(train_eps)
    rng.shuffle(test_eps)

    return train_eps, test_eps


def save_labeled_dataset(episodes: list[dict], output_path: Path) -> None:
    """Save episodes with bellman_value as compressed NPZ."""
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

    # Bellman value stats for successful episodes
    bv_stats = {}
    success_eps = [ep for ep in episodes if ep.get("success", False)]
    if success_eps:
        all_bv = np.concatenate([ep["bellman_value"] for ep in success_eps])
        bv_stats = {
            "bellman_value_min": float(np.min(all_bv)),
            "bellman_value_max": float(np.max(all_bv)),
            "bellman_value_mean": float(np.mean(all_bv)),
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
        **bv_stats,
    }


def verify_bellman_returns(episodes: list[dict], gamma: float) -> None:
    """Run verification checks on Bellman return labels."""
    print("\n  Verification checks:")
    all_ok = True

    for i, ep in enumerate(episodes):
        bv = ep["bellman_value"]
        T = len(bv)
        success = ep.get("success", False)
        success_step = ep.get("success_step", None)

        if not success:
            # Failed episodes should have all-zero values
            if not np.allclose(bv, 0.0):
                print(f"    FAIL ep{i}: failure episode has non-zero bellman values")
                all_ok = False
            continue

        # Successful episodes
        if success_step is not None:
            ss = min(success_step, T - 1)
            # Value at success_step should be 1.0
            if not np.isclose(bv[ss], 1.0, atol=1e-6):
                print(f"    FAIL ep{i}: V(success_step={ss}) = {bv[ss]:.6f}, expected 1.0")
                all_ok = False

            # Values before success_step should be monotonically non-decreasing
            pre_values = bv[:ss + 1]
            diffs = np.diff(pre_values)
            if np.any(diffs < -1e-6):
                print(f"    FAIL ep{i}: non-monotonic values before success_step")
                all_ok = False

            # Values after success_step should be 0
            if ss < T - 1:
                post_values = bv[ss + 1:]
                if not np.allclose(post_values, 0.0):
                    print(f"    FAIL ep{i}: non-zero values after success_step")
                    all_ok = False

            # Check V(0) ≈ gamma^ss
            expected_v0 = gamma ** ss
            if not np.isclose(bv[0], expected_v0, rtol=1e-4):
                print(f"    FAIL ep{i}: V(0)={bv[0]:.6f}, expected gamma^{ss}={expected_v0:.6f}")
                all_ok = False

    status = "PASSED" if all_ok else "FAILED"
    print(f"    All checks: {status}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"=" * 60)
    print(f"Preparing critic data for Task {args.task}")
    print(f"  data_dir: {data_dir}")
    print(f"  gamma: {args.gamma}  train_ratio: {args.train_ratio}  seed: {args.seed}")
    print(f"=" * 60)

    # Step 1: Load all partition files
    print(f"\n[1/4] Loading partitions...")
    episodes = load_all_partitions(data_dir, args.task, args.prefix)

    # Step 2: Compute Bellman returns for all episodes
    print(f"\n[2/4] Computing Bellman returns (gamma={args.gamma})...")
    add_bellman_values(episodes, args.gamma, args.success_reward)

    # Verify on all episodes
    verify_bellman_returns(episodes, args.gamma)

    # Step 3: Split into train/test
    print(f"\n[3/4] Splitting {len(episodes)} episodes ({args.train_ratio:.0%} train)...")
    train_eps, test_eps = split_episodes(episodes, args.train_ratio, args.seed)

    train_stats = compute_split_stats(train_eps, "train")
    test_stats = compute_split_stats(test_eps, "test")

    print(f"  Train: {train_stats['num_episodes']} eps "
          f"({train_stats['num_success']} success, {train_stats['num_failure']} failure, "
          f"{train_stats['total_frames']} frames)")
    print(f"  Test:  {test_stats['num_episodes']} eps "
          f"({test_stats['num_success']} success, {test_stats['num_failure']} failure, "
          f"{test_stats['total_frames']} frames)")

    # Step 4: Save labeled datasets
    print(f"\n[4/4] Saving labeled datasets...")
    train_path = data_dir / f"critic_{args.task}_train.npz"
    test_path = data_dir / f"critic_{args.task}_test.npz"
    save_labeled_dataset(train_eps, train_path)
    save_labeled_dataset(test_eps, test_path)

    # Save split metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "train": train_stats,
        "test": test_stats,
    }
    meta_path = data_dir / f"critic_{args.task}_split.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata to {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"Done! Output files:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"  {meta_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
