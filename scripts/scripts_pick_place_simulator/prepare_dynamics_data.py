#!/usr/bin/env python3
"""Extract transition pairs (obs[t], action[t], obs[t+1]) from encoded episodes for dynamics model training.

Performs episode-level train/val split (80/20) to avoid data leakage.

Usage:
    python scripts/scripts_pick_place_simulator/prepare_dynamics_data.py \
        --input_npz data/.../iter1_collect_A_encoded.npz \
        --output_dir data/.../iter1_dynamics_data/ \
        --train_ratio 0.8 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract transition pairs for dynamics model training.")
    parser.add_argument("--input_npz", type=str, required=True,
                        help="Input encoded .npz (episodes with obs_latent).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return parser.parse_args()


def extract_transitions(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (obs[t], action[t], obs[t+1]) from all episodes.

    Returns:
        obs: (N, obs_dim) float32
        action: (N, action_dim) float32
        obs_next: (N, obs_dim) float32
    """
    obs_list, action_list, obs_next_list = [], [], []

    for ep in episodes:
        obs_latent = ep["obs_latent"]  # (T, 143)
        action = ep["action"]          # (T, 8)
        T = len(obs_latent)

        # Each episode yields T-1 transitions
        obs_list.append(obs_latent[:-1])
        action_list.append(action[:-1])
        obs_next_list.append(obs_latent[1:])

    return (
        np.concatenate(obs_list, axis=0).astype(np.float32),
        np.concatenate(action_list, axis=0).astype(np.float32),
        np.concatenate(obs_next_list, axis=0).astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Load encoded episodes
    print(f"Loading encoded episodes from: {args.input_npz}")
    with np.load(args.input_npz, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"  Loaded {len(episodes)} episodes")

    # Episode-level split
    rng = np.random.default_rng(args.seed)
    n_episodes = len(episodes)
    indices = rng.permutation(n_episodes)
    n_train = int(n_episodes * args.train_ratio)

    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]

    print(f"  Split: {len(train_episodes)} train / {len(val_episodes)} val episodes")

    # Extract transitions
    train_obs, train_action, train_obs_next = extract_transitions(train_episodes)
    val_obs, val_action, val_obs_next = extract_transitions(val_episodes)

    print(f"  Train transitions: {len(train_obs)}")
    print(f"  Val transitions:   {len(val_obs)}")
    print(f"  Obs dim: {train_obs.shape[1]}, Action dim: {train_action.shape[1]}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out_dir / "train.npz"),
        obs=train_obs, action=train_action, obs_next=train_obs_next,
    )
    np.savez_compressed(
        str(out_dir / "val.npz"),
        obs=val_obs, action=val_action, obs_next=val_obs_next,
    )

    # Save split info
    split_info = {
        "n_episodes_total": n_episodes,
        "n_episodes_train": len(train_episodes),
        "n_episodes_val": len(val_episodes),
        "n_transitions_train": int(len(train_obs)),
        "n_transitions_val": int(len(val_obs)),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "obs_dim": int(train_obs.shape[1]),
        "action_dim": int(train_action.shape[1]),
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
