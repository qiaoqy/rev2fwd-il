#!/usr/bin/env python3
"""Advantage indicator labeling for RECAP CFG-RL (Exp56).

Pipeline:
1. Load demo NPZ and cumulative NPZ (demo prefix + rollout episodes)
2. Strip the demo prefix from the cumulative input
3. Critic inference → per-frame V(s) on rollout success episodes only
4. Advantage A(t) = step_reward + V(t+1) − V(t)  (gamma=1, lambda=0 → 1-step TD)
5. Percentile only on rollout success frames (exclude demo)
6. Top 30% → indicator=+1, Bottom 70% → indicator=-1
7. Demo frames: all indicator=+1
8. Output: labeled NPZ with 'indicator' (T,) per episode

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/script_recap/label_advantage.py \
        --rollout_npz iter1_cumulative_A.npz \
        --demo_npz task_A_demo.npz \
        --checkpoint iter1_critic/checkpoints/best/checkpoint.pt \
        --gamma 1.0 --lam 0 \
        --step_reward -0.000167 \
        --value_truncate -0.005 \
        --positive_percentile 0.3 \
        --out iter1_labeled.npz \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from debug.eval_critic_visual import (
    load_critic_model,
    build_state,
    predict_episode_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advantage indicator labeling for RECAP CFG-RL.",
    )
    parser.add_argument("--rollout_npz", type=str, required=True,
                        help="Cumulative NPZ built as demo + rollout episodes. "
                            "The demo prefix is excluded before percentile computation.")
    parser.add_argument("--demo_npz", type=str, required=True,
                        help="Demo NPZ (all marked success). Demo episodes get indicator=+1.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Critic checkpoint path.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor for advantage (1.0 = undiscounted).")
    parser.add_argument("--lam", type=float, default=0.0,
                        help="GAE lambda (0 = 1-step TD advantage).")
    parser.add_argument("--step_reward", type=float, default=-1.0 / (2 * 3000),
                        help="Per-step reward for advantage computation.")
    parser.add_argument("--value_truncate", type=float, default=-0.005,
                        help="Truncate predicted values below this to this value.")
    parser.add_argument("--positive_percentile", type=float, default=0.3,
                        help="Top fraction of advantages → indicator=+1 (default 0.3 = 30%).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output labeled NPZ path.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for critic inference.")
    return parser.parse_args()


def compute_1step_td_advantage(
    values: np.ndarray,
    step_reward: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute 1-step TD advantage: A(t) = r(t) + gamma * V(t+1) - V(t).

    For t = T-1 (last step), V(T) is approximated as V(T-1) (terminal bootstrap).

    Args:
        values: (T,) predicted values.
        step_reward: Per-step reward r(t).
        gamma: Discount factor.

    Returns:
        advantages: (T,) per-frame advantage.
    """
    T = len(values)
    advantages = np.zeros(T, dtype=np.float32)

    for t in range(T):
        if t == T - 1:
            next_val = values[t]  # terminal bootstrap
        else:
            next_val = values[t + 1]
        advantages[t] = step_reward + gamma * next_val - values[t]

    return advantages


def compute_gae_advantage(
    values: np.ndarray,
    step_reward: float,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> np.ndarray:
    """Compute GAE advantage.

    Args:
        values: (T,) predicted values.
        step_reward: Per-step reward r(t).
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        advantages: (T,) per-frame GAE advantage.
    """
    T = len(values)
    rewards = np.full(T, step_reward, dtype=np.float32)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            next_value = values[t]  # terminal bootstrap
        else:
            next_value = values[t + 1]
        delta_t = rewards[t] + gamma * next_value - values[t]
        last_gae = delta_t + gamma * lam * last_gae
        advantages[t] = last_gae

    return advantages


def _episodes_match(lhs: dict, rhs: dict) -> bool:
    """Return True when two episodes are the same demo episode.

    The cumulative file is expected to start with an exact copy of the demo NPZ.
    Compare a few stable arrays to guard against accidentally stripping the wrong
    prefix when the cumulative merge order changes.
    """
    keys_to_compare = ("action", "ee_pose", "obj_pose")
    for key in keys_to_compare:
        lhs_has = key in lhs
        rhs_has = key in rhs
        if lhs_has != rhs_has:
            return False
        if lhs_has:
            if lhs[key].shape != rhs[key].shape:
                return False
            if not np.array_equal(lhs[key], rhs[key]):
                return False
    return bool(lhs.get("success", False)) == bool(rhs.get("success", False))


def split_demo_prefix_from_cumulative(
    cumulative_eps: list[dict],
    demo_eps: list[dict],
) -> list[dict]:
    """Split a cumulative dataset into rollout episodes by removing the demo prefix."""
    demo_count = len(demo_eps)
    if demo_count == 0:
        return cumulative_eps

    if len(cumulative_eps) < demo_count:
        raise ValueError(
            "Cumulative input has fewer episodes than demo input: "
            f"cumulative={len(cumulative_eps)}, demo={demo_count}"
        )

    prefix_eps = cumulative_eps[:demo_count]
    for idx, (prefix_ep, demo_ep) in enumerate(zip(prefix_eps, demo_eps, strict=True)):
        if not _episodes_match(prefix_ep, demo_ep):
            raise ValueError(
                "Cumulative input does not start with the demo prefix expected by "
                "Exp56 labeling. "
                f"Mismatch at demo episode {idx}."
            )

    return cumulative_eps[demo_count:]


def main() -> None:
    args = parse_args()

    device = args.device

    # ---- Load critic model ----
    print(f"Loading critic from {args.checkpoint}")
    critic = load_critic_model(args.checkpoint, device)

    # ---- Load demo data ----
    print(f"Loading demo data from {args.demo_npz}")
    with np.load(args.demo_npz, allow_pickle=True) as data:
        demo_eps = list(data["episodes"])
    print(f"  Demo episodes: {len(demo_eps)}")

    # ---- Load cumulative input and strip demo prefix ----
    print(f"Loading cumulative data from {args.rollout_npz}")
    with np.load(args.rollout_npz, allow_pickle=True) as data:
        cumulative_eps = list(data["episodes"])

    rollout_eps = split_demo_prefix_from_cumulative(cumulative_eps, demo_eps)
    rollout_success_eps = [ep for ep in rollout_eps if ep.get("success", False)]
    rollout_fail_eps = [ep for ep in rollout_eps if not ep.get("success", False)]
    print(f"  Cumulative episodes: {len(cumulative_eps)}")
    print(f"  Rollout-only episodes: {len(rollout_eps)}")
    print(f"    Success: {len(rollout_success_eps)}, Failure: {len(rollout_fail_eps)}")

    # ---- Critic inference on rollout success episodes ----
    print("Running critic inference on rollout success episodes...")
    all_advantages = []
    all_values = []

    for i, ep in enumerate(rollout_success_eps):
        T = len(ep["action"])
        # Predict V(t) for each frame
        pred_values = predict_episode_values(
            critic, ep, horizon=T,
            n_obs_steps=2, device=device,
            batch_size=args.batch_size,
        )

        # Truncate values
        if args.value_truncate is not None:
            pred_values = np.clip(pred_values, args.value_truncate, None)

        # Compute advantage
        if args.lam == 0:
            advantages = compute_1step_td_advantage(
                pred_values, args.step_reward, args.gamma,
            )
        else:
            advantages = compute_gae_advantage(
                pred_values, args.step_reward, args.gamma, args.lam,
            )

        all_advantages.append(advantages)
        all_values.append(pred_values)

        if (i + 1) % 5 == 0 or i == len(rollout_success_eps) - 1:
            print(f"  [{i+1}/{len(rollout_success_eps)}] T={T}, "
                  f"V=[{pred_values.min():.4f}, {pred_values.max():.4f}], "
                  f"A=[{advantages.min():.6f}, {advantages.max():.6f}]")

    # ---- Compute percentile threshold on rollout success frames only ----
    if all_advantages:
        all_adv_flat = np.concatenate(all_advantages)
        # Top positive_percentile → +1, rest → -1
        # np.percentile(x, 70) gives the value below which 70% of data falls
        # So frames with A >= percentile_70 are the top 30%
        threshold_percentile = (1.0 - args.positive_percentile) * 100.0
        threshold = np.percentile(all_adv_flat, threshold_percentile)
        n_positive = np.sum(all_adv_flat >= threshold)
        n_total = len(all_adv_flat)
        print(f"\nAdvantage statistics (rollout success frames only):")
        print(f"  Total frames: {n_total}")
        print(f"  Advantage range: [{all_adv_flat.min():.6f}, {all_adv_flat.max():.6f}]")
        print(f"  Mean: {all_adv_flat.mean():.6f}, Std: {all_adv_flat.std():.6f}")
        print(f"  Threshold (p{threshold_percentile:.0f}): {threshold:.6f}")
        print(f"  Positive (+1): {n_positive}/{n_total} = {n_positive/n_total:.1%}")
        print(f"  Negative (-1): {n_total - n_positive}/{n_total} = {(n_total - n_positive)/n_total:.1%}")
    else:
        threshold = 0.0
        print("\nWARNING: No rollout success episodes! All rollout frames will be skipped.")

    # ---- Label rollout success episodes ----
    labeled_rollout_eps = []
    for i, (ep, advantages) in enumerate(zip(rollout_success_eps, all_advantages)):
        indicator = np.where(advantages >= threshold, 1.0, -1.0).astype(np.float32)
        ep_labeled = dict(ep)
        ep_labeled["indicator"] = indicator
        ep_labeled["_advantage"] = advantages  # store for debugging
        ep_labeled["_pred_values"] = all_values[i]
        labeled_rollout_eps.append(ep_labeled)

    # ---- Label demo episodes: all indicator=+1 ----
    labeled_demo_eps = []
    for ep in demo_eps:
        T = len(ep["action"])
        ep_labeled = dict(ep)
        ep_labeled["indicator"] = np.ones(T, dtype=np.float32)
        ep_labeled["_is_demo"] = True
        labeled_demo_eps.append(ep_labeled)

    # ---- Merge: demo + labeled rollout success ----
    all_labeled = labeled_demo_eps + labeled_rollout_eps

    # ---- Save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        episodes=np.array(all_labeled, dtype=object),
    )

    n_demo = len(labeled_demo_eps)
    n_rollout = len(labeled_rollout_eps)
    total_demo_frames = sum(len(ep["action"]) for ep in labeled_demo_eps)
    total_rollout_frames = sum(len(ep["action"]) for ep in labeled_rollout_eps)
    total_pos_rollout = sum(
        np.sum(ep["indicator"] > 0) for ep in labeled_rollout_eps
    ) if labeled_rollout_eps else 0
    total_neg_rollout = sum(
        np.sum(ep["indicator"] < 0) for ep in labeled_rollout_eps
    ) if labeled_rollout_eps else 0

    print(f"\nSaved labeled data to {out_path}")
    print(f"  Demo episodes:    {n_demo} ({total_demo_frames} frames, all +1)")
    print(f"  Rollout episodes: {n_rollout} ({total_rollout_frames} frames)")
    print(f"    Positive (+1):  {total_pos_rollout} frames")
    print(f"    Negative (-1):  {total_neg_rollout} frames")
    print(f"  Total:            {n_demo + n_rollout} episodes, "
          f"{total_demo_frames + total_rollout_frames} frames")

    # ---- Save statistics JSON ----
    stats = {
        "config": vars(args),
        "cumulative_total_episodes": len(cumulative_eps),
        "rollout_total_episodes": len(rollout_eps),
        "rollout_success_episodes": len(rollout_success_eps),
        "rollout_failure_episodes": len(rollout_fail_eps),
        "demo_episodes": n_demo,
        "advantage_threshold": float(threshold),
        "positive_percentile": args.positive_percentile,
        "total_rollout_frames": int(total_rollout_frames),
        "positive_rollout_frames": int(total_pos_rollout),
        "negative_rollout_frames": int(total_neg_rollout),
        "total_demo_frames": int(total_demo_frames),
        "output_episodes": n_demo + n_rollout,
    }
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics: {stats_path}")


if __name__ == "__main__":
    main()
