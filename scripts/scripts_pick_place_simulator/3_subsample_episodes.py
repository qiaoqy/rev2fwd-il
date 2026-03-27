#!/usr/bin/env python3
"""Subsample episodes to a target number of frames via uniform temporal downsampling.

For each episode with T frames, if T > target_frames, pick target_frames evenly
spaced indices via np.linspace and keep only those frames.  After subsampling,
recompute actions so that action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t].

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/3_subsample_episodes.py \
        --input data/exp22/iter1_collect_B.npz \
        --out data/exp22/iter1_collect_B_subsampled.npz \
        --target_frames 400
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniformly subsample episodes to a target frame count.",
    )
    parser.add_argument("--input", type=str, required=True, help="Input NPZ file.")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ file.")
    parser.add_argument("--target_frames", type=int, default=400,
                        help="Target number of frames per episode (default: 400).")
    parser.add_argument("--success_only", type=int, default=1, choices=[0, 1],
                        help="Only keep successful episodes. Default: 1.")
    return parser.parse_args()


def subsample_episode(ep: dict, target_frames: int) -> dict:
    """Uniformly subsample an episode to target_frames, then recompute actions."""
    T = len(ep["images"])

    if T <= target_frames:
        # No subsampling needed; still recompute actions for consistency
        result = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in ep.items()}
        _recompute_actions(result)
        return result

    # Uniform indices
    indices = np.round(np.linspace(0, T - 1, target_frames)).astype(int)

    # Subsample all time-indexed arrays
    TEMPORAL_KEYS = ["images", "ee_pose", "obj_pose", "action", "obs",
                     "gripper", "fsm_state", "wrist_images"]

    result = {}
    for key, val in ep.items():
        if key in TEMPORAL_KEYS and hasattr(val, "__len__") and len(val) == T:
            result[key] = val[indices].copy()
        elif hasattr(val, "copy"):
            result[key] = val.copy()
        else:
            result[key] = val

    _recompute_actions(result)
    return result


def _recompute_actions(ep: dict) -> None:
    """Recompute action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]."""
    ee = ep["ee_pose"]
    T = len(ee)
    actions = np.zeros((T, 8), dtype=np.float32)
    actions[:T - 1, :7] = ee[1:]
    actions[T - 1, :7] = ee[T - 1]

    if "gripper" in ep:
        actions[:, 7] = ep["gripper"].flatten()[:T]
    elif "action" in ep:
        actions[:, 7] = ep["action"][:, 7]

    ep["action"] = actions


def _compute_avg_step_size(ee_pose: np.ndarray) -> float:
    """Average Euclidean step size in position (first 3 dims of ee_pose)."""
    if len(ee_pose) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(ee_pose[:, :3], axis=0), axis=1).mean())


def main() -> None:
    args = _parse_args()
    t0 = datetime.datetime.now()
    print(f"[{t0.strftime('%Y-%m-%d %H:%M:%S')}] Subsample start")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.out}")
    print(f"  Target: {args.target_frames} frames/episode")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return
    input_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"  Input size: {input_mb:.1f} MB")

    with np.load(args.input, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes from {args.input}")

    if args.success_only:
        n_before = len(episodes)
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Filtered to {len(episodes)} successful episodes (dropped {n_before - len(episodes)})")

    if not episodes:
        print("ERROR: No episodes to process!")
        return

    subsampled = []
    total_orig = 0
    total_new = 0
    orig_step_sizes = []
    sub_step_sizes = []

    for i, ep in enumerate(episodes):
        orig_len = len(ep["images"])
        orig_step = _compute_avg_step_size(ep["ee_pose"])

        sub_ep = subsample_episode(ep, args.target_frames)
        new_len = len(sub_ep["images"])
        sub_step = _compute_avg_step_size(sub_ep["ee_pose"])

        subsampled.append(sub_ep)
        total_orig += orig_len
        total_new += new_len
        orig_step_sizes.append(orig_step)
        sub_step_sizes.append(sub_step)

        amp = sub_step / orig_step if orig_step > 0 else 0.0
        if i < 5 or (i + 1) % 10 == 0 or i == len(episodes) - 1:
            print(f"  ep {i:3d}: {orig_len:5d} → {new_len:4d} frames  "
                  f"step_size {orig_step:.6f} → {sub_step:.6f}  ({amp:.2f}x)")

    avg_orig = total_orig / len(episodes)
    avg_new = total_new / len(subsampled)
    avg_orig_step = np.mean(orig_step_sizes)
    avg_sub_step = np.mean(sub_step_sizes)
    avg_amp = avg_sub_step / avg_orig_step if avg_orig_step > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"Subsample Statistics")
    print(f"{'=' * 70}")
    print(f"Total episodes:           {len(subsampled)}")
    print(f"Target frames:            {args.target_frames}")
    print(f"Original avg length:      {avg_orig:.1f}")
    print(f"Subsampled avg length:    {avg_new:.1f}")
    print(f"Frame compression ratio:  {avg_new / avg_orig:.2f}")
    print(f"Original avg step size:   {avg_orig_step:.6f}")
    print(f"Subsampled avg step size: {avg_sub_step:.6f}")
    print(f"Step-size amplification:  {avg_amp:.2f}x")
    print(f"{'=' * 70}\n")

    # Verify actions on ALL episodes
    print("Verifying action[t][:7] == ee_pose[t+1] for ALL episodes...")
    all_ok = True
    for i, ep in enumerate(subsampled):
        ee = ep["ee_pose"]
        act = ep["action"]
        T = len(ee)
        if T < 2:
            continue
        diff = np.linalg.norm(act[:T - 1, :7] - ee[1:], axis=1)
        max_diff = diff.max()
        if max_diff >= 1e-6:
            print(f"  MISMATCH ep {i}: max_diff={max_diff:.8f}")
            all_ok = False
        elif i < 5:
            print(f"  ep {i}: max_diff={max_diff:.8f} [OK]")
    if all_ok:
        print(f"  All {len(subsampled)} episodes verified OK")
    else:
        print("  WARNING: Some episodes have action mismatches!")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, episodes=subsampled)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    t1 = datetime.datetime.now()
    elapsed = (t1 - t0).total_seconds()
    print(f"\nSaved {len(subsampled)} episodes to {out_path}  ({file_mb:.1f} MB)")
    print(f"[{t1.strftime('%Y-%m-%d %H:%M:%S')}] Subsample done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
