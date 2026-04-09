#!/usr/bin/env python3
"""Build Task A (stack) data by time-reversing Task B (partial unstack) trajectories.

Adapted for the Exp44 format (2-round partial unstack, 3 cube poses).

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/2_reverse_to_task_A_exp44.py \
        --input data/pick_place_isaac_lab_simulation/exp44/task_B_exp44_100.npz \
        --out data/pick_place_isaac_lab_simulation/exp44/task_A_exp44_100.npz --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-reverse Task B Exp44 data to get Task A training data.",
    )
    parser.add_argument("--input", type=str, required=True, help="Input Task B NPZ.")
    parser.add_argument("--out", type=str, required=True, help="Output Task A NPZ.")
    parser.add_argument("--success_only", type=int, default=1, choices=[0, 1],
                        help="Only keep successful episodes. Default: 1.")
    parser.add_argument("--verify", action="store_true",
                        help="Verify action[t][:7] == ee_pose[t+1].")
    return parser.parse_args()


def load_episodes(path: str) -> list[dict]:
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


# Temporal arrays to reverse and trim (drop last frame)
_TEMPORAL_KEYS = [
    "obs", "images", "wrist_images", "ee_pose",
    "gripper", "fsm_state", "fsm_round",
    "cube_large_pose", "cube_medium_pose", "cube_small_pose",
]


def reverse_episode(ep: dict) -> dict:
    """Reverse an episode in time for the Exp44 task."""
    T = len(ep["images"])

    # Reverse all temporal arrays
    reversed_arrays = {}
    for key in _TEMPORAL_KEYS:
        if key in ep:
            reversed_arrays[key] = ep[key][::-1].copy()

    # Recompute actions: action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]
    ee_rev = reversed_arrays["ee_pose"]
    gripper_rev = reversed_arrays.get("gripper")
    if gripper_rev is None:
        gripper_rev = ep["action"][:, 7][::-1].copy()
        reversed_arrays["gripper"] = gripper_rev

    new_actions = np.zeros((T, 8), dtype=np.float32)
    new_actions[:T - 1, :7] = ee_rev[1:]
    new_actions[T - 1, :7] = ee_rev[T - 1]
    new_actions[:, 7] = gripper_rev

    # Build result, dropping last frame
    result = {}
    for key in _TEMPORAL_KEYS:
        if key in reversed_arrays:
            result[key] = reversed_arrays[key][:-1]
    result["action"] = new_actions[:-1].astype(np.float32)

    # Ensure correct dtypes
    for key in ["obs", "ee_pose", "gripper",
                "cube_large_pose", "cube_medium_pose", "cube_small_pose"]:
        if key in result and result[key].dtype != np.float32:
            result[key] = result[key].astype(np.float32)
    for key in ["fsm_state", "fsm_round"]:
        if key in result and result[key].dtype != np.int32:
            result[key] = result[key].astype(np.int32)

    # Copy metadata (non-temporal) as-is
    for key in ["success", "expert_completed", "success_per_cube",
                "place_targets", "goal_pose"]:
        if key in ep:
            val = ep[key]
            result[key] = val.copy() if hasattr(val, "copy") else val

    return result


def verify_episode(ep: dict, ep_idx: int) -> None:
    ee = ep["ee_pose"]
    act = ep["action"]
    T = len(ee)
    diff = np.linalg.norm(act[:T - 1, :7] - ee[1:], axis=1)
    max_diff = diff.max()
    mean_diff = diff.mean()
    status = "OK" if max_diff < 1e-6 else "MISMATCH"
    print(f"  ep {ep_idx}: max={max_diff:.8f}  mean={mean_diff:.8f}  [{status}]")


def main() -> None:
    args = _parse_args()

    episodes = load_episodes(args.input)

    if args.success_only:
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Filtered to {len(episodes)} successful episodes")

    if not episodes:
        print("ERROR: No episodes to process!")
        return

    forward_episodes = []
    total_steps = 0
    for ep_idx, ep in enumerate(episodes):
        fwd = reverse_episode(ep)
        forward_episodes.append(fwd)
        total_steps += len(fwd["images"])
        if (ep_idx + 1) % 20 == 0 or ep_idx == 0:
            T = len(fwd["images"])
            close_ratio = np.mean(fwd["gripper"] < 0)
            print(f"Episode {ep_idx + 1:4d} | Length: {T:4d} | CLOSE ratio: {100 * close_ratio:.1f}%")

    if args.verify:
        print(f"\n{'='*60}")
        print("Verification: action[t][:7] == ee_pose[t+1]")
        print(f"{'='*60}")
        for i, fwd in enumerate(forward_episodes[:5]):
            verify_episode(fwd, i)

    avg_len = total_steps / len(forward_episodes)
    total_close = sum(np.sum(ep["gripper"] < 0) for ep in forward_episodes)
    close_ratio = total_close / total_steps if total_steps else 0

    print(f"\n{'='*60}")
    print(f"Exp44 Task A Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total episodes:        {len(forward_episodes)}")
    print(f"Total steps:           {total_steps}")
    print(f"Avg episode length:    {avg_len:.1f}")
    if forward_episodes:
        ep0 = forward_episodes[0]
        if "obs" in ep0 and ep0["obs"].ndim == 2:
            print(f"Observation dim:       {ep0['obs'].shape[1]}")
        print(f"Table image shape:     {ep0['images'].shape[1:]}")
        if "wrist_images" in ep0:
            print(f"Wrist image shape:     {ep0['wrist_images'].shape[1:]}")
        print(f"Action dim:            {ep0['action'].shape[1]}")
        for key in ["cube_large_pose", "cube_medium_pose", "cube_small_pose"]:
            if key in ep0:
                print(f"{key} shape: {ep0[key].shape}")
    print(f"Gripper CLOSE ratio:   {100 * close_ratio:.1f}%")
    print(f"{'='*60}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, episodes=forward_episodes)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(forward_episodes)} episodes to {out_path}  ({file_mb:.1f} MB)")


if __name__ == "__main__":
    main()
