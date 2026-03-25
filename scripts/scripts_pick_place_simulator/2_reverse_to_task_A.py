#!/usr/bin/env python3
"""Step 2: Build Task A data by time-reversing Task B trajectories.

Time-reverses all sequences and recomputes action[t][:7] = ee_pose[t+1].
Drops the last frame (no valid next-frame target).

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/2_reverse_to_task_A.py \
        --input data/exp_new/task_B_100.npz \
        --out data/exp_new/task_A_reversed_100.npz --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-reverse Task B data to get Task A training data.",
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


def reverse_episode(ep: dict) -> dict:
    """Reverse an episode in time.

    Handles both the original collection format (with obs, gripper, fsm_state)
    and the rollout format (which only has images, ee_pose, obj_pose, action).
    For rollout data, gripper is extracted from action[:, 7] and obs/fsm_state
    are synthesised as zeros.
    """
    T = len(ep["images"])

    # Handle fields that may be absent in rollout data
    has_obs = "obs" in ep
    has_gripper = "gripper" in ep
    has_fsm_state = "fsm_state" in ep

    if has_obs:
        obs_rev = ep["obs"][::-1].copy()
    else:
        obs_rev = np.zeros((T, 36), dtype=np.float32)

    images_rev = ep["images"][::-1].copy()
    ee_rev = ep["ee_pose"][::-1].copy()
    obj_rev = ep["obj_pose"][::-1].copy()

    if has_gripper:
        gripper_rev = ep["gripper"][::-1].copy()
    else:
        # Extract gripper from action[:, 7]
        gripper_rev = ep["action"][:, 7][::-1].copy()

    if has_fsm_state:
        fsm_state_rev = ep["fsm_state"][::-1].copy()
    else:
        fsm_state_rev = np.zeros(T, dtype=np.int32)

    has_wrist = "wrist_images" in ep
    if has_wrist:
        wrist_rev = ep["wrist_images"][::-1].copy()

    # action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]
    new_actions = np.zeros((T, 8), dtype=np.float32)
    new_actions[:T - 1, :7] = ee_rev[1:]
    new_actions[T - 1, :7] = ee_rev[T - 1]
    new_actions[:, 7] = gripper_rev

    # Drop last frame
    result = {
        "obs": obs_rev[:-1].astype(np.float32),
        "images": images_rev[:-1],
        "ee_pose": ee_rev[:-1].astype(np.float32),
        "obj_pose": obj_rev[:-1].astype(np.float32),
        "action": new_actions[:-1].astype(np.float32),
        "gripper": gripper_rev[:-1].astype(np.float32),
        "fsm_state": fsm_state_rev[:-1].astype(np.int32),
        "success": ep["success"],
    }
    # Copy metadata fields if present
    if "place_pose" in ep:
        result["place_pose"] = ep["place_pose"].copy() if hasattr(ep["place_pose"], 'copy') else ep["place_pose"]
    if "goal_pose" in ep:
        result["goal_pose"] = ep["goal_pose"].copy() if hasattr(ep["goal_pose"], 'copy') else ep["goal_pose"]
    if has_wrist:
        result["wrist_images"] = wrist_rev[:-1]
    return result


def verify_episode(ep_fwd: dict, ep_idx: int) -> None:
    ee = ep_fwd["ee_pose"]
    act = ep_fwd["action"]
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
    print(f"Dataset Statistics")
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
    print(f"Gripper CLOSE ratio:   {100 * close_ratio:.1f}%")
    print(f"{'='*60}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, episodes=forward_episodes)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(forward_episodes)} episodes to {out_path}  ({file_mb:.1f} MB)")


if __name__ == "__main__":
    main()
