#!/usr/bin/env python3
"""Step 3.1: Build forward data by time-reversing reverse rollouts (next-frame action).

This script is designed for data collected with next-frame ee_pose actions
(action[t][:7] = ee_pose[t+1]). It replaces the FSM-waypoint-based
compute_forward_goal_actions logic in script 3 with a much simpler approach:

    After time reversal, action_A[t][:7] = ee_pose_A[t+1]

This is derived directly from the reversed ee_pose sequence — no FSM state
analysis needed.

=============================================================================
WHY THIS IS CORRECT
=============================================================================
In the B dataset (collected by script 1):
    action_B[t][:7] = ee_pose_B[t+1]     (next-frame ee_pose)

After time reversal (ee_pose_A[t] = ee_pose_B[T-1-t]):
    We want: action_A[t][:7] = ee_pose_A[t+1]

This equals:
    ee_pose_A[t+1] = ee_pose_B[T-1-(t+1)] = ee_pose_B[T-2-t]

Which is simply the (t+1)-th element of the reversed ee_pose array.
No FSM segment analysis or waypoint remapping is needed.

=============================================================================
USAGE
=============================================================================
python scripts/scripts_pick_place/3_1_make_forward_data.py \
    --input data/B_circle_200.npz \
    --out data/A_circle_200.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forward dataset by time-reversing reverse rollouts (next-frame action).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input NPZ file with reverse episodes (from script 1, next-frame action format).",
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Output NPZ file for forward episodes.",
    )
    parser.add_argument(
        "--success_only", type=int, default=1, choices=[0, 1],
        help="If 1, only use episodes where the reverse task succeeded.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification checks on the output data.",
    )
    return parser.parse_args()


def load_episodes(path: str) -> list[dict]:
    with np.load(path, allow_pickle=True) as data:
        episodes = list(data["episodes"])
    print(f"Loaded {len(episodes)} episodes from {path}")
    return episodes


def reverse_episode(ep: dict) -> dict:
    """Reverse an episode in time with next-frame ee_pose as action.

    Steps:
        1. Reverse all sequences in time.
        2. Set action[t][:7] = ee_pose[t+1] (next-frame ee_pose from the
           reversed sequence).
        3. Set action[t][7] = gripper[t] (reversed gripper command).
        4. Drop the last frame (it has no next-frame target).
    """
    T = len(ep["obs"])

    # --- Time reversal of all sequences ---
    obs_rev = ep["obs"][::-1].copy()
    images_rev = ep["images"][::-1].copy()
    ee_rev = ep["ee_pose"][::-1].copy()           # (T, 7)
    obj_rev = ep["obj_pose"][::-1].copy()
    gripper_rev = ep["gripper"][::-1].copy()       # (T,)
    fsm_state_rev = ep["fsm_state"][::-1].copy()

    has_wrist = "wrist_images" in ep
    if has_wrist:
        wrist_rev = ep["wrist_images"][::-1].copy()

    # --- Compute action = next-frame ee_pose + gripper ---
    new_actions = np.zeros((T, 8), dtype=np.float32)
    new_actions[:T-1, :7] = ee_rev[1:]       # action[t][:7] = ee_pose[t+1]
    new_actions[T-1, :7] = ee_rev[T-1]       # last frame: stay in place
    new_actions[:, 7] = gripper_rev           # gripper from reversed sequence

    # --- Drop last frame (no valid next-frame target) ---
    result = {
        "obs":       obs_rev[:-1].astype(np.float32),
        "images":    images_rev[:-1],
        "ee_pose":   ee_rev[:-1].astype(np.float32),
        "obj_pose":  obj_rev[:-1].astype(np.float32),
        "action":    new_actions[:-1].astype(np.float32),
        "gripper":   gripper_rev[:-1].astype(np.float32),
        "fsm_state": fsm_state_rev[:-1].astype(np.int32),
        "place_pose": ep["place_pose"].copy(),
        "goal_pose":  ep["goal_pose"].copy(),
        "success":    ep["success"],
    }
    if has_wrist:
        result["wrist_images"] = wrist_rev[:-1]

    return result


def verify_episode(ep_fwd: dict, ep_idx: int) -> None:
    """Verify that action[t][:7] == ee_pose[t+1] in the forward episode."""
    ee = ep_fwd["ee_pose"]     # (T-1, 7)
    act = ep_fwd["action"]     # (T-1, 8)
    T = len(ee)

    # action[t][:7] should equal ee_pose[t+1] for t = 0..T-2
    diff = np.linalg.norm(act[:T-1, :7] - ee[1:], axis=1)
    max_diff = diff.max()
    mean_diff = diff.mean()

    status = "OK" if max_diff < 1e-6 else "MISMATCH"
    print(f"  ep {ep_idx}: action[:7] vs ee_pose[t+1]  "
          f"max={max_diff:.8f}  mean={mean_diff:.8f}  [{status}]")


def main() -> None:
    args = _parse_args()

    # --- Load ---
    episodes = load_episodes(args.input)

    if args.success_only:
        episodes = [ep for ep in episodes if ep.get("success", False)]
        print(f"Filtered to {len(episodes)} successful episodes")

    if not episodes:
        print("ERROR: No episodes to process!")
        return

    # --- Process ---
    forward_episodes = []
    total_steps = 0

    for ep_idx, ep in enumerate(episodes):
        fwd = reverse_episode(ep)
        forward_episodes.append(fwd)
        total_steps += len(fwd["obs"])

        if (ep_idx + 1) % 20 == 0 or ep_idx == 0:
            T = len(fwd["obs"])
            close_ratio = np.mean(fwd["gripper"] < 0)
            print(f"Episode {ep_idx+1:4d} | Length: {T:4d} | CLOSE ratio: {100*close_ratio:.1f}%")

    # --- Verify ---
    if args.verify:
        print(f"\n{'='*60}")
        print("Verification: action[t][:7] == ee_pose[t+1]")
        print(f"{'='*60}")
        for i, fwd in enumerate(forward_episodes[:5]):
            verify_episode(fwd, i)

    # --- Stats ---
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
        print(f"Observation dim:       {ep0['obs'].shape[1]}")
        print(f"Table image shape:     {ep0['images'].shape[1:]}")
        if "wrist_images" in ep0:
            print(f"Wrist image shape:     {ep0['wrist_images'].shape[1:]}")
        print(f"Action dim:            {ep0['action'].shape[1]}")
    print(f"Gripper CLOSE ratio:   {100*close_ratio:.1f}%")
    print(f"{'='*60}\n")

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, episodes=forward_episodes)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(forward_episodes)} episodes to {out_path}  ({file_mb:.1f} MB)")


if __name__ == "__main__":
    main()
