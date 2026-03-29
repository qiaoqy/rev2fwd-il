#!/usr/bin/env python3
"""Apply a Z-offset correction to ee_pose in rollout episodes and recompute actions.

Usage:
    python 3_apply_z_offset.py \
        --input iter2_collect_A_adjusted.npz \
        --out iter2_collect_A_zfixed.npz \
        --z_offset -0.02 \
        --success_only 1

This shifts ee_pose[:, 2] by z_offset for every episode, then recomputes
action[t][:7] = ee_pose[t+1] and action[t][7] = gripper[t].
"""

import argparse
import numpy as np
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply Z-offset correction to ee_pose and recompute actions."
    )
    p.add_argument("--input", type=str, required=True, help="Input NPZ file.")
    p.add_argument("--out", type=str, required=True, help="Output NPZ file.")
    p.add_argument("--z_offset", type=float, required=True,
                   help="Z offset to ADD to ee_pose[:, 2]. Negative = lower.")
    p.add_argument("--success_only", type=int, default=0, choices=[0, 1],
                   help="If 1, only keep episodes with success=True.")
    return p.parse_args()


def apply_z_offset(episode: dict, z_offset: float) -> dict:
    """Apply z_offset to ee_pose and recompute actions in-place."""
    ep = {k: (v.copy() if isinstance(v, np.ndarray) else v)
          for k, v in episode.items()}

    ee_pose = ep["ee_pose"]  # (T, 7) or (T, >=3)
    ee_pose[:, 2] += z_offset
    ep["ee_pose"] = ee_pose

    # Recompute actions: action[t][:7] = ee_pose[t+1], action[t][7] = gripper (unchanged)
    T = len(ee_pose)
    if "action" in ep and T > 1:
        action = ep["action"].copy()
        # Preserve original gripper values (column 7) while updating pose (columns 0-6)
        gripper = action[:, 7].copy()
        for t in range(T - 1):
            action[t, :7] = ee_pose[t + 1]
            action[t, 7] = gripper[t]
        # Last frame: repeat previous action target
        action[T - 1, :7] = ee_pose[T - 1]
        action[T - 1, 7] = gripper[T - 1]
        ep["action"] = action

    return ep


def main() -> None:
    args = _parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}")
        return

    data = np.load(str(in_path), allow_pickle=True)
    episodes = list(data["episodes"])

    if args.success_only:
        episodes = [ep for ep in episodes if ep.get("success", False)]

    if not episodes:
        print(f"WARNING: No episodes to process in {in_path.name}")
        # Save empty
        np.savez_compressed(args.out, episodes=np.array([], dtype=object))
        return

    total_frames_before = sum(len(ep["ee_pose"]) for ep in episodes)

    corrected = []
    for ep in episodes:
        corrected.append(apply_z_offset(ep, args.z_offset))

    total_frames_after = sum(len(ep["ee_pose"]) for ep in corrected)

    print(f"Z-offset correction: {in_path.name}")
    print(f"  Episodes: {len(corrected)}")
    print(f"  Frames: {total_frames_before} → {total_frames_after} (unchanged)")
    print(f"  Z offset: {args.z_offset:+.4f} m")

    # Verify offset was applied
    sample_ep = corrected[0]
    orig_ep = episodes[0]
    z_diff = sample_ep["ee_pose"][0, 2] - orig_ep["ee_pose"][0, 2]
    print(f"  Verification: first frame z diff = {z_diff:+.6f} (expected {args.z_offset:+.4f})")

    np.savez_compressed(args.out,
                        episodes=np.array(corrected, dtype=object))
    out_size = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"  Saved: {args.out} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
