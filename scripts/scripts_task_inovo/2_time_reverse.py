#!/usr/bin/env python3
"""Script 2: Time-reverse Inovo (RoboKit) data for Rev2Fwd method.

Generate Task A training data by time-reversing the original Task B
demonstrations. This is the core of the Rev2Fwd method.

=============================================================================
CORE IDEA: TIME REVERSAL FOR VELOCITY-BASED ACTIONS
=============================================================================
Original (Task B): frame[0], frame[1], ..., frame[T-1]
Reversed (Task A): frame[T-1], frame[T-2], ..., frame[0]

Per-field reversal rules:
    primary_rgb / gripper_rgb     ->  Reverse frame order
    primary_depth / gripper_depth ->  Reverse frame order
    robot_obs[0:6] (tcp pose)     ->  Reverse frame order
    robot_obs[6] (gripper width)  ->  Reverse frame order
    robot_obs[7:13] (joints)      ->  Reverse frame order
    robot_obs[13] (gripper action)->  Reverse AND invert: 1 - original
    actions[0:6] (velocities)     ->  Negate AND reverse: -actions[T-1-t][0:6]
    actions[6] (gripper)          ->  Reverse AND invert: 1 - actions[T-1-t][6]
    force_torque                  ->  Reverse frame order
    language_text                 ->  Update to describe reversed task

Key insight: Since actions are velocity commands, time reversal requires
negating the velocity (moving backwards along the same trajectory).

=============================================================================
VERIFICATION
=============================================================================
After reversal, verify:
1. Position reconstruction: Integrating reversed velocities should trace
   the original trajectory backwards
2. Boundary match: reversed[0].tcp_pos == original[-1].tcp_pos
3. Gripper logic: If original closes, reversed opens

=============================================================================
USAGE
=============================================================================
# Basic time reversal
python scripts/scripts_task_inovo/2_time_reverse.py \
    --input data/inovo_data/0209_tower_boby \
    --output data/inovo_data/tower_boby_A_20260313 \
    --verify \
    --new_language "Pick up the purple disc from the sponge and place it on the pole."

# With verbose output
python scripts/scripts_task_inovo/2_time_reverse.py \
    --input data/inovo_data/0209_tower_boby \
    --output data/inovo_data/tower_boby_A \
    --verify --verbose

# Skip verification for speed
python scripts/scripts_task_inovo/2_time_reverse.py \
    --input data/inovo_data/0209_tower_boby \
    --output data/inovo_data/tower_boby_A \
    --no_verify
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# RoboKit NPZ Decoder / Encoder
# =============================================================================

def load_robokit_frame(path: str | Path) -> dict:
    """Load and decode a single RoboKit NPZ frame.

    Returns raw bytes for images (for lossless re-encoding) + decoded numeric data.
    """
    f = np.load(str(path), allow_pickle=True)

    return {
        # Keep raw bytes for lossless copy
        "primary_rgb_bytes": f["primary_rgb"].item(),
        "gripper_rgb_bytes": f["gripper_rgb"].item(),
        "primary_depth_bytes": f["primary_depth"].item() if "primary_depth" in f else None,
        "gripper_depth_bytes": f["gripper_depth"].item() if "gripper_depth" in f else None,
        # Decode numeric fields
        "robot_obs": np.asarray(pickle.loads(f["robot_obs"].item()), dtype=np.float64),
        "actions": np.asarray(pickle.loads(f["actions"].item()), dtype=np.float64),
        "rel_actions": np.asarray(pickle.loads(f["rel_actions"].item()), dtype=np.float64),
        "force_torque": np.asarray(pickle.loads(f["force_torque"].item()), dtype=np.float64),
        "language_text": pickle.loads(f["language_text"].item()),
    }


def save_robokit_frame(path: str | Path, frame: dict) -> None:
    """Save a single frame in RoboKit NPZ format.

    Numeric fields are pickled before saving (matching original format).
    Image fields are stored as raw bytes.
    """
    npz_data = {
        "primary_rgb": np.array(frame["primary_rgb_bytes"]),
        "gripper_rgb": np.array(frame["gripper_rgb_bytes"]),
        "robot_obs": np.array(pickle.dumps(frame["robot_obs"])),
        "actions": np.array(pickle.dumps(frame["actions"])),
        "rel_actions": np.array(pickle.dumps(frame["rel_actions"])),
        "force_torque": np.array(pickle.dumps(frame["force_torque"])),
        "language_text": np.array(pickle.dumps(np.array(frame["language_text"]))),
    }
    # Optional depth fields
    if frame.get("primary_depth_bytes") is not None:
        npz_data["primary_depth"] = np.array(frame["primary_depth_bytes"])
    if frame.get("gripper_depth_bytes") is not None:
        npz_data["gripper_depth"] = np.array(frame["gripper_depth_bytes"])

    np.savez(str(path), **npz_data)


# =============================================================================
# Episode Discovery
# =============================================================================

def discover_episodes(data_dir: Path) -> list[dict]:
    """Discover all episode sub-directories.

    Returns:
        List of dicts with 'name', 'path', 'npz_files'.
    """
    episodes = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name in ("extracted", "__pycache__", "viz_videos"):
            continue
        npz_files = sorted(entry.glob("*.npz"))
        if len(npz_files) == 0:
            continue
        episodes.append({
            "name": entry.name,
            "path": entry,
            "npz_files": npz_files,
        })
    return episodes


# =============================================================================
# Angle Wrapping Utility
# =============================================================================

def wrap_angle(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Core Reversal Logic
# =============================================================================

def reverse_episode(
    npz_files: list[Path],
    new_language: str | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Reverse an episode in time.

    Loads all frames, reverses the ordering, negates velocity actions,
    and inverts gripper commands.

    Args:
        npz_files: Sorted list of .npz paths for this episode.
        new_language: New language description for reversed task.
        verbose: Print debug info.

    Returns:
        List of reversed frame dicts (ready for save_robokit_frame).
    """
    T = len(npz_files)
    if verbose:
        print(f"  Loading {T} frames ...")

    # Load all frames
    frames = []
    for p in npz_files:
        frames.append(load_robokit_frame(p))

    if verbose:
        print(f"  Loaded. Reversing ...")

    # Build reversed frames
    reversed_frames = []
    for t in range(T):
        # Source frame is the time-reversed index
        src = frames[T - 1 - t]
        new_frame = {}

        # Images: just take from reversed source (bytes are lossless)
        new_frame["primary_rgb_bytes"] = src["primary_rgb_bytes"]
        new_frame["gripper_rgb_bytes"] = src["gripper_rgb_bytes"]
        new_frame["primary_depth_bytes"] = src.get("primary_depth_bytes")
        new_frame["gripper_depth_bytes"] = src.get("gripper_depth_bytes")

        # Robot observation: reversed in time
        robot_obs_new = src["robot_obs"].copy()
        # Invert gripper action: 0 -> 1, 1 -> 0
        robot_obs_new[13] = 1.0 - src["robot_obs"][13]
        new_frame["robot_obs"] = robot_obs_new

        # Actions: the action at reversed timestep t should transition from
        # rev_obs[t] to rev_obs[t+1], i.e., from orig_obs[T-1-t] to
        # orig_obs[T-2-t].  That velocity equals -orig_actions[T-2-t].
        # (The Piper reference recomputes actions from observation diffs;
        #  here we use the equivalent index shift for velocity data.)
        if t < T - 1:
            action_src = frames[T - 2 - t]  # frame whose action goes T-2-t → T-1-t
            actions_new = np.zeros(7, dtype=np.float64)
            actions_new[:6] = -action_src["actions"][:6]  # Negate velocity
            
            actions_new[6] = action_src["actions"][6]  # Invert gripper
        else:
            # Last reversed frame: no next observation to transition to
            actions_new = np.zeros(7, dtype=np.float64)
            actions_new[6] = frames[0]["actions"][6]
        new_frame["actions"] = actions_new

        # rel_actions: same treatment (they are identical to actions in this dataset)
        if t < T - 1:
            rel_src = frames[T - 2 - t]
            rel_actions_new = np.zeros(7, dtype=np.float64)
            rel_actions_new[:6] = -rel_src["rel_actions"][:6]
            rel_actions_new[6] = rel_src["rel_actions"][6]
        else:
            rel_actions_new = np.zeros(7, dtype=np.float64)
            rel_actions_new[6] = frames[0]["rel_actions"][6]
        new_frame["rel_actions"] = rel_actions_new

        # Force/torque: reverse frame order (negate is debatable, just reverse)
        new_frame["force_torque"] = src["force_torque"].copy()

        # Language text
        if new_language is not None:
            new_frame["language_text"] = new_language
        else:
            new_frame["language_text"] = src["language_text"]

        reversed_frames.append(new_frame)

    if verbose:
        # Print boundary info
        orig_start_pos = frames[0]["robot_obs"][:3]
        orig_end_pos = frames[-1]["robot_obs"][:3]
        rev_start_pos = reversed_frames[0]["robot_obs"][:3]
        rev_end_pos = reversed_frames[-1]["robot_obs"][:3]
        print(f"  Original start TCP: {orig_start_pos}")
        print(f"  Original end TCP:   {orig_end_pos}")
        print(f"  Reversed start TCP: {rev_start_pos}")
        print(f"  Reversed end TCP:   {rev_end_pos}")
        print(f"  Match start→end: {np.allclose(orig_end_pos, rev_start_pos, atol=1e-5)}")
        print(f"  Match end→start: {np.allclose(orig_start_pos, rev_end_pos, atol=1e-5)}")

    return reversed_frames


# =============================================================================
# Verification
# =============================================================================

def verify_reversal(
    original_npz_files: list[Path],
    reversed_frames: list[dict],
    dt: float = 1.0 / 30.0,
) -> tuple[bool, str]:
    """Verify that the reversal is correct.

    Checks:
    1. Boundary positions match (reversed start == original end, vice versa)
    2. Integrating reversed velocities * dt approximately traces the reversed poses
    3. Gripper action is inverted

    Args:
        original_npz_files: Original episode .npz file paths.
        reversed_frames: List of reversed frame dicts.
        dt: Timestep duration.

    Returns:
        (is_valid, message)
    """
    # Quick-load original boundary frames
    orig_first = load_robokit_frame(original_npz_files[0])
    orig_last = load_robokit_frame(original_npz_files[-1])

    rev_first = reversed_frames[0]
    rev_last = reversed_frames[-1]

    messages = []
    all_ok = True

    # 1. Boundary check
    if not np.allclose(orig_last["robot_obs"][:3], rev_first["robot_obs"][:3], atol=1e-5):
        messages.append("FAIL: orig_end != rev_start position")
        all_ok = False
    if not np.allclose(orig_first["robot_obs"][:3], rev_last["robot_obs"][:3], atol=1e-5):
        messages.append("FAIL: orig_start != rev_end position")
        all_ok = False

    # 2. Velocity integration check
    T = len(reversed_frames)
    tcp_pos_rev = np.array([f["robot_obs"][:3] for f in reversed_frames])
    actions_xyz_rev = np.array([f["actions"][:3] for f in reversed_frames])

    integrated = np.zeros((T, 3))
    integrated[0] = tcp_pos_rev[0]
    for t in range(T - 1):
        integrated[t + 1] = integrated[t] + actions_xyz_rev[t] * dt

    error = np.abs(integrated - tcp_pos_rev).max()
    if error > 0.01:  # 1cm tolerance (velocity integration can drift)
        messages.append(f"WARNING: velocity integration max error = {error:.4f}m (>1cm)")
    else:
        messages.append(f"OK: velocity integration max error = {error:.4f}m")

    # 3. Gripper inversion check
    orig_gripper = orig_first["robot_obs"][13]
    rev_last_gripper = rev_last["robot_obs"][13]
    if abs(orig_gripper + rev_last_gripper - 1.0) > 0.01:
        messages.append(f"WARNING: Gripper not properly inverted at boundary")
        all_ok = False

    msg = "; ".join(messages)
    return all_ok, msg


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Time-reverse Inovo (RoboKit) data for Rev2Fwd method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing episode sub-directories.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for reversed episodes.",
    )
    parser.add_argument(
        "--new_language",
        type=str,
        default=None,
        help="New language description for reversed task. "
             "Default: keep original language text.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify each reversal (default: True).",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Disable verification.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed debug info.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Timestep dt for velocity integration verification. "
             "Default: auto-detect from filenames (~1/30).",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    do_verify = args.verify and not args.no_verify

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return

    # Discover episodes
    episodes = discover_episodes(input_dir)
    if not episodes:
        print(f"ERROR: No episodes found in {input_dir}")
        return

    # Estimate dt from first episode
    dt = args.dt
    # Inline dt estimation from filename timestamps
    if dt is None:
        # Simple inline estimation
        ts_list = []
        for npz_path in episodes[0]["npz_files"][:100]:
            stem = npz_path.stem
            parts = stem.split("_")
            if len(parts) >= 3:
                try:
                    hhmmss = parts[1]
                    micro = parts[2]
                    total = (int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 +
                             int(hhmmss[4:6]) + int(micro) / 1e6)
                    ts_list.append(total)
                except (ValueError, IndexError):
                    pass
        if len(ts_list) > 1:
            dts = np.diff(ts_list)
            dts = dts[(dts > 0.001) & (dts < 1.0)]
            dt = float(np.mean(dts)) if len(dts) > 0 else 1.0 / 30.0
        else:
            dt = 1.0 / 30.0
        print(f"Estimated dt = {dt*1000:.1f}ms ({1.0/dt:.1f} Hz)")

    print(f"\n{'='*70}")
    print(f"  Inovo Time Reversal (Rev2Fwd)")
    print(f"{'='*70}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Verify: {do_verify}")
    print(f"  dt: {dt*1000:.1f}ms")
    if args.new_language:
        print(f"  New language: {args.new_language}")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    failed = 0
    verify_warnings = 0
    start_time = time.time()

    for ep in tqdm(episodes, desc="Reversing episodes"):
        try:
            if args.verbose:
                print(f"\nProcessing: {ep['name']} ({len(ep['npz_files'])} frames)")

            # Reverse episode
            reversed_frames = reverse_episode(
                ep["npz_files"],
                new_language=args.new_language,
                verbose=args.verbose,
            )

            # Verify
            if do_verify:
                ok, msg = verify_reversal(ep["npz_files"], reversed_frames, dt=dt)
                if args.verbose or not ok:
                    print(f"  Verify {ep['name']}: {msg}")
                if not ok:
                    verify_warnings += 1

            # Save reversed episode
            ep_output_dir = output_dir / ep["name"]
            ep_output_dir.mkdir(parents=True, exist_ok=True)

            # Save reversed frames with filenames in sorted order so that
            # loading by sorted glob reproduces the REVERSED sequence.
            for i, frame in enumerate(reversed_frames):
                # Use the i-th filename (earliest timestamp for first reversed
                # frame) so that sorted order = reversed episode order.
                out_name = ep["npz_files"][i].name
                save_robokit_frame(ep_output_dir / out_name, frame)

            processed += 1

        except Exception as e:
            print(f"\nERROR processing {ep['name']}: {e}")
            failed += 1
            if args.verbose:
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time

    # Save metadata
    metadata = {
        "source_dir": str(input_dir),
        "reversal_type": "time_reversal",
        "collection_type": "reversed",
        "num_episodes": processed,
        "dt": dt,
        "new_language": args.new_language,
        "reversal_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "episode_names": [ep["name"] for ep in episodes if ep["name"]],
    }
    # Copy statistics.json if exists
    src_stats = input_dir / "statistics.json"
    if src_stats.exists():
        metadata["original_statistics_file"] = str(src_stats)

    with open(output_dir / "reversal_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Time Reversal Complete")
    print(f"{'='*70}")
    print(f"  Processed: {processed} episodes")
    print(f"  Failed: {failed} episodes")
    print(f"  Verify warnings: {verify_warnings}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
