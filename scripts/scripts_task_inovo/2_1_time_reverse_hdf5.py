#!/usr/bin/env python3
"""Script 2-1: Time-reverse HDF5 Inovo (RoboKit) data for Rev2Fwd method.

HDF5 version of Script 2. Reverses episodes in an HDF5 file and saves
the result as a new HDF5 file with the same structure.

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
    robot_obs[13] (gripper action)->  Reverse frame order (keep original value)
    actions[0:6] (velocities)     ->  Negate AND reverse: -actions[T-1-t][0:6]
    actions[6] (gripper)          ->  Reverse AND invert: 1 - actions[T-1-t][6]
    force_torque                  ->  Reverse frame order
    language_text                 ->  Update to describe reversed task

=============================================================================
HDF5 DATA FORMAT
=============================================================================
The HDF5 file contains these datasets, with N = total frames across all episodes:
    primary_rgb   : (N,) object  - JPEG-encoded bytes per frame
    gripper_rgb   : (N,) object  - JPEG-encoded bytes per frame
    primary_depth : (N,) object  - PNG-encoded depth per frame
    gripper_depth : (N,) object  - PNG-encoded depth per frame
    robot_obs     : (N, 14) float64
    actions       : (N, 7) float64
    rel_actions   : (N, 7) float64
    force_torque  : (N, 6) float64
    language_text : (N,) |S300   - fixed-length byte strings

Episode boundaries are detected from the original NPZ directory structure
(--ref_npz_dir) or provided explicitly (--episode_ends).

=============================================================================
USAGE
=============================================================================
# Using original NPZ directory for episode boundary detection
python scripts/scripts_task_inovo/2_1_time_reverse_hdf5.py \
    --input /path/to/data.h5 \
    --output /path/to/reversed.h5 \
    --ref_npz_dir /path/to/original_npz_episodes \
    --new_language "Pick up the purple disc from the sponge and place it on the pole." \
    --verify

# Using explicit episode ends (comma-separated cumulative frame indices)
python scripts/scripts_task_inovo/2_1_time_reverse_hdf5.py \
    --input /path/to/data.h5 \
    --output /path/to/reversed.h5 \
    --episode_ends 1158,2337,3647,4874 \
    --verify

# Quick run without verification
python scripts/scripts_task_inovo/2_1_time_reverse_hdf5.py \
    --input /path/to/data.h5 \
    --output /path/to/reversed.h5 \
    --ref_npz_dir /path/to/original_npz_episodes \
    --no_verify
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


# =============================================================================
# Episode Boundary Discovery
# =============================================================================

def discover_episode_ends_from_npz(npz_dir: Path) -> list[int]:
    """Discover episode boundaries from the original NPZ directory structure.

    Each sub-directory in npz_dir is one episode. The number of .npz files
    in each directory gives the episode length.

    Returns:
        List of cumulative end indices (exclusive), e.g. [1158, 2337, ...].
    """
    episodes = []
    for entry in sorted(npz_dir.iterdir()):
        if not entry.is_dir() or entry.name in ("extracted", "__pycache__", "viz_videos"):
            continue
        n_frames = len(list(entry.glob("*.npz")))
        if n_frames > 0:
            episodes.append(n_frames)

    if not episodes:
        raise ValueError(f"No episodes found in {npz_dir}")

    return list(np.cumsum(episodes))


def parse_episode_ends(episode_ends_str: str) -> list[int]:
    """Parse comma-separated episode end indices."""
    return [int(x.strip()) for x in episode_ends_str.split(",")]


# =============================================================================
# Core Reversal Logic
# =============================================================================

def reverse_episode_slice(
    actions: np.ndarray,
    rel_actions: np.ndarray,
    robot_obs: np.ndarray,
    force_torque: np.ndarray,
    primary_rgb: np.ndarray,
    gripper_rgb: np.ndarray,
    primary_depth: np.ndarray | None,
    gripper_depth: np.ndarray | None,
    language_text: np.ndarray,
    new_language: str | None,
) -> dict:
    """Reverse a single episode slice (all arrays are for one episode).

    Args:
        actions: (T, 7) velocity actions
        rel_actions: (T, 7) relative actions
        robot_obs: (T, 14) robot observations
        force_torque: (T, 6) force/torque readings
        primary_rgb: (T,) encoded image bytes
        gripper_rgb: (T,) encoded image bytes
        primary_depth: (T,) encoded depth bytes or None
        gripper_depth: (T,) encoded depth bytes or None
        language_text: (T,) byte strings
        new_language: New language description, or None to keep original

    Returns:
        Dict of reversed arrays with same shapes.
    """
    T = len(actions)

    # --- Images: reverse frame order ---
    rev_primary_rgb = primary_rgb[::-1].copy()
    rev_gripper_rgb = gripper_rgb[::-1].copy()
    rev_primary_depth = primary_depth[::-1].copy() if primary_depth is not None else None
    rev_gripper_depth = gripper_depth[::-1].copy() if gripper_depth is not None else None

    # --- Robot observation: reverse frame order ---
    rev_robot_obs = robot_obs[::-1].copy()

    # --- Force/torque: reverse frame order ---
    rev_force_torque = force_torque[::-1].copy()

    # --- Actions: negate velocity + reverse index ---
    # At reversed timestep t, the action should transition from
    # rev_obs[t] to rev_obs[t+1], i.e., from orig_obs[T-1-t] to orig_obs[T-2-t].
    # That velocity = -orig_actions[T-2-t].
    rev_actions = np.zeros_like(actions)
    if T > 1:
        # For t = 0..T-2: rev_actions[t] = -actions[T-2-t]
        # Equivalently: rev_actions[0:T-1] comes from actions[0:T-1] reversed and negated
        rev_actions[:T-1, :6] = -actions[:T-1, :6][::-1]
        rev_actions[:T-1, 6] = actions[:T-1, 6][::-1]
    # Last frame: zero velocity, copy gripper from original first frame
    rev_actions[T-1, :6] = 0.0
    rev_actions[T-1, 6] = actions[0, 6]

    # --- Rel actions: same treatment ---
    rev_rel_actions = np.zeros_like(rel_actions)
    if T > 1:
        rev_rel_actions[:T-1, :6] = -rel_actions[:T-1, :6][::-1]
        rev_rel_actions[:T-1, 6] = rel_actions[:T-1, 6][::-1]
    rev_rel_actions[T-1, :6] = 0.0
    rev_rel_actions[T-1, 6] = rel_actions[0, 6]

    # --- Language text ---
    if new_language is not None:
        lang_bytes = new_language.encode("utf-8")
        # Pad to S300 format
        rev_language = np.array([lang_bytes] * T, dtype="|S300")
    else:
        rev_language = language_text[::-1].copy()

    return {
        "actions": rev_actions,
        "rel_actions": rev_rel_actions,
        "robot_obs": rev_robot_obs,
        "force_torque": rev_force_torque,
        "primary_rgb": rev_primary_rgb,
        "gripper_rgb": rev_gripper_rgb,
        "primary_depth": rev_primary_depth,
        "gripper_depth": rev_gripper_depth,
        "language_text": rev_language,
    }


# =============================================================================
# Verification
# =============================================================================

def verify_episode_reversal(
    orig_actions: np.ndarray,
    orig_robot_obs: np.ndarray,
    rev_actions: np.ndarray,
    rev_robot_obs: np.ndarray,
    dt: float,
    ep_idx: int,
) -> tuple[bool, str]:
    """Verify reversal correctness for one episode.

    Checks:
    1. Boundary positions match (reversed start == original end)
    2. Velocity integration roughly traces reversed poses
    3. Gripper inversion
    """
    messages = []
    all_ok = True
    T = len(orig_actions)

    # 1. Boundary check
    if not np.allclose(orig_robot_obs[-1, :3], rev_robot_obs[0, :3], atol=1e-5):
        messages.append(f"Ep{ep_idx} FAIL: orig_end != rev_start position")
        all_ok = False
    if not np.allclose(orig_robot_obs[0, :3], rev_robot_obs[-1, :3], atol=1e-5):
        messages.append(f"Ep{ep_idx} FAIL: orig_start != rev_end position")
        all_ok = False

    # 2. Velocity integration check
    tcp_pos_rev = rev_robot_obs[:, :3]
    actions_xyz_rev = rev_actions[:, :3]

    integrated = np.zeros((T, 3))
    integrated[0] = tcp_pos_rev[0]
    for t in range(T - 1):
        integrated[t + 1] = integrated[t] + actions_xyz_rev[t] * dt

    error = np.abs(integrated - tcp_pos_rev).max()
    if error > 0.01:
        messages.append(f"Ep{ep_idx} WARNING: velocity integration max error={error:.4f}m")
    else:
        messages.append(f"Ep{ep_idx} OK: vel integration error={error:.4f}m")

    # 3. Gripper consistency check (reversed value should equal original)
    orig_gripper_first = orig_robot_obs[0, 13]
    rev_gripper_last = rev_robot_obs[-1, 13]
    if abs(orig_gripper_first - rev_gripper_last) > 0.01:
        messages.append(f"Ep{ep_idx} WARNING: gripper mismatch at boundary: orig={orig_gripper_first:.3f}, rev={rev_gripper_last:.3f}")
        all_ok = False

    return all_ok, "; ".join(messages)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Time-reverse HDF5 Inovo (RoboKit) data for Rev2Fwd method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input HDF5 file path.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output HDF5 file path for reversed data.",
    )
    parser.add_argument(
        "--ref_npz_dir",
        type=str,
        default=None,
        help="Reference NPZ directory to detect episode boundaries. "
             "Each sub-directory = one episode.",
    )
    parser.add_argument(
        "--episode_ends",
        type=str,
        default=None,
        help="Comma-separated cumulative episode end indices (exclusive). "
             "E.g. '1158,2337,3647'. Alternative to --ref_npz_dir.",
    )
    parser.add_argument(
        "--new_language",
        type=str,
        default=None,
        help="New language description for reversed task.",
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
        default=1.0 / 30.0,
        help="Timestep dt for velocity integration verification (default: 1/30).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    do_verify = args.verify and not args.no_verify

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    # --- Determine episode boundaries ---
    if args.ref_npz_dir is not None:
        ref_dir = Path(args.ref_npz_dir)
        if not ref_dir.exists():
            print(f"ERROR: Reference NPZ directory not found: {ref_dir}")
            return
        episode_ends = discover_episode_ends_from_npz(ref_dir)
        print(f"Discovered {len(episode_ends)} episodes from NPZ directory.")
    elif args.episode_ends is not None:
        episode_ends = parse_episode_ends(args.episode_ends)
        print(f"Using {len(episode_ends)} episodes from --episode_ends.")
    else:
        print("ERROR: Must provide either --ref_npz_dir or --episode_ends "
              "to specify episode boundaries.")
        return

    # --- Load all data into memory ---
    print(f"\nLoading HDF5 file: {input_path}")
    load_start = time.time()

    with h5py.File(str(input_path), "r") as f_in:
        total_frames = f_in["actions"].shape[0]

        # Validate episode_ends
        if episode_ends[-1] != total_frames:
            print(f"ERROR: Last episode_end ({episode_ends[-1]}) != "
                  f"total frames ({total_frames})")
            return

        # Load all datasets into memory
        actions = f_in["actions"][:]           # (N, 7)
        rel_actions = f_in["rel_actions"][:]   # (N, 7)
        robot_obs = f_in["robot_obs"][:]       # (N, 14)
        force_torque = f_in["force_torque"][:] # (N, 6)
        language_text = f_in["language_text"][:] # (N,)

        # Variable-length image datasets - load as object arrays
        primary_rgb = f_in["primary_rgb"][:]   # (N,) object
        gripper_rgb = f_in["gripper_rgb"][:]   # (N,) object

        has_primary_depth = "primary_depth" in f_in
        has_gripper_depth = "gripper_depth" in f_in
        primary_depth = f_in["primary_depth"][:] if has_primary_depth else None
        gripper_depth = f_in["gripper_depth"][:] if has_gripper_depth else None

    load_elapsed = time.time() - load_start
    print(f"Loaded {total_frames} frames in {load_elapsed:.1f}s")

    # --- Episode info ---
    ep_starts = [0] + episode_ends[:-1]
    n_episodes = len(episode_ends)
    ep_sizes = [episode_ends[i] - ep_starts[i] for i in range(n_episodes)]

    print(f"\n{'='*70}")
    print(f"  HDF5 Time Reversal (Rev2Fwd)")
    print(f"{'='*70}")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Frames:   {total_frames}")
    print(f"  Verify:   {do_verify}")
    print(f"  dt:       {args.dt*1000:.1f}ms")
    if args.new_language:
        print(f"  New lang: {args.new_language}")
    print(f"{'='*70}\n")

    # --- Reverse each episode ---
    # Pre-allocate output arrays
    rev_actions = np.zeros_like(actions)
    rev_rel_actions = np.zeros_like(rel_actions)
    rev_robot_obs = np.zeros_like(robot_obs)
    rev_force_torque = np.zeros_like(force_torque)
    rev_language_text = np.empty_like(language_text)
    rev_primary_rgb = np.empty_like(primary_rgb)
    rev_gripper_rgb = np.empty_like(gripper_rgb)
    rev_primary_depth = np.empty_like(primary_depth) if has_primary_depth else None
    rev_gripper_depth = np.empty_like(gripper_depth) if has_gripper_depth else None

    verify_warnings = 0
    start_time = time.time()

    for ep_idx in tqdm(range(n_episodes), desc="Reversing episodes"):
        s = ep_starts[ep_idx]
        e = episode_ends[ep_idx]

        result = reverse_episode_slice(
            actions=actions[s:e],
            rel_actions=rel_actions[s:e],
            robot_obs=robot_obs[s:e],
            force_torque=force_torque[s:e],
            primary_rgb=primary_rgb[s:e],
            gripper_rgb=gripper_rgb[s:e],
            primary_depth=primary_depth[s:e] if has_primary_depth else None,
            gripper_depth=gripper_depth[s:e] if has_gripper_depth else None,
            language_text=language_text[s:e],
            new_language=args.new_language,
        )

        # Store into pre-allocated arrays
        rev_actions[s:e] = result["actions"]
        rev_rel_actions[s:e] = result["rel_actions"]
        rev_robot_obs[s:e] = result["robot_obs"]
        rev_force_torque[s:e] = result["force_torque"]
        rev_language_text[s:e] = result["language_text"]
        rev_primary_rgb[s:e] = result["primary_rgb"]
        rev_gripper_rgb[s:e] = result["gripper_rgb"]
        if has_primary_depth:
            rev_primary_depth[s:e] = result["primary_depth"]
        if has_gripper_depth:
            rev_gripper_depth[s:e] = result["gripper_depth"]

        # Verify
        if do_verify:
            ok, msg = verify_episode_reversal(
                orig_actions=actions[s:e],
                orig_robot_obs=robot_obs[s:e],
                rev_actions=result["actions"],
                rev_robot_obs=result["robot_obs"],
                dt=args.dt,
                ep_idx=ep_idx,
            )
            if args.verbose or not ok:
                print(f"  {msg}")
            if not ok:
                verify_warnings += 1

        if args.verbose:
            print(f"  Ep {ep_idx}: {e-s} frames, "
                  f"orig TCP start={robot_obs[s, :3]}, "
                  f"rev TCP start={result['robot_obs'][0, :3]}")

    reverse_elapsed = time.time() - start_time
    print(f"\nReversed {n_episodes} episodes in {reverse_elapsed:.1f}s")

    # --- Save output HDF5 ---
    print(f"\nSaving to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.time()

    with h5py.File(str(output_path), "w") as f_out:
        # Numeric datasets
        f_out.create_dataset("actions", data=rev_actions)
        f_out.create_dataset("rel_actions", data=rev_rel_actions)
        f_out.create_dataset("robot_obs", data=rev_robot_obs)
        f_out.create_dataset("force_torque", data=rev_force_torque)

        # Fixed-length string dataset
        f_out.create_dataset("language_text", data=rev_language_text)

        # Variable-length byte datasets (images)
        vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        ds_primary_rgb = f_out.create_dataset(
            "primary_rgb", shape=(total_frames,), dtype=vlen_dtype
        )
        ds_gripper_rgb = f_out.create_dataset(
            "gripper_rgb", shape=(total_frames,), dtype=vlen_dtype
        )
        for i in tqdm(range(total_frames), desc="Writing primary_rgb"):
            ds_primary_rgb[i] = rev_primary_rgb[i]
        for i in tqdm(range(total_frames), desc="Writing gripper_rgb"):
            ds_gripper_rgb[i] = rev_gripper_rgb[i]

        if has_primary_depth:
            ds_primary_depth = f_out.create_dataset(
                "primary_depth", shape=(total_frames,), dtype=vlen_dtype
            )
            for i in tqdm(range(total_frames), desc="Writing primary_depth"):
                ds_primary_depth[i] = rev_primary_depth[i]

        if has_gripper_depth:
            ds_gripper_depth = f_out.create_dataset(
                "gripper_depth", shape=(total_frames,), dtype=vlen_dtype
            )
            for i in tqdm(range(total_frames), desc="Writing gripper_depth"):
                ds_gripper_depth[i] = rev_gripper_depth[i]

    save_elapsed = time.time() - save_start
    print(f"Saved in {save_elapsed:.1f}s")

    # --- Save metadata ---
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "source_file": str(input_path),
        "reversal_type": "time_reversal",
        "num_episodes": n_episodes,
        "total_frames": int(total_frames),
        "episode_ends": [int(x) for x in episode_ends],
        "episode_sizes": [int(x) for x in ep_sizes],
        "dt": args.dt,
        "new_language": args.new_language,
        "verify_warnings": verify_warnings,
        "reversal_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_elapsed = time.time() - load_start
    print(f"\n{'='*70}")
    print(f"  Time Reversal Complete")
    print(f"{'='*70}")
    print(f"  Episodes:  {n_episodes}")
    print(f"  Frames:    {total_frames}")
    print(f"  Warnings:  {verify_warnings}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Output:    {output_path}")
    print(f"  Metadata:  {meta_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
