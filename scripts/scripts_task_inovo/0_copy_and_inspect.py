#!/usr/bin/env python3
"""Script 0: Copy & Inspect Inovo (RoboKit) Data.

Copy raw data from shared storage to local workspace, enumerate episodes,
decode and validate a sample of frames, and print summary statistics.

=============================================================================
SOURCE DATA FORMAT (RoboKit DataHandler per-frame NPZ)
=============================================================================
Each .npz file encodes one timestep with:
    primary_rgb  : JPEG bytes  -> (480, 848, 3) uint8  RGB
    gripper_rgb  : JPEG bytes  -> (480, 848, 3) uint8  RGB
    primary_depth: PNG  bytes  -> (480, 848) uint16
    gripper_depth: PNG  bytes  -> (480, 848) uint16
    robot_obs    : pickle      -> (14,) float64
    actions      : pickle      -> (7,)  float64  velocity commands
    rel_actions  : pickle      -> (7,)  float64  (identical to actions)
    force_torque : pickle      -> (6,)  float64
    language_text: pickle      -> str

robot_obs breakdown (14D):
    [0:3]  tcp_pos  (m)              [3:6]  tcp_ori  (rad, Euler RPY)
    [6]    gripper_width [0..1]      [7:13] joint_states (rad)
    [13]   gripper_action {0,1}

actions breakdown (7D):
    [0:3]  v_xyz (m/s)              [3:6]  v_rpy (rad/s)
    [6]    gripper {0,1}

=============================================================================
USAGE
=============================================================================
# Copy from shared storage and inspect
python scripts/scripts_task_inovo/0_copy_and_inspect.py \
    --source /mnt/dongxu-fs1/data-hdd/geyuan/datasets/TCL/0209_tower_boby \
    --dest data/inovo_data/0209_tower_boby \
    --validate_ratio 0.05

# Inspect only (data already copied)
python scripts/scripts_task_inovo/0_copy_and_inspect.py \
    --dest data/inovo_data/0209_tower_boby \
    --skip_copy --validate_ratio 0.1

# Quick inspection (no frame validation)
python scripts/scripts_task_inovo/0_copy_and_inspect.py \
    --dest data/inovo_data/0209_tower_boby \
    --skip_copy --validate_ratio 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# RoboKit NPZ Decoder
# =============================================================================

def load_robokit_frame(path: str | Path) -> dict:
    """Load and decode a single RoboKit NPZ frame.

    Args:
        path: Path to .npz file.

    Returns:
        Dictionary with decoded fields:
        - primary_rgb: (480, 848, 3) uint8 RGB
        - gripper_rgb: (480, 848, 3) uint8 RGB
        - robot_obs: (14,) float64
        - actions: (7,) float64
        - rel_actions: (7,) float64
        - force_torque: (6,) float64
        - language_text: str
    """
    f = np.load(str(path), allow_pickle=True)

    # Decode images (stored as encoded bytes)
    primary_rgb = cv2.imdecode(
        np.frombuffer(f["primary_rgb"].item(), np.uint8), cv2.IMREAD_COLOR
    )
    if primary_rgb is not None:
        primary_rgb = cv2.cvtColor(primary_rgb, cv2.COLOR_BGR2RGB)

    gripper_rgb = cv2.imdecode(
        np.frombuffer(f["gripper_rgb"].item(), np.uint8), cv2.IMREAD_COLOR
    )
    if gripper_rgb is not None:
        gripper_rgb = cv2.cvtColor(gripper_rgb, cv2.COLOR_BGR2RGB)

    # Decode numeric fields (stored as pickled numpy arrays)
    robot_obs = pickle.loads(f["robot_obs"].item())
    actions = pickle.loads(f["actions"].item())
    rel_actions = pickle.loads(f["rel_actions"].item())
    force_torque = pickle.loads(f["force_torque"].item())
    language = pickle.loads(f["language_text"].item())

    return {
        "primary_rgb": primary_rgb,
        "gripper_rgb": gripper_rgb,
        "robot_obs": np.asarray(robot_obs, dtype=np.float64),
        "actions": np.asarray(actions, dtype=np.float64),
        "rel_actions": np.asarray(rel_actions, dtype=np.float64),
        "force_torque": np.asarray(force_torque, dtype=np.float64),
        "language_text": str(language) if not isinstance(language, str) else language,
    }


def parse_timestamp_from_filename(fname: str) -> Optional[float]:
    """Parse timestamp (seconds) from RoboKit filename.

    Filename format: MMDD_HHMMSS_microseconds.npz
    e.g., 0209_175443_722075.npz -> Feb 09, 17:54:43.722075
    """
    stem = Path(fname).stem  # e.g. "0209_175443_722075"
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        mmdd = parts[0]       # "0209"
        hhmmss = parts[1]     # "175443"
        micro = parts[2]      # "722075"
        hours = int(hhmmss[:2])
        minutes = int(hhmmss[2:4])
        seconds = int(hhmmss[4:6])
        microseconds = int(micro)
        total_seconds = hours * 3600 + minutes * 60 + seconds + microseconds / 1e6
        return total_seconds
    except (ValueError, IndexError):
        return None


# =============================================================================
# Episode Discovery
# =============================================================================

def discover_episodes(data_dir: Path) -> list[dict]:
    """Discover all episodes in the data directory.

    Each episode is a timestamped sub-directory containing per-frame .npz files.

    Args:
        data_dir: Root data directory.

    Returns:
        List of dicts with 'name', 'path', 'num_frames', 'npz_files'.
    """
    episodes = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Skip non-episode directories
        if entry.name in ("extracted", "__pycache__"):
            continue
        npz_files = sorted(entry.glob("*.npz"))
        if len(npz_files) == 0:
            continue
        episodes.append({
            "name": entry.name,
            "path": entry,
            "num_frames": len(npz_files),
            "npz_files": npz_files,
        })
    return episodes


# =============================================================================
# Validation
# =============================================================================

def validate_frame(npz_path: Path) -> tuple[bool, str]:
    """Validate a single frame by attempting to decode all fields.

    Returns:
        (is_valid, error_message)
    """
    try:
        frame = load_robokit_frame(npz_path)

        # Check images
        if frame["primary_rgb"] is None:
            return False, "primary_rgb decode failed"
        if frame["gripper_rgb"] is None:
            return False, "gripper_rgb decode failed"
        if frame["primary_rgb"].shape != (480, 848, 3):
            return False, f"primary_rgb shape {frame['primary_rgb'].shape} != (480, 848, 3)"
        if frame["gripper_rgb"].shape != (480, 848, 3):
            return False, f"gripper_rgb shape {frame['gripper_rgb'].shape} != (480, 848, 3)"

        # Check numeric fields
        if frame["robot_obs"].shape != (14,):
            return False, f"robot_obs shape {frame['robot_obs'].shape} != (14,)"
        if frame["actions"].shape != (7,):
            return False, f"actions shape {frame['actions'].shape} != (7,)"
        if frame["rel_actions"].shape != (7,):
            return False, f"rel_actions shape {frame['rel_actions'].shape} != (7,)"
        if frame["force_torque"].shape != (6,):
            return False, f"force_torque shape {frame['force_torque'].shape} != (6,)"

        # Check for NaN/Inf
        for key in ["robot_obs", "actions", "rel_actions", "force_torque"]:
            if np.any(np.isnan(frame[key])) or np.any(np.isinf(frame[key])):
                return False, f"{key} contains NaN/Inf"

        return True, ""
    except Exception as e:
        return False, str(e)


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(episodes: list[dict], sample_per_episode: int = 50) -> dict:
    """Compute dataset statistics by sampling frames from all episodes.

    Args:
        episodes: List of episode dicts from discover_episodes().
        sample_per_episode: Number of frames to sample per episode.

    Returns:
        Dictionary with statistics for each field.
    """
    all_tcp_pos = []
    all_tcp_ori = []
    all_gripper_width = []
    all_gripper_action = []
    all_actions_xyz = []
    all_actions_rpy = []
    all_actions_gripper = []
    all_force_torque = []
    all_dt = []
    all_languages = set()

    for ep in tqdm(episodes, desc="Computing statistics"):
        npz_files = ep["npz_files"]
        # Sample evenly
        indices = np.linspace(0, len(npz_files) - 1, min(sample_per_episode, len(npz_files)), dtype=int)
        indices = sorted(set(indices))

        prev_ts = None
        for idx in indices:
            try:
                frame = load_robokit_frame(npz_files[idx])
                obs = frame["robot_obs"]
                act = frame["actions"]

                all_tcp_pos.append(obs[:3])
                all_tcp_ori.append(obs[3:6])
                all_gripper_width.append(obs[6])
                all_gripper_action.append(obs[13])
                all_actions_xyz.append(act[:3])
                all_actions_rpy.append(act[3:6])
                all_actions_gripper.append(act[6])
                all_force_torque.append(frame["force_torque"])
                all_languages.add(frame["language_text"])

                # Estimate dt from filename timestamps
                ts = parse_timestamp_from_filename(npz_files[idx].name)
                if ts is not None and prev_ts is not None:
                    dt = ts - prev_ts
                    if 0.001 < dt < 1.0:  # reasonable range
                        all_dt.append(dt)
                prev_ts = ts
            except Exception:
                continue

    tcp_pos = np.array(all_tcp_pos)
    tcp_ori = np.array(all_tcp_ori)
    actions_xyz = np.array(all_actions_xyz)
    actions_rpy = np.array(all_actions_rpy)
    force_torque = np.array(all_force_torque)

    stats = {
        "tcp_pos": {
            "min": tcp_pos.min(axis=0).tolist(),
            "max": tcp_pos.max(axis=0).tolist(),
            "mean": tcp_pos.mean(axis=0).tolist(),
            "std": tcp_pos.std(axis=0).tolist(),
        },
        "tcp_ori": {
            "min": tcp_ori.min(axis=0).tolist(),
            "max": tcp_ori.max(axis=0).tolist(),
            "mean": tcp_ori.mean(axis=0).tolist(),
            "std": tcp_ori.std(axis=0).tolist(),
        },
        "gripper_width": {
            "min": float(min(all_gripper_width)),
            "max": float(max(all_gripper_width)),
            "mean": float(np.mean(all_gripper_width)),
            "std": float(np.std(all_gripper_width)),
        },
        "gripper_action_distribution": {
            "0_open": int(sum(1 for g in all_gripper_action if g == 0)),
            "1_closed": int(sum(1 for g in all_gripper_action if g == 1)),
        },
        "actions_xyz_velocity": {
            "min": actions_xyz.min(axis=0).tolist(),
            "max": actions_xyz.max(axis=0).tolist(),
            "mean": actions_xyz.mean(axis=0).tolist(),
            "std": actions_xyz.std(axis=0).tolist(),
        },
        "actions_rpy_velocity": {
            "min": actions_rpy.min(axis=0).tolist(),
            "max": actions_rpy.max(axis=0).tolist(),
            "mean": actions_rpy.mean(axis=0).tolist(),
            "std": actions_rpy.std(axis=0).tolist(),
        },
        "force_torque": {
            "min": force_torque.min(axis=0).tolist(),
            "max": force_torque.max(axis=0).tolist(),
            "mean": force_torque.mean(axis=0).tolist(),
            "std": force_torque.std(axis=0).tolist(),
        },
        "estimated_dt": {
            "mean": float(np.mean(all_dt)) if all_dt else None,
            "std": float(np.std(all_dt)) if all_dt else None,
            "estimated_fps": float(1.0 / np.mean(all_dt)) if all_dt else None,
        },
        "language_texts": list(all_languages),
        "samples_used": len(all_tcp_pos),
    }
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Copy & inspect Inovo (RoboKit) data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/dongxu-fs1/data-hdd/geyuan/datasets/TCL/0209_tower_boby",
        help="Source directory (shared storage). Only used for copying.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/inovo_data/0209_tower_boby",
        help="Destination directory (local workspace).",
    )
    parser.add_argument(
        "--skip_copy",
        action="store_true",
        help="Skip copying data (assume already present at --dest).",
    )
    parser.add_argument(
        "--validate_ratio",
        type=float,
        default=0.05,
        help="Fraction of frames to validate per episode (0 = skip validation). Default: 0.05.",
    )
    parser.add_argument(
        "--stats_sample",
        type=int,
        default=50,
        help="Number of frames to sample per episode for statistics. Default: 50.",
    )

    args = parser.parse_args()
    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    print(f"\n{'='*70}")
    print(f"  Inovo Data Copy & Inspection")
    print(f"{'='*70}")
    print(f"  Source: {source_dir}")
    print(f"  Dest:   {dest_dir}")
    print(f"{'='*70}\n")

    # =========================================================================
    # Step 1: Copy data
    # =========================================================================
    if not args.skip_copy:
        if not source_dir.exists():
            print(f"ERROR: Source directory not found: {source_dir}")
            return

        if dest_dir.exists():
            print(f"Destination already exists: {dest_dir}")
            print("  Skipping copy. Use --skip_copy to explicitly skip.")
        else:
            print(f"Copying data from {source_dir} to {dest_dir} ...")
            start = time.time()
            shutil.copytree(str(source_dir), str(dest_dir))
            elapsed = time.time() - start
            size_gb = sum(
                f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()
            ) / (1024**3)
            print(f"  Copied {size_gb:.2f} GB in {elapsed:.1f}s")
    else:
        print("Skipping copy (--skip_copy flag).")

    if not dest_dir.exists():
        print(f"ERROR: Destination directory not found: {dest_dir}")
        return

    # =========================================================================
    # Step 2: Discover episodes
    # =========================================================================
    print(f"\nDiscovering episodes in {dest_dir} ...")
    episodes = discover_episodes(dest_dir)
    print(f"  Found {len(episodes)} episodes\n")

    # Summary table
    frame_counts = [ep["num_frames"] for ep in episodes]
    print(f"  {'Episode':<35s} {'Frames':>8s}")
    print(f"  {'-'*35} {'-'*8}")
    for ep in episodes:
        print(f"  {ep['name']:<35s} {ep['num_frames']:>8d}")
    print(f"  {'-'*35} {'-'*8}")
    print(f"  {'TOTAL':<35s} {sum(frame_counts):>8d}")
    print(f"  Min frames: {min(frame_counts)}, Max: {max(frame_counts)}, "
          f"Mean: {np.mean(frame_counts):.0f}")

    # =========================================================================
    # Step 3: Validate frames
    # =========================================================================
    if args.validate_ratio > 0:
        print(f"\nValidating frames (ratio={args.validate_ratio:.0%}) ...")
        total_validated = 0
        total_failed = 0
        failed_details = []

        for ep in tqdm(episodes, desc="Validating"):
            npz_files = ep["npz_files"]
            n_sample = max(1, int(len(npz_files) * args.validate_ratio))
            indices = np.linspace(0, len(npz_files) - 1, n_sample, dtype=int)
            indices = sorted(set(indices))

            for idx in indices:
                ok, err = validate_frame(npz_files[idx])
                total_validated += 1
                if not ok:
                    total_failed += 1
                    failed_details.append(
                        f"  {ep['name']}/{npz_files[idx].name}: {err}"
                    )

        print(f"\n  Validated {total_validated} frames, "
              f"{total_failed} failed ({total_failed/max(1,total_validated)*100:.1f}%)")
        if failed_details:
            print("  Failed frames:")
            for d in failed_details[:20]:
                print(d)
            if len(failed_details) > 20:
                print(f"  ... and {len(failed_details) - 20} more")
        else:
            print("  All validated frames OK!")
    else:
        print("\nSkipping frame validation (validate_ratio=0).")

    # =========================================================================
    # Step 4: Compute statistics
    # =========================================================================
    print(f"\nComputing dataset statistics (sampling {args.stats_sample} frames/episode) ...")
    stats = compute_statistics(episodes, sample_per_episode=args.stats_sample)

    print(f"\n{'='*70}")
    print("  Dataset Statistics")
    print(f"{'='*70}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {sum(frame_counts)}")
    print(f"  Samples used for stats: {stats['samples_used']}")

    if stats["estimated_dt"]["estimated_fps"] is not None:
        print(f"\n  Estimated FPS: {stats['estimated_dt']['estimated_fps']:.1f} Hz "
              f"(dt_mean={stats['estimated_dt']['mean']*1000:.1f}ms, "
              f"dt_std={stats['estimated_dt']['std']*1000:.1f}ms)")

    print(f"\n  TCP Position (m):")
    for i, label in enumerate(["x", "y", "z"]):
        print(f"    {label}: [{stats['tcp_pos']['min'][i]:.4f}, {stats['tcp_pos']['max'][i]:.4f}] "
              f"mean={stats['tcp_pos']['mean'][i]:.4f} std={stats['tcp_pos']['std'][i]:.4f}")

    print(f"\n  TCP Orientation (rad):")
    for i, label in enumerate(["roll", "pitch", "yaw"]):
        print(f"    {label}: [{stats['tcp_ori']['min'][i]:.4f}, {stats['tcp_ori']['max'][i]:.4f}] "
              f"mean={stats['tcp_ori']['mean'][i]:.4f} std={stats['tcp_ori']['std'][i]:.4f}")

    print(f"\n  Gripper Width: [{stats['gripper_width']['min']:.3f}, "
          f"{stats['gripper_width']['max']:.3f}] "
          f"mean={stats['gripper_width']['mean']:.3f}")
    ga = stats["gripper_action_distribution"]
    print(f"  Gripper Action: open={ga['0_open']}, closed={ga['1_closed']}")

    print(f"\n  Action Velocity XYZ (m/s):")
    for i, label in enumerate(["vx", "vy", "vz"]):
        print(f"    {label}: [{stats['actions_xyz_velocity']['min'][i]:.4f}, "
              f"{stats['actions_xyz_velocity']['max'][i]:.4f}] "
              f"mean={stats['actions_xyz_velocity']['mean'][i]:.4f} "
              f"std={stats['actions_xyz_velocity']['std'][i]:.4f}")

    print(f"\n  Action Velocity RPY (rad/s):")
    for i, label in enumerate(["v_roll", "v_pitch", "v_yaw"]):
        print(f"    {label}: [{stats['actions_rpy_velocity']['min'][i]:.4f}, "
              f"{stats['actions_rpy_velocity']['max'][i]:.4f}] "
              f"mean={stats['actions_rpy_velocity']['mean'][i]:.4f} "
              f"std={stats['actions_rpy_velocity']['std'][i]:.4f}")

    print(f"\n  Language texts: {stats['language_texts']}")
    print(f"{'='*70}\n")

    # =========================================================================
    # Step 5: Save inspection report
    # =========================================================================
    report = {
        "inspection_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": str(source_dir),
        "num_episodes": len(episodes),
        "total_frames": sum(frame_counts),
        "episodes": [
            {"name": ep["name"], "num_frames": ep["num_frames"]}
            for ep in episodes
        ],
        "statistics": stats,
    }

    report_path = dest_dir / "inspection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Inspection report saved to {report_path}")


if __name__ == "__main__":
    main()
