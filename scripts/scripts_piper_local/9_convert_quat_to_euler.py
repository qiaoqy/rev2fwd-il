#!/usr/bin/env python3
"""Script 9: Convert quaternion-based episode data to Euler angle representation.

This script processes episode data collected by script 1 (1_teleop_ps5_controller.py)
and converts any quaternion-based orientation data to Euler angle (XYZ) representation.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================
Script 1 saves poses as:
    ee_pose: (T, 7) [x, y, z, qw, qx, qy, qz]           (quaternion)
    action:  (T, 8) [dx, dy, dz, dqw, dqx, dqy, dqz, gripper]  (quaternion delta)

This script converts them to:
    ee_pose: (T, 6) [x, y, z, rx, ry, rz]                 (Euler XYZ in radians)
    action:  (T, 7) [dx, dy, dz, drx, dry, drz, gripper]  (Euler angle delta in radians)

Detection is automatic:
    - ee_pose with shape (T, 7) → quaternion representation, will convert
    - ee_pose with shape (T, 6) → already Euler, will skip (or copy as-is)

The converted data is saved to a new output directory with the same archive
structure (tar.gz). Images and other fields are copied unchanged.

=============================================================================
EULER ANGLE CONVENTION
=============================================================================
- Euler order: XYZ (extrinsic), matching scipy Rotation.from_euler('xyz', ...)
- Units: radians
- Quaternion convention: [qw, qx, qy, qz] (scalar-first, matching script 1)

For action deltas, we recompute them from the converted ee_pose trajectory
(same approach as script 3: action[t] = ee_pose[t+1] - ee_pose[t]), ensuring
the actions are always consistent with the Euler ee_pose observations.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Convert a dataset
python scripts/scripts_piper_local/9_convert_quat_to_euler.py \
    --input data/pickplace_piper_0226_B \
    --output data/pickplace_piper_0226_B_euler

# Dry run (check data format without converting)
python scripts/scripts_piper_local/9_convert_quat_to_euler.py \
    --input data/pickplace_piper_0226_B \
    --dry_run

# Verbose output
python scripts/scripts_piper_local/9_convert_quat_to_euler.py \
    --input data/pickplace_piper_0226_B \
    --output data/pickplace_piper_0226_B_euler \
    --verbose
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# =============================================================================
# Angle Wrapping Utility
# =============================================================================

def wrap_angle(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Quaternion/Euler Conversion
# =============================================================================

def quat_to_euler_array(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion array to Euler XYZ angles.

    Args:
        quat: (T, 4) array of [qw, qx, qy, qz] (scalar-first)

    Returns:
        (T, 3) array of [rx, ry, rz] in radians
    """
    # scipy expects [qx, qy, qz, qw] (scalar-last)
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)
    euler = rot.as_euler('xyz')  # radians
    return euler.astype(np.float32)


def quat_delta_to_euler_delta(quat_delta: np.ndarray) -> np.ndarray:
    """Convert a single quaternion delta to Euler angle delta.

    Args:
        quat_delta: (4,) array of [dqw, dqx, dqy, dqz]

    Returns:
        (3,) array of [drx, dry, drz] in radians
    """
    # scipy expects [qx, qy, qz, qw]
    quat_xyzw = np.array([quat_delta[1], quat_delta[2], quat_delta[3], quat_delta[0]])
    rot = R.from_quat(quat_xyzw)
    euler = rot.as_euler('xyz')  # radians
    return euler.astype(np.float32)


# =============================================================================
# Detection
# =============================================================================

def detect_format(ee_pose: np.ndarray, action: np.ndarray) -> str:
    """Detect whether data uses quaternion or Euler representation.

    Args:
        ee_pose: (T, D) pose array
        action: (T, D) action array

    Returns:
        'quaternion' if ee_pose dim is 7 (and action dim is 8)
        'euler' if ee_pose dim is 6 (and action dim is 7)
        'unknown' otherwise
    """
    if ee_pose.ndim != 2 or action.ndim != 2:
        return 'unknown'

    pose_dim = ee_pose.shape[1]
    action_dim = action.shape[1]

    if pose_dim == 7 and action_dim == 8:
        return 'quaternion'
    elif pose_dim == 6 and action_dim == 7:
        return 'euler'
    else:
        return 'unknown'


# =============================================================================
# Conversion
# =============================================================================

def convert_episode(episode_data: dict, verbose: bool = False) -> Tuple[dict, str]:
    """Convert a single episode from quaternion to Euler representation.

    Args:
        episode_data: Dictionary loaded from npz with keys:
            ee_pose, action, gripper_state, timestamp, etc.
        verbose: Print detailed conversion info

    Returns:
        Tuple of (converted_data_dict, format_detected)
        If already Euler, returns data unchanged.
    """
    ee_pose = episode_data['ee_pose']
    action = episode_data['action']

    fmt = detect_format(ee_pose, action)

    if fmt == 'euler':
        if verbose:
            print("    Already in Euler format, skipping conversion")
        return episode_data, 'euler'

    if fmt == 'unknown':
        print(f"    WARNING: Unknown format (ee_pose shape={ee_pose.shape}, "
              f"action shape={action.shape}). Skipping.")
        return episode_data, 'unknown'

    # --- Quaternion → Euler conversion ---
    T = ee_pose.shape[0]

    # 1. Convert ee_pose: [x, y, z, qw, qx, qy, qz] → [x, y, z, rx, ry, rz]
    xyz = ee_pose[:, :3]  # (T, 3)
    quat = ee_pose[:, 3:]  # (T, 4) [qw, qx, qy, qz]
    euler = quat_to_euler_array(quat)  # (T, 3) [rx, ry, rz] in radians

    ee_pose_euler = np.concatenate([xyz, euler], axis=1)  # (T, 6)

    if verbose:
        print(f"    ee_pose: ({T}, 7) → ({T}, 6)")
        print(f"      xyz range: [{xyz.min(axis=0)} , {xyz.max(axis=0)}]")
        print(f"      euler range (rad): [{euler.min(axis=0)} , {euler.max(axis=0)}]")

    # 2. Recompute action deltas from converted ee_pose trajectory
    #    Same approach as script 3 (3_make_forward_data.py):
    #        action[t] = ee_pose[t+1] - ee_pose[t]  (forward difference)
    #        action[T-1] = 0  (last frame: no movement)
    #    This ensures action deltas are always consistent with ee_pose,
    #    regardless of how the original quaternion deltas were computed.
    gripper_targets = action[:, 7]  # (T,) - gripper target is last column in quat format
    action_euler = np.zeros((T, 7), dtype=np.float32)

    for t in range(T - 1):
        # Position delta: xyz[t+1] - xyz[t]
        action_euler[t, :3] = ee_pose_euler[t + 1, :3] - ee_pose_euler[t, :3]
        # Rotation delta: Euler angle subtraction with wrapping
        action_euler[t, 3:6] = wrap_angle(ee_pose_euler[t + 1, 3:6] - ee_pose_euler[t, 3:6])
        # Gripper target
        action_euler[t, 6] = gripper_targets[t]

    # Last timestep: zero position/rotation delta, keep gripper
    action_euler[T - 1, 6] = gripper_targets[T - 1]

    if verbose:
        pos_deltas = action_euler[:, :3]
        rot_deltas = action_euler[:, 3:6]
        print(f"    action: ({T}, 8) → ({T}, 7)")
        print(f"      pos delta range: [{pos_deltas.min(axis=0)} , {pos_deltas.max(axis=0)}]")
        print(f"      rot delta range (rad): [{rot_deltas.min(axis=0)} , {rot_deltas.max(axis=0)}]")

    # 3. Build output dictionary (copy all other fields unchanged)
    result = dict(episode_data)
    result['ee_pose'] = ee_pose_euler.astype(np.float32)
    result['action'] = action_euler.astype(np.float32)

    return result, 'quaternion'


# =============================================================================
# Archive I/O (reused from script 3 pattern)
# =============================================================================

def load_episode_from_archive(archive_path: Path) -> dict:
    """Load episode data from tar.gz archive.

    Args:
        archive_path: Path to tar.gz archive

    Returns:
        Dictionary containing episode data and images
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(temp_path)

        # Find the episode directory
        episode_dirs = list(temp_path.glob("episode_*"))
        if not episode_dirs:
            raise ValueError(f"No episode directory found in {archive_path}")
        episode_dir = episode_dirs[0]

        # Load npz data
        npz_path = episode_dir / "episode_data.npz"
        if not npz_path.exists():
            raise ValueError(f"No episode_data.npz found in {archive_path}")

        data = np.load(npz_path, allow_pickle=True)

        result = {
            'ee_pose': data['ee_pose'].copy(),
            'action': data['action'].copy(),
            'gripper_state': data['gripper_state'].copy(),
            'timestamp': data['timestamp'].copy(),
            'success': bool(data['success']),
            'episode_id': str(data['episode_id']),
            'num_timesteps': int(data['num_timesteps']),
        }

        # Optional fields
        for key in ('joint_angles', 'joint_torques', 'ee_force'):
            if key in data:
                result[key] = data[key].copy()

        # Load images
        fixed_cam_dir = episode_dir / "fixed_cam"
        wrist_cam_dir = episode_dir / "wrist_cam"

        result['fixed_images'] = []
        result['wrist_images'] = []

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        num_timesteps = result['num_timesteps']
        for i in range(num_timesteps):
            for cam_dir, img_list in [(fixed_cam_dir, result['fixed_images']),
                                       (wrist_cam_dir, result['wrist_images'])]:
                img_path = cam_dir / f"{i:06d}.png"
                if img_path.exists() and has_cv2:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_list.append(img)
                else:
                    img_list.append(None)

        return result


def save_episode_to_archive(episode: dict, output_dir: Path, episode_id: str) -> Path:
    """Save episode data as a compressed tar.gz archive.

    Args:
        episode: Episode dictionary with all data
        output_dir: Output directory for the archive
        episode_id: Episode identifier

    Returns:
        Path to the created archive
    """
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    output_dir.mkdir(parents=True, exist_ok=True)

    episode_name = f"episode_{episode_id}"
    archive_path = output_dir / f"{episode_name}.tar.gz"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        episode_dir = temp_path / episode_name
        episode_dir.mkdir(parents=True)

        # Create camera directories
        fixed_cam_dir = episode_dir / "fixed_cam"
        wrist_cam_dir = episode_dir / "wrist_cam"
        fixed_cam_dir.mkdir()
        wrist_cam_dir.mkdir()

        # Save images
        for i, img in enumerate(episode.get('fixed_images', [])):
            if img is not None and has_cv2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(fixed_cam_dir / f"{i:06d}.png"), img_bgr)

        for i, img in enumerate(episode.get('wrist_images', [])):
            if img is not None and has_cv2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(wrist_cam_dir / f"{i:06d}.png"), img_bgr)

        # Prepare npz data
        npz_data = {
            'ee_pose': episode['ee_pose'].astype(np.float32),
            'action': episode['action'].astype(np.float32),
            'gripper_state': episode['gripper_state'].astype(np.float32),
            'timestamp': episode['timestamp'].astype(np.float64),
            'success': episode['success'],
            'episode_id': episode_id,
            'num_timesteps': len(episode['ee_pose']),
        }

        for key in ('joint_angles', 'joint_torques', 'ee_force'):
            if key in episode and episode[key] is not None:
                npz_data[key] = np.asarray(episode[key]).astype(np.float32)

        np.savez(episode_dir / "episode_data.npz", **npz_data)

        # Create tar.gz archive
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(episode_dir, arcname=episode_name)

    return archive_path


# =============================================================================
# Batch Processing
# =============================================================================

def process_dataset(input_dir: Path, output_dir: Path,
                    dry_run: bool = False, verbose: bool = False):
    """Process all episodes in a dataset directory.

    Args:
        input_dir: Input directory containing episode_*.tar.gz files
        output_dir: Output directory for converted data
        dry_run: If True, only detect formats without converting
        verbose: Print detailed info per episode
    """
    # Find all episode archives
    archives = sorted(input_dir.glob("episode_*.tar.gz"))
    if not archives:
        print(f"No episode archives found in {input_dir}")
        return

    print(f"Found {len(archives)} episode(s) in {input_dir}")

    # Track statistics
    stats = {'quaternion': 0, 'euler': 0, 'unknown': 0, 'errors': 0}

    if dry_run:
        print("\n--- DRY RUN: Detecting data formats only ---\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for archive_path in tqdm(archives, desc="Processing episodes"):
        episode_name = archive_path.stem.replace("episode_", "")

        try:
            episode = load_episode_from_archive(archive_path)
        except Exception as e:
            print(f"\n  ERROR loading {archive_path.name}: {e}")
            stats['errors'] += 1
            continue

        fmt = detect_format(episode['ee_pose'], episode['action'])

        if dry_run:
            pose_shape = episode['ee_pose'].shape
            action_shape = episode['action'].shape
            print(f"  {archive_path.name}: {fmt} "
                  f"(ee_pose={pose_shape}, action={action_shape})")
            stats[fmt] = stats.get(fmt, 0) + 1
            continue

        if verbose:
            print(f"\n  Processing {archive_path.name} (format={fmt})")

        # Convert
        converted, detected_fmt = convert_episode(episode, verbose=verbose)
        stats[detected_fmt] = stats.get(detected_fmt, 0) + 1

        # Save
        try:
            save_episode_to_archive(converted, output_dir, episode['episode_id'])
        except Exception as e:
            print(f"\n  ERROR saving {archive_path.name}: {e}")
            stats['errors'] += 1
            continue

    # Copy metadata.json if it exists
    metadata_src = input_dir / "metadata.json"
    if metadata_src.exists() and not dry_run:
        metadata_dst = output_dir / "metadata.json"
        shutil.copy2(metadata_src, metadata_dst)
        if verbose:
            print(f"\nCopied metadata.json")

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Input directory:  {input_dir}")
    if not dry_run:
        print(f"  Output directory: {output_dir}")
    print(f"  Total episodes:   {len(archives)}")
    print(f"  Quaternion → Euler converted: {stats['quaternion']}")
    print(f"  Already Euler (copied as-is): {stats['euler']}")
    if stats['unknown'] > 0:
        print(f"  Unknown format (skipped):     {stats['unknown']}")
    if stats['errors'] > 0:
        print(f"  Errors:                       {stats['errors']}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert quaternion-based episode data to Euler angle representation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a dataset
  python scripts/scripts_piper_local/9_convert_quat_to_euler.py \\
      --input data/pickplace_piper_0226_B \\
      --output data/pickplace_piper_0226_B_euler

  # Dry run: check formats without converting
  python scripts/scripts_piper_local/9_convert_quat_to_euler.py \\
      --input data/pickplace_piper_0226_B \\
      --dry_run

  # Verbose output
  python scripts/scripts_piper_local/9_convert_quat_to_euler.py \\
      --input data/pickplace_piper_0226_B \\
      --output data/pickplace_piper_0226_B_euler \\
      --verbose
"""
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input directory containing episode_*.tar.gz files")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for converted data. "
                             "Defaults to <input>_euler if not specified.")
    parser.add_argument("--dry_run", "-n", action="store_true",
                        help="Only detect data formats, do not convert")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed conversion info per episode")

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    if args.dry_run:
        output_dir = None
    else:
        if args.output is not None:
            output_dir = Path(args.output)
        else:
            # Default: append _euler to input directory name
            output_dir = input_dir.parent / f"{input_dir.name}_euler"

        if output_dir.exists():
            existing = list(output_dir.glob("episode_*.tar.gz"))
            if existing:
                print(f"Warning: Output directory already has {len(existing)} episodes: {output_dir}")
                resp = input("Overwrite? [y/N]: ").strip().lower()
                if resp not in ('y', 'yes'):
                    print("Aborted.")
                    return

    print(f"\nInput:  {input_dir}")
    if output_dir:
        print(f"Output: {output_dir}")
    print()

    process_dataset(input_dir, output_dir, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
