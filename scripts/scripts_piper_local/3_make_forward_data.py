#!/usr/bin/env python3
"""Step 3: Time-reverse teleop trajectories to create forward training data.

This script processes teleop trajectories from script 1 (e.g., place → pick movements)
and converts them into forward training data (pick → place movements) by reversing
the trajectories in time.

=============================================================================
CORE IDEA: TIME REVERSAL WITH RELATIVE ACTIONS
=============================================================================
For teleop data with relative (delta) actions:
1. The observed ee_pose sequence is the ground truth trajectory
2. Time reversal means playing this trajectory backwards
3. New relative actions are computed as deltas between consecutive reversed poses

Key insight:
- Original: ee_pose[t+1] = ee_pose[t] + action[t] (approximately)
- Reversed: ee_pose_rev[t] = ee_pose[T-1-t]
- New action: action_rev[t] = ee_pose_rev[t+1] - ee_pose_rev[t]
                            = ee_pose[T-2-t] - ee_pose[T-1-t]
                            = -action[T-2-t] (if perfect tracking)

We compute from the actual trajectory to handle tracking errors.

=============================================================================
INPUT/OUTPUT DATA FORMATS
=============================================================================
INPUT (from script 1_teleop_ps5_controller.py):
    Directory with tar.gz archives containing teleop trajectories
    - ee_pose: (T, 6) [x, y, z, rx, ry, rz] in meters/radians
    - action: (T, 7) [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper] in meters/radians
    - images: fixed_cam/*.png, wrist_cam/*.png

OUTPUT:
    Directory with tar.gz archives containing reversed trajectories
    Same format, but sequences are time-reversed with recomputed deltas.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage
python scripts/scripts_piper_local/3_make_forward_data.py \
    --input data/pickplace_piper_0210_B \
    --output data/pickplace_piper_0210_A_01gripper_action

python scripts/scripts_piper_local/3_make_forward_data.py \
    --input data/pickplace_piper_0226_A_ps5collected \
    --output data/pickplace_piper_0226_B_reversed

python scripts/scripts_piper_local/3_make_forward_data.py \
    --input data/pickplace_piper_0226_B \
    --output data/pickplace_piper_0226_A

python scripts/scripts_piper_local/3_make_forward_data.py \
    --input data/pickplace_piper_0221_B \
    --output data/pickplace_piper_0221_A

# With verbose output
python scripts/scripts_piper_local/3_make_forward_data.py \
    --input data/pick_place_piper \
    --output data/pick_place_piper_A \
    --verbose

# Process only successful episodes
python scripts/scripts_piper_local/3_make_forward_data.py \
    --input data/pick_place_piper \
    --output data/pick_place_piper_A \
    --success_only
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm


# =============================================================================
# Angle Wrapping Utility
# =============================================================================

def wrap_angle(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Data Loading and Saving
# =============================================================================

def load_episode_from_archive(archive_path: Path) -> dict:
    """Load episode data from tar.gz archive.
    
    Args:
        archive_path: Path to tar.gz archive (e.g., episode_*.tar.gz)
        
    Returns:
        Dictionary containing episode data
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(temp_path)
        
        # Find the episode directory (should be only one)
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
        if 'joint_angles' in data:
            result['joint_angles'] = data['joint_angles'].copy()
        if 'joint_torques' in data:
            result['joint_torques'] = data['joint_torques'].copy()
        if 'ee_force' in data:
            result['ee_force'] = data['ee_force'].copy()
        
        # Load images
        fixed_cam_dir = episode_dir / "fixed_cam"
        wrist_cam_dir = episode_dir / "wrist_cam"
        
        result['fixed_images'] = []
        result['wrist_images'] = []
        
        num_timesteps = result['num_timesteps']
        
        for i in range(num_timesteps):
            # Fixed camera image
            fixed_path = fixed_cam_dir / f"{i:06d}.png"
            if fixed_path.exists():
                import cv2
                img = cv2.imread(str(fixed_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result['fixed_images'].append(img)
            else:
                result['fixed_images'].append(None)
            
            # Wrist camera image
            wrist_path = wrist_cam_dir / f"{i:06d}.png"
            if wrist_path.exists():
                import cv2
                img = cv2.imread(str(wrist_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result['wrist_images'].append(img)
            else:
                result['wrist_images'].append(None)
        
        return result


def save_episode_to_archive(episode: dict, output_dir: Path, episode_id: str) -> Path:
    """Save episode data as a compressed tar.gz archive.
    
    Args:
        episode: Episode dictionary with all data
        output_dir: Output directory for the archive
        episode_id: Episode identifier (e.g., "20260204_164919_570878")
        
    Returns:
        Path to the created archive
    """
    import cv2
    
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
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(fixed_cam_dir / f"{i:06d}.png"), img_bgr)
        
        for i, img in enumerate(episode.get('wrist_images', [])):
            if img is not None:
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
        
        # Optional fields
        if 'joint_angles' in episode:
            npz_data['joint_angles'] = episode['joint_angles'].astype(np.float32)
        if 'joint_torques' in episode:
            npz_data['joint_torques'] = episode['joint_torques'].astype(np.float32)
        if 'ee_force' in episode:
            npz_data['ee_force'] = episode['ee_force'].astype(np.float32)
        
        # Save npz
        np.savez(episode_dir / "episode_data.npz", **npz_data)
        
        # Create tar.gz archive
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(episode_dir, arcname=episode_name)
    
    return archive_path


# =============================================================================
# Core Reversal Logic
# =============================================================================

def reverse_episode(episode: dict, verbose: bool = False) -> dict:
    """Reverse an episode in time with recomputed relative actions.
    
    The key insight is:
    - Original trajectory: ee_pose[0], ee_pose[1], ..., ee_pose[T-1]
    - Reversed trajectory: ee_pose[T-1], ee_pose[T-2], ..., ee_pose[0]
    
    For relative actions:
    - Original: action[t] ≈ ee_pose[t+1] - ee_pose[t] (commanded delta)
    - Reversed: action_rev[t] = ee_pose_rev[t+1] - ee_pose_rev[t]
                              = ee_pose[T-2-t] - ee_pose[T-1-t]
    
    We recompute actions from the actual ee_pose trajectory to ensure consistency.
    
    Args:
        episode: Original episode dictionary
        verbose: Whether to print detailed debug information
        
    Returns:
        New episode dictionary with reversed trajectory and recomputed actions
    """
    T = episode['num_timesteps']
    
    if verbose:
        print(f"  Original episode length: {T}")
    
    # Get original data
    ee_pose_orig = episode['ee_pose']  # (T, 6): [x, y, z, rx, ry, rz] in meters/radians
    gripper_state_orig = episode['gripper_state']  # (T,)
    action_orig = episode['action']  # (T, 7): [dx, dy, dz, drx, dry, drz, gripper]
    timestamp_orig = episode['timestamp']  # (T,)
    
    # Reverse all sequences in time
    ee_pose_rev = ee_pose_orig[::-1].copy()  # (T, 6)
    gripper_state_rev = gripper_state_orig[::-1].copy()  # (T,)
    
    # Reverse the original action's gripper targets (user commands: 0 or 1)
    # IMPORTANT: Do NOT use gripper_state (measured position) because it can be
    # e.g. 0.6 when gripping an object even though the command was 0 (close).
    gripper_targets_rev = action_orig[::-1, 6].copy()  # (T,) - gripper is at index 6 now
    
    # Compute new relative actions from reversed ee_pose
    # action_rev[t] should transform ee_pose_rev[t] to ee_pose_rev[t+1]
    action_rev = np.zeros((T, 7), dtype=np.float32)
    
    for t in range(T - 1):
        # Position delta: xyz[t+1] - xyz[t]
        delta_xyz = ee_pose_rev[t + 1, :3] - ee_pose_rev[t, :3]
        
        # Rotation delta: simple Euler angle subtraction with wrapping
        delta_euler = ee_pose_rev[t + 1, 3:6] - ee_pose_rev[t, 3:6]
        # Wrap to [-pi, pi] to handle angle discontinuities
        delta_euler = wrap_angle(delta_euler)
        
        action_rev[t, :3] = delta_xyz
        action_rev[t, 3:6] = delta_euler
        action_rev[t, 6] = gripper_targets_rev[t]
    
    # Last action: no movement
    action_rev[T - 1, :3] = 0.0
    action_rev[T - 1, 3:6] = 0.0  # Zero rotation delta
    action_rev[T - 1, 6] = gripper_targets_rev[T - 1]
    
    # Create relative timestamps (reset to start from 0)
    dt = np.mean(np.diff(timestamp_orig)) if T > 1 else 0.05
    timestamp_rev = np.arange(T) * dt
    
    # Build reversed episode
    result = {
        'ee_pose': ee_pose_rev,
        'action': action_rev,
        'gripper_state': gripper_state_rev,
        'timestamp': timestamp_rev,
        'success': episode['success'],
        'episode_id': episode['episode_id'],
        'num_timesteps': T,
    }
    
    # Reverse optional fields if they exist
    if 'joint_angles' in episode:
        result['joint_angles'] = episode['joint_angles'][::-1].copy()
    if 'joint_torques' in episode:
        result['joint_torques'] = episode['joint_torques'][::-1].copy()
    if 'ee_force' in episode:
        result['ee_force'] = episode['ee_force'][::-1].copy()
    
    # Reverse images
    if 'fixed_images' in episode and episode['fixed_images']:
        result['fixed_images'] = episode['fixed_images'][::-1]
    if 'wrist_images' in episode and episode['wrist_images']:
        result['wrist_images'] = episode['wrist_images'][::-1]
    
    if verbose:
        # Verify the reversal by checking some properties
        print(f"  Reversed episode length: {result['num_timesteps']}")
        print(f"  Original start pos: {ee_pose_orig[0, :3]}")
        print(f"  Reversed start pos: {ee_pose_rev[0, :3]}")
        print(f"  Original end pos: {ee_pose_orig[-1, :3]}")
        print(f"  Reversed end pos: {ee_pose_rev[-1, :3]}")
        
        # Verify that integrating reversed actions recovers the trajectory
        integrated = np.zeros((T, 3), dtype=np.float32)
        integrated[0] = ee_pose_rev[0, :3]
        for t in range(T - 1):
            integrated[t + 1] = integrated[t] + action_rev[t, :3]
        
        error = np.abs(integrated - ee_pose_rev[:, :3]).max()
        print(f"  Max position integration error: {error:.6f} m")
        
        # Verify rotation integration
        integrated_rot = np.zeros((T, 3), dtype=np.float32)
        integrated_rot[0] = ee_pose_rev[0, 3:6]
        for t in range(T - 1):
            integrated_rot[t + 1] = integrated_rot[t] + action_rev[t, 3:6]
        
        rot_error = np.abs(integrated_rot - ee_pose_rev[:, 3:6]).max()
        print(f"  Max rotation integration error: {np.rad2deg(rot_error):.4f} deg")
    
    return result


def verify_reversal(original: dict, reversed_ep: dict) -> bool:
    """Verify that the reversal is correct.
    
    Checks:
    1. Reversed start position == Original end position
    2. Reversed end position == Original start position
    3. Integrating reversed actions recovers the trajectory
    
    Args:
        original: Original episode dictionary
        reversed_ep: Reversed episode dictionary
        
    Returns:
        True if verification passes, False otherwise
    """
    T = original['num_timesteps']
    
    # Check position reversal
    orig_start = original['ee_pose'][0, :3]
    orig_end = original['ee_pose'][-1, :3]
    rev_start = reversed_ep['ee_pose'][0, :3]
    rev_end = reversed_ep['ee_pose'][-1, :3]
    
    pos_check1 = np.allclose(orig_end, rev_start, atol=1e-5)
    pos_check2 = np.allclose(orig_start, rev_end, atol=1e-5)
    
    if not pos_check1:
        print(f"  WARNING: Original end != Reversed start")
        print(f"    Original end: {orig_end}")
        print(f"    Reversed start: {rev_start}")
    
    if not pos_check2:
        print(f"  WARNING: Original start != Reversed end")
        print(f"    Original start: {orig_start}")
        print(f"    Reversed end: {rev_end}")
    
    # Check position action integration
    ee_pose_rev = reversed_ep['ee_pose']
    action_rev = reversed_ep['action']
    
    integrated = np.zeros((T, 3), dtype=np.float32)
    integrated[0] = ee_pose_rev[0, :3]
    for t in range(T - 1):
        integrated[t + 1] = integrated[t] + action_rev[t, :3]
    
    integration_error = np.abs(integrated - ee_pose_rev[:, :3]).max()
    integration_ok = integration_error < 1e-4
    
    if not integration_ok:
        print(f"  WARNING: Position integration error too high: {integration_error:.6f} m")
    
    # Check rotation action integration (Euler angle addition)
    euler_error_max = 0.0
    integrated_rot = np.zeros(3, dtype=np.float64)
    integrated_rot[:] = ee_pose_rev[0, 3:6]
    for t in range(T - 1):
        integrated_rot += action_rev[t, 3:6]
        actual_rot = ee_pose_rev[t + 1, 3:6]
        error = np.abs(wrap_angle(integrated_rot - actual_rot))
        euler_error_max = max(euler_error_max, error.max())
    
    rotation_ok = euler_error_max < np.deg2rad(1.0)  # Less than 1 degree tolerance
    if not rotation_ok:
        print(f"  WARNING: Rotation integration error too high: {np.rad2deg(euler_error_max):.4f} deg")
    
    return pos_check1 and pos_check2 and integration_ok and rotation_ok


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def get_episode_archives(input_dir: Path) -> List[Path]:
    """Get list of episode archives in the input directory.
    
    Args:
        input_dir: Input directory containing tar.gz archives
        
    Returns:
        Sorted list of archive paths
    """
    archives = list(input_dir.glob("episode_*.tar.gz"))
    # Sort by episode timestamp
    archives.sort(key=lambda p: p.stem)
    return archives


def copy_and_update_metadata(input_dir: Path, output_dir: Path, num_episodes: int) -> None:
    """Copy metadata from input and update for reversed data.
    
    Args:
        input_dir: Input directory with original metadata
        output_dir: Output directory for updated metadata
        num_episodes: Number of episodes in the output
    """
    input_metadata_path = input_dir / "metadata.json"
    output_metadata_path = output_dir / "metadata.json"
    
    if input_metadata_path.exists():
        with open(input_metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update metadata for reversed data
    metadata['source_collection_type'] = metadata.get('collection_type', 'unknown')
    metadata['collection_type'] = 'reversed_teleop'
    metadata['action_type'] = 'relative_delta'  # Same action type
    metadata['num_episodes'] = num_episodes
    metadata['reversed_from'] = str(input_dir)
    metadata['reversal_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update episode_ids (will be filled later or kept empty)
    if 'episode_ids' in metadata:
        del metadata['episode_ids']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {output_metadata_path}")


def main():
    """Main entry point for trajectory reversal."""
    parser = argparse.ArgumentParser(
        description="Time-reverse teleop trajectories to create forward training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing teleop episode archives (tar.gz files).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for reversed episode archives.",
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="Only process episodes marked as successful.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed debug information for each episode.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify each reversal is correct (default: True).",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Disable verification (faster processing).",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    do_verify = args.verify and not args.no_verify
    
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return
    
    # Get episode archives
    archives = get_episode_archives(input_dir)
    print(f"\n{'='*60}")
    print(f"Time Reversal of Teleop Trajectories")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(archives)} episode archives")
    print(f"Success only: {args.success_only}")
    print(f"Verify: {do_verify}")
    print(f"{'='*60}\n")
    
    if len(archives) == 0:
        print("ERROR: No episode archives found!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each episode
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    episode_ids = []
    
    for archive_path in tqdm(archives, desc="Reversing episodes"):
        try:
            # Load episode
            episode = load_episode_from_archive(archive_path)
            
            # Filter by success if requested
            if args.success_only and not episode['success']:
                skipped_count += 1
                if args.verbose:
                    print(f"  Skipping {archive_path.name} (not successful)")
                continue
            
            if args.verbose:
                print(f"\nProcessing: {archive_path.name}")
            
            # Reverse the episode
            reversed_ep = reverse_episode(episode, verbose=args.verbose)
            
            # Verify reversal
            if do_verify:
                if not verify_reversal(episode, reversed_ep):
                    print(f"  WARNING: Verification failed for {archive_path.name}")
            
            # Save reversed episode
            episode_id = episode['episode_id']
            save_episode_to_archive(reversed_ep, output_dir, episode_id)
            
            episode_ids.append(episode_id)
            processed_count += 1
            
        except Exception as e:
            print(f"\nERROR processing {archive_path.name}: {e}")
            failed_count += 1
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save metadata
    copy_and_update_metadata(input_dir, output_dir, processed_count)
    
    # Update metadata with episode IDs
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    metadata['episode_ids'] = episode_ids
    metadata['num_episodes'] = len(episode_ids)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Processed: {processed_count} episodes")
    print(f"Skipped: {skipped_count} episodes (not successful)")
    print(f"Failed: {failed_count} episodes")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    # Show sample of reversed data for verification
    if processed_count > 0 and args.verbose:
        print("Sample verification of first reversed episode:")
        sample_archive = list(output_dir.glob("episode_*.tar.gz"))[0]
        sample_ep = load_episode_from_archive(sample_archive)
        print(f"  Episode ID: {sample_ep['episode_id']}")
        print(f"  Timesteps: {sample_ep['num_timesteps']}")
        print(f"  Start XYZ: {sample_ep['ee_pose'][0, :3]}")
        print(f"  End XYZ: {sample_ep['ee_pose'][-1, :3]}")
        print(f"  Gripper start: {sample_ep['gripper_state'][0]:.2f}")
        print(f"  Gripper end: {sample_ep['gripper_state'][-1]:.2f}")


if __name__ == "__main__":
    main()
