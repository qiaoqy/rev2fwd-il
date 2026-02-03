#!/usr/bin/env python3
"""Script 3: Visualize Collected Piper Data.

This script creates a comprehensive visualization video of collected data:
- Side-by-side camera views (front camera + wrist camera)
- XYZ curves for observation (ee_pose) and action
- FSM state indicator (for scripted data) or teleop mode indicator
- All episodes concatenated into a single MP4 video

Supports both:
- Scripted data (absolute actions with FSM states)
- Teleop data (relative delta actions - integrated for visualization)

=============================================================================
USAGE EXAMPLES
=============================================================================
# Visualize a specific episode (tar.gz archive)
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --episode data/teleop_data/episode_0000.tar.gz

# Visualize a specific episode (directory format)
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --episode data/piper_pick_place/episode_0000

# Custom output path and fps
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --episode data/teleop_data/episode_0000.tar.gz --output data/viz_output.mp4 --fps 15

# Skip XYZ curves (faster rendering)
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --episode data/teleop_data/episode_0000.tar.gz --no_xyz_curves

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# =============================================================================
# FSM State Names (from 1_collect_data_piper.py)
# =============================================================================
FSM_STATE_NAMES = {
    0: "TELEOP",  # For teleop data
    1: "IDLE",
    2: "GO_TO_HOME",
    3: "HOVER_PLATE",
    4: "LOWER_GRASP",
    5: "CLOSE_GRIP",
    6: "LIFT_OBJECT",
    7: "HOVER_PLACE",
    8: "LOWER_PLACE",
    9: "OPEN_GRIP",
    10: "LIFT_RETREAT",
    11: "RETURN_HOME",
    12: "DONE",
}


# =============================================================================
# Metadata Loading
# =============================================================================

def load_metadata(data_dir: Path) -> dict:
    """Load metadata from data directory.
    
    Args:
        data_dir: Root data directory containing metadata.json
        
    Returns:
        Metadata dictionary with defaults for missing fields
    """
    metadata_path = data_dir / "metadata.json"
    
    default_metadata = {
        'collection_type': 'scripted',
        'action_type': 'absolute',
    }
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # Merge with defaults
        default_metadata.update(metadata)
    
    return default_metadata


def is_teleop_data(metadata: dict) -> bool:
    """Check if the data is from teleoperation (relative actions)."""
    return (
        metadata.get('collection_type') == 'teleop' or 
        metadata.get('action_type') == 'relative_delta'
    )


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_episode_data(episode_path: Union[Path, str], is_teleop: bool = False) -> dict:
    """Load episode data from directory or tar.gz archive.
    
    Supports two formats:
    - Directory format: episode_XXXX/ (from script 1)
    - Archive format: episode_XXXX.tar.gz (from script 2)
    
    Args:
        episode_path: Path to episode directory or tar.gz archive
        is_teleop: Whether this is teleop data (no fsm_state, relative actions)
        
    Returns:
        Dictionary containing episode data with keys:
        - ee_pose: (T, 7) array [x, y, z, qw, qx, qy, qz]
        - action: (T, 8) array [x, y, z, qw, qx, qy, qz, gripper]
        - action_integrated: (T, 3) array - integrated XYZ for teleop data
        - gripper_state: (T,) array
        - fsm_state: (T,) array (zeros for teleop)
        - timestamp: (T,) array
        - fixed_images: list of (H, W, 3) arrays
        - wrist_images: list of (H, W, 3) arrays
        - place_pose: (7,) array (optional)
        - goal_pose: (7,) array (optional)
        - joint_angles: (T, 6) array (optional, scripted data only)
        - joint_torques: (T, 6) array (optional, scripted data only)
        - ee_force: (T, 6) array [Fx,Fy,Fz,Mx,My,Mz] (optional, scripted data only)
        - has_force_data: bool - whether force data is available
        - success: bool
        - episode_id: int
        - is_teleop: bool
    """
    episode_path = Path(episode_path)
    
    # Check if it's a tar.gz archive
    if episode_path.suffix == '.gz' and episode_path.stem.endswith('.tar'):
        return _load_episode_from_archive(episode_path, is_teleop)
    elif episode_path.is_dir():
        return _load_episode_from_directory(episode_path, is_teleop)
    else:
        raise FileNotFoundError(f"Episode not found: {episode_path}")


def _load_episode_from_archive(archive_path: Path, is_teleop: bool = False) -> dict:
    """Load episode data from tar.gz archive.
    
    Args:
        archive_path: Path to tar.gz archive (e.g., episode_0000.tar.gz)
        is_teleop: Whether this is teleop data
        
    Returns:
        Dictionary containing episode data
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(temp_path)
        
        # Find the episode directory inside the extracted content
        extracted_dirs = list(temp_path.iterdir())
        if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
            episode_dir = extracted_dirs[0]
        else:
            # Assume the episode name matches the archive name
            episode_name = archive_path.stem.replace('.tar', '')
            episode_dir = temp_path / episode_name
        
        if not episode_dir.exists():
            raise FileNotFoundError(f"Could not find episode directory in archive: {archive_path}")
        
        # Load using the directory loader
        return _load_episode_from_directory(episode_dir, is_teleop)


def _load_episode_from_directory(episode_dir: Path, is_teleop: bool = False) -> dict:
    """Load episode data from directory.
    
    Args:
        episode_dir: Path to episode directory (e.g., episode_0000/)
        is_teleop: Whether this is teleop data
        
    Returns:
        Dictionary containing episode data
    """
    npz_path = episode_dir / "episode_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Episode data not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    result = {
        'ee_pose': data['ee_pose'],
        'action': data['action'],
        'gripper_state': data['gripper_state'],
        'timestamp': data['timestamp'],
        'success': bool(data['success']),
        'episode_id': int(data['episode_id']),
        'num_timesteps': int(data['num_timesteps']),
        'is_teleop': is_teleop,
    }
    
    # Handle FSM state (may not exist in teleop data)
    if 'fsm_state' in data:
        result['fsm_state'] = data['fsm_state']
    else:
        # Create dummy fsm_state = 0 for teleop
        result['fsm_state'] = np.zeros(result['num_timesteps'], dtype=np.int32)
    
    # Handle optional fields
    if 'place_pose' in data:
        result['place_pose'] = data['place_pose']
    if 'goal_pose' in data:
        result['goal_pose'] = data['goal_pose']
    
    # Load force/torque data (only available in scripted data from script 1)
    result['has_force_data'] = False
    
    # DEBUG: Print all keys in the npz file
    print(f"[DEBUG Load] NPZ file keys: {list(data.keys())}")
    
    if 'joint_angles' in data:
        result['joint_angles'] = data['joint_angles']
        print(f"[DEBUG Load] joint_angles: shape={data['joint_angles'].shape}")
    else:
        result['joint_angles'] = np.zeros((result['num_timesteps'], 6), dtype=np.float32)
        print(f"[DEBUG Load] joint_angles: NOT FOUND in npz")
    
    if 'joint_torques' in data:
        result['joint_torques'] = data['joint_torques']
        torques = data['joint_torques']
        has_nonzero_torques = np.any(np.abs(torques) > 1e-6)
        print(f"[DEBUG Load] joint_torques: shape={torques.shape}, has_nonzero={has_nonzero_torques}")
        print(f"  torques range: [{torques.min():.6f}, {torques.max():.6f}]")
        # Check if we have non-zero torque data
        if has_nonzero_torques:
            result['has_force_data'] = True
    else:
        result['joint_torques'] = np.zeros((result['num_timesteps'], 6), dtype=np.float32)
        print(f"[DEBUG Load] joint_torques: NOT FOUND in npz")
    
    if 'ee_force' in data:
        result['ee_force'] = data['ee_force']
        force = data['ee_force']
        has_nonzero_force = np.any(np.abs(force) > 1e-6)
        print(f"[DEBUG Load] ee_force: shape={force.shape}, has_nonzero={has_nonzero_force}")
        print(f"  force range: [{force.min():.6f}, {force.max():.6f}]")
        # Check if we have non-zero force data
        if has_nonzero_force:
            result['has_force_data'] = True
    else:
        result['ee_force'] = np.zeros((result['num_timesteps'], 6), dtype=np.float32)
        print(f"[DEBUG Load] ee_force: NOT FOUND in npz")
    
    print(f"[DEBUG Load] Final has_force_data = {result['has_force_data']}")
    
    # For teleop data, integrate relative actions to get absolute trajectory
    if is_teleop:
        result['action_integrated'] = integrate_relative_actions(
            data['ee_pose'], data['action']
        )
    else:
        # For absolute actions, just use action directly
        result['action_integrated'] = data['action'][:, :3]
    
    # Load images
    fixed_cam_dir = episode_dir / "fixed_cam"
    wrist_cam_dir = episode_dir / "wrist_cam"
    
    result['fixed_images'] = []
    result['wrist_images'] = []
    
    num_timesteps = result['num_timesteps']
    
    for i in range(num_timesteps):
        # Load fixed camera image
        fixed_img_path = fixed_cam_dir / f"{i:06d}.png"
        if fixed_img_path.exists():
            img = cv2.imread(str(fixed_img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result['fixed_images'].append(img)
        else:
            result['fixed_images'].append(None)
        
        # Load wrist camera image
        wrist_img_path = wrist_cam_dir / f"{i:06d}.png"
        if wrist_img_path.exists():
            img = cv2.imread(str(wrist_img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result['wrist_images'].append(img)
        else:
            result['wrist_images'].append(None)
    
    return result


def integrate_relative_actions(ee_pose: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Integrate relative delta actions to get absolute trajectory.
    
    For teleop data, action[0:3] contains delta_xyz. We integrate these
    starting from the initial ee_pose to get the commanded trajectory.
    
    Args:
        ee_pose: (T, 7) array of end-effector poses
        action: (T, 8) array of relative actions [delta_xyz, delta_quat, gripper]
        
    Returns:
        (T, 3) array of integrated XYZ positions
    """
    T = action.shape[0]
    integrated = np.zeros((T, 3), dtype=np.float32)
    
    # Start from initial ee_pose
    integrated[0] = ee_pose[0, :3]
    
    # Cumulative sum of deltas
    delta_xyz = action[:, :3]
    integrated = ee_pose[0, :3] + np.cumsum(delta_xyz, axis=0)
    
    return integrated


def get_episode_paths(data_dir: Path, episode_indices: Optional[List[int]] = None) -> List[Path]:
    """Get list of episode paths (directories or tar.gz archives).
    
    Supports both formats:
    - Directory format: episode_XXXX/ (from script 1)
    - Archive format: episode_XXXX.tar.gz (from script 2)
    
    Args:
        data_dir: Root data directory containing episode folders/archives.
        episode_indices: Optional list of specific episode indices to load.
        
    Returns:
        List of episode paths, sorted by episode index.
    """
    # Find both directories and tar.gz archives
    episode_dirs = list(data_dir.glob("episode_*/"))  # Directories
    episode_archives = list(data_dir.glob("episode_*.tar.gz"))  # Archives
    
    # Combine and deduplicate (prefer archives if both exist)
    episode_map = {}
    
    for d in episode_dirs:
        try:
            idx = int(d.name.split("_")[1])
            episode_map[idx] = d
        except (IndexError, ValueError):
            continue
    
    for a in episode_archives:
        try:
            # episode_0000.tar.gz -> 0000
            idx = int(a.stem.replace('.tar', '').split("_")[1])
            episode_map[idx] = a  # Archives override directories
        except (IndexError, ValueError):
            continue
    
    # Filter by indices if specified
    if episode_indices is not None:
        episode_map = {k: v for k, v in episode_map.items() if k in episode_indices}
    
    # Sort by index and return paths
    return [episode_map[k] for k in sorted(episode_map.keys())]


# Keep old function name for backward compatibility
def get_episode_dirs(data_dir: Path, episode_indices: Optional[List[int]] = None) -> List[Path]:
    """Deprecated: Use get_episode_paths instead."""
    return get_episode_paths(data_dir, episode_indices)
    
    return episode_dirs


# =============================================================================
# Visualization Functions
# =============================================================================

def create_xyz_curve_frame(
    ee_pose_history: np.ndarray,
    action_history: np.ndarray,
    action_integrated: np.ndarray,
    current_t: int,
    total_t: int,
    episode_id: int,
    fsm_state: int,
    is_teleop: bool = False,
    joint_torques: np.ndarray = None,
    ee_force: np.ndarray = None,
    has_force_data: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
) -> np.ndarray:
    """Create a single frame showing XYZ curves and optionally force data.
    
    Args:
        ee_pose_history: (T, 7) array of EE poses up to current timestep
        action_history: (T, 8) array of actions up to current timestep
        action_integrated: (T, 3) array of integrated action XYZ (for teleop)
        current_t: Current timestep index
        total_t: Total timesteps in episode
        episode_id: Episode ID for title
        fsm_state: Current FSM state
        is_teleop: Whether this is teleop data (relative actions)
        joint_torques: (T, 6) array of joint torques (optional)
        ee_force: (T, 6) array of EE force/torque (optional)
        has_force_data: Whether force data is available
        figsize: Figure size in inches
        dpi: Dots per inch for rendering
        
    Returns:
        RGB image array of shape (H, W, 3)
    """
    # Use 3x2 grid if force data available, otherwise 2x2
    if has_force_data:
        fig, axes = plt.subplots(3, 2, figsize=(figsize[0], figsize[1] * 1.5), dpi=dpi)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    
    fsm_name = FSM_STATE_NAMES.get(fsm_state, f"UNKNOWN({fsm_state})")
    mode_str = "TELEOP" if is_teleop else "SCRIPTED"
    force_str = " | FORCE" if has_force_data else ""
    fig.suptitle(
        f"Episode {episode_id} | Frame {current_t}/{total_t} | {mode_str}{force_str} | {fsm_name}",
        fontsize=12, fontweight='bold'
    )
    
    colors = {'x': 'r', 'y': 'g', 'z': 'b'}
    labels = ['X', 'Y', 'Z']
    timesteps = np.arange(current_t + 1)
    
    # Extract XYZ from ee_pose
    ee_xyz = ee_pose_history[:current_t + 1, :3]  # (t+1, 3)
    gripper = action_history[:current_t + 1, 7]  # (t+1,)
    
    # For teleop: use integrated action; for scripted: use raw action
    if is_teleop:
        action_xyz = action_integrated[:current_t + 1]  # (t+1, 3)
        delta_xyz = action_history[:current_t + 1, :3]  # (t+1, 3) - raw deltas
    else:
        action_xyz = action_history[:current_t + 1, :3]  # (t+1, 3)
        delta_xyz = None
    
    # Subplot 1: Observation XYZ (ee_pose)
    ax = axes[0, 0]
    ax.set_title("Observation: EE Pose XYZ", fontsize=10)
    for i, (label, color) in enumerate(zip(labels, colors.values())):
        ax.plot(timesteps, ee_xyz[:, i], color=color, label=label, linewidth=1.5)
    ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position (m)")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, total_t)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Action XYZ (integrated for teleop)
    ax = axes[0, 1]
    if is_teleop:
        ax.set_title("Action: Integrated Target XYZ", fontsize=10)
    else:
        ax.set_title("Action: Target XYZ", fontsize=10)
    for i, (label, color) in enumerate(zip(labels, colors.values())):
        ax.plot(timesteps, action_xyz[:, i], color=color, label=label, linewidth=1.5)
    ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position (m)")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, total_t)
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Different content for teleop vs scripted
    ax = axes[1, 0]
    if is_teleop:
        # For teleop: show raw delta actions
        ax.set_title("Action: Raw Delta XYZ", fontsize=10)
        for i, (label, color) in enumerate(zip(labels, colors.values())):
            ax.plot(timesteps, delta_xyz[:, i], color=color, label=f"Δ{label}", linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Delta (m)")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, total_t)
        ax.grid(True, alpha=0.3)
    else:
        # For scripted: Obs vs Action comparison
        ax.set_title("Obs vs Action Comparison", fontsize=10)
        ax.plot(timesteps, ee_xyz[:, 0], color='r', label='Obs X', linewidth=1.5)
        ax.plot(timesteps, action_xyz[:, 0], color='r', linestyle='--', label='Act X', linewidth=1.5)
        ax.plot(timesteps, ee_xyz[:, 1], color='g', label='Obs Y', linewidth=1.5)
        ax.plot(timesteps, action_xyz[:, 1], color='g', linestyle='--', label='Act Y', linewidth=1.5)
        ax.plot(timesteps, ee_xyz[:, 2], color='b', label='Obs Z', linewidth=1.5)
        ax.plot(timesteps, action_xyz[:, 2], color='b', linestyle='--', label='Act Z', linewidth=1.5)
        ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position (m)")
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.set_xlim(0, total_t)
        ax.grid(True, alpha=0.3)
    
    # Subplot 4: Gripper state
    ax = axes[1, 1]
    ax.set_title("Gripper Action", fontsize=10)
    ax.plot(timesteps, gripper, color='purple', label='Gripper', linewidth=2)
    ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gripper (0=close, 1=open)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(0, total_t)
    ax.grid(True, alpha=0.3)
    
    # Force visualization (for both scripted and teleop data with force data)
    if has_force_data and ee_force is not None and joint_torques is not None:
        joint_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
        joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        
        # Subplot 5: Joint Torques
        ax = axes[2, 0]
        ax.set_title("Joint Torques", fontsize=10)
        torques = joint_torques[:current_t + 1]
        for i in range(6):
            ax.plot(timesteps, torques[:, i], color=joint_colors[i], 
                   label=joint_labels[i], linewidth=1.2, alpha=0.8)
        ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Torque (N·m)")
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.set_xlim(0, total_t)
        ax.grid(True, alpha=0.3)
        
        # Subplot 6: End-Effector Force/Torque
        ax = axes[2, 1]
        ax.set_title("EE Force/Torque", fontsize=10)
        force = ee_force[:current_t + 1]
        force_labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        force_colors = ['r', 'g', 'b', 'r', 'g', 'b']
        force_styles = ['-', '-', '-', '--', '--', '--']
        for i in range(6):
            ax.plot(timesteps, force[:, i], color=force_colors[i], 
                   linestyle=force_styles[i], label=force_labels[i], 
                   linewidth=1.2, alpha=0.8)
        ax.axvline(x=current_t, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Force (N) / Torque (N·m)")
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.set_xlim(0, total_t)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert figure to image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3]  # RGBA -> RGB
    
    plt.close(fig)
    
    return img


def resize_image(img: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to target height while maintaining aspect ratio."""
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))


def create_placeholder_image(width: int, height: int, text: str = "No Image") -> np.ndarray:
    """Create a placeholder image with text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (50, 50, 50)  # Dark gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (150, 150, 150), thickness)
    
    return img


def add_text_overlay(img: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """Add text overlay to image."""
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    if position == "top":
        text_x = 10
        text_y = 25
    else:  # bottom
        text_x = 10
        text_y = img.shape[0] - 10
    
    # Draw background rectangle
    cv2.rectangle(
        img, 
        (text_x - 5, text_y - text_size[1] - 5),
        (text_x + text_size[0] + 5, text_y + 5),
        bg_color, -1
    )
    
    # Draw text
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return img


def compose_frame(
    fixed_img: Optional[np.ndarray],
    wrist_img: Optional[np.ndarray],
    xyz_curve_img: np.ndarray,
    target_height: int = 480,
) -> np.ndarray:
    """Compose a single frame with cameras and XYZ curves.
    
    Layout:
    +-------------------+-------------------+
    |   Fixed Camera    |   Wrist Camera    |
    +-------------------+-------------------+
    |              XYZ Curves               |
    +---------------------------------------+
    
    Args:
        fixed_img: Fixed camera image (H, W, 3) or None
        wrist_img: Wrist camera image (H, W, 3) or None
        xyz_curve_img: XYZ curves image (H, W, 3)
        target_height: Target height for camera images
        
    Returns:
        Composed frame (H, W, 3)
    """
    cam_height = target_height
    
    # Resize camera images
    if fixed_img is not None:
        fixed_resized = resize_image(fixed_img, cam_height)
        fixed_resized = add_text_overlay(fixed_resized, "Front Camera")
    else:
        fixed_resized = create_placeholder_image(640, cam_height, "No Front Camera")
    
    if wrist_img is not None:
        wrist_resized = resize_image(wrist_img, cam_height)
        wrist_resized = add_text_overlay(wrist_resized, "Wrist Camera")
    else:
        wrist_resized = create_placeholder_image(640, cam_height, "No Wrist Camera")
    
    # Stack cameras horizontally
    cam_row = np.hstack([fixed_resized, wrist_resized])
    cam_width = cam_row.shape[1]
    
    # Resize XYZ curves to match camera row width
    xyz_height = int(xyz_curve_img.shape[0] * cam_width / xyz_curve_img.shape[1])
    xyz_resized = cv2.resize(xyz_curve_img, (cam_width, xyz_height))
    
    # Stack vertically
    frame = np.vstack([cam_row, xyz_resized])
    
    return frame


def compose_frame_simple(
    fixed_img: Optional[np.ndarray],
    wrist_img: Optional[np.ndarray],
    episode_id: int,
    current_t: int,
    total_t: int,
    fsm_state: int,
    is_teleop: bool = False,
    target_height: int = 480,
) -> np.ndarray:
    """Compose a simple frame with just cameras (no XYZ curves).
    
    Layout:
    +-------------------+-------------------+
    |   Fixed Camera    |   Wrist Camera    |
    +-------------------+-------------------+
    """
    cam_height = target_height
    fsm_name = FSM_STATE_NAMES.get(fsm_state, f"UNKNOWN({fsm_state})")
    mode_str = "TELEOP" if is_teleop else "SCRIPTED"
    
    # Resize camera images
    if fixed_img is not None:
        fixed_resized = resize_image(fixed_img, cam_height)
        fixed_resized = add_text_overlay(
            fixed_resized, 
            f"Ep{episode_id} | {current_t}/{total_t} | {mode_str} | {fsm_name}"
        )
    else:
        fixed_resized = create_placeholder_image(640, cam_height, "No Front Camera")
    
    if wrist_img is not None:
        wrist_resized = resize_image(wrist_img, cam_height)
        wrist_resized = add_text_overlay(wrist_resized, "Wrist Camera")
    else:
        wrist_resized = create_placeholder_image(640, cam_height, "No Wrist Camera")
    
    # Stack cameras horizontally
    frame = np.hstack([fixed_resized, wrist_resized])
    
    return frame


# =============================================================================
# Main Visualization Function
# =============================================================================

def get_episode_timestamp(episode_path: Path, is_teleop: bool = False) -> Optional[str]:
    """Extract collection timestamp from episode data.
    
    Args:
        episode_path: Path to episode directory or tar.gz archive
        is_teleop: Whether this is teleop data
        
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS, or None if not available
    """
    try:
        episode_data = load_episode_data(episode_path, is_teleop=is_teleop)
        timestamps = episode_data.get('timestamp')
        if timestamps is not None and len(timestamps) > 0:
            # Use the first timestamp (episode start time)
            # Timestamps are typically in seconds since epoch
            first_ts = timestamps[0]
            dt = datetime.fromtimestamp(first_ts)
            return dt.strftime("%Y%m%d_%H%M%S")
    except Exception as e:
        print(f"Warning: Could not extract timestamp: {e}")
    return None


def create_visualization_video(
    episode_path: Path,
    output_path: Path,
    fps: int = 30,
    include_xyz_curves: bool = True,
    target_height: int = 480,
) -> None:
    """Create visualization video from a single episode.
    
    Args:
        episode_path: Path to episode directory or tar.gz archive.
        output_path: Path to save the output video.
        fps: Frames per second for output video.
        include_xyz_curves: Whether to include XYZ curve plots.
        target_height: Target height for camera images.
    """
    # Detect data type from parent directory metadata
    data_dir = episode_path.parent
    metadata = load_metadata(data_dir)
    is_teleop = is_teleop_data(metadata)
    
    if is_teleop:
        print("Detected TELEOP data (relative delta actions)")
        print("  - Actions will be integrated for trajectory visualization")
    else:
        print("Detected SCRIPTED data (absolute actions)")
    
    # Single episode
    episode_paths = [episode_path]
    
    print(f"Visualizing episode: {episode_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize video writer
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec='libx264',
        quality=8,
        pixelformat='yuv420p',
    )
    
    total_frames = 0
    
    # Load episode data
    try:
        episode_data = load_episode_data(episode_path, is_teleop=is_teleop)
    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        writer.close()
        return
    
    episode_id = episode_data['episode_id']
    num_timesteps = episode_data['num_timesteps']
    has_force = episode_data.get('has_force_data', False)
    
    force_str = " (with force data)" if has_force else ""
    print(f"  Episode {episode_id}: {num_timesteps} timesteps{force_str}")
    
    for t in tqdm(range(num_timesteps), desc="Rendering frames"):
        fixed_img = episode_data['fixed_images'][t]
        wrist_img = episode_data['wrist_images'][t]
        fsm_state = episode_data['fsm_state'][t]
        
        if include_xyz_curves:
            # Create XYZ curve frame (with optional force data)
            xyz_img = create_xyz_curve_frame(
                ee_pose_history=episode_data['ee_pose'],
                action_history=episode_data['action'],
                action_integrated=episode_data['action_integrated'],
                current_t=t,
                total_t=num_timesteps,
                episode_id=episode_id,
                fsm_state=fsm_state,
                is_teleop=is_teleop,
                joint_torques=episode_data.get('joint_torques'),
                ee_force=episode_data.get('ee_force'),
                has_force_data=episode_data.get('has_force_data', False),
            )
            
            # Compose final frame
            frame = compose_frame(
                fixed_img=fixed_img,
                wrist_img=wrist_img,
                xyz_curve_img=xyz_img,
                target_height=target_height,
            )
        else:
            # Simple frame without XYZ curves
            frame = compose_frame_simple(
                fixed_img=fixed_img,
                wrist_img=wrist_img,
                episode_id=episode_id,
                current_t=t,
                total_t=num_timesteps,
                fsm_state=fsm_state,
                is_teleop=is_teleop,
                target_height=target_height,
            )
        
        writer.append_data(frame)
        total_frames += 1
    
    writer.close()
    
    duration = total_frames / fps
    print(f"\nVisualization complete!")
    print(f"  Output: {output_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize collected Piper pick-and-place data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a specific episode (tar.gz archive)
  python 3_visualize_collected_data.py --episode data/teleop_data/episode_0000.tar.gz
  
  # Visualize a specific episode (directory format)
  python 3_visualize_collected_data.py --episode data/piper_pick_place/episode_0000
  
  # Without XYZ curves (faster, no force visualization)
  python 3_visualize_collected_data.py --episode data/teleop_data/episode_0000.tar.gz --no_xyz_curves

Note:
  - Both scripted (script 1) and teleop (script 2) data can include force data
  - Force data (joint_torques, ee_force) is automatically visualized if available
  - Output filename includes episode collection timestamp
"""
    )
    
    parser.add_argument(
        "--episode", "-e",
        type=str,
        required=True,
        help="Path to episode directory or tar.gz archive (e.g., data/teleop_data/episode_0000.tar.gz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: auto-generated with timestamp suffix)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS (default: 30)"
    )
    parser.add_argument(
        "--no_xyz_curves",
        action="store_true",
        help="Disable XYZ curve visualization (faster rendering)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Target height for camera images (default: 480)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    episode_path = Path(args.episode)
    
    if not episode_path.exists():
        print(f"Error: Episode not found: {episode_path}")
        return
    
    # Load metadata for display
    data_dir = episode_path.parent
    metadata = load_metadata(data_dir)
    is_teleop = is_teleop_data(metadata)
    
    # Extract episode name (without .tar.gz extension)
    if episode_path.suffix == '.gz' and episode_path.stem.endswith('.tar'):
        episode_name = episode_path.stem.replace('.tar', '')
    else:
        episode_name = episode_path.name
    
    # Get episode timestamp for output filename
    timestamp_suffix = get_episode_timestamp(episode_path, is_teleop=is_teleop)
    if timestamp_suffix is None:
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Warning: Could not extract collection timestamp, using current time")
    
    # Default output path with timestamp suffix
    if args.output is None:
        output_path = data_dir / f"{episode_name}_viz_{timestamp_suffix}.mp4"
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("Piper Data Visualization")
    print("=" * 60)
    print(f"Episode:        {episode_path}")
    print(f"Output video:   {output_path}")
    print(f"Data type:      {'TELEOP (relative delta)' if is_teleop else 'SCRIPTED (absolute)'}")
    print(f"Force data:     Will be visualized if present")
    print(f"Collection:     {timestamp_suffix}")
    print(f"FPS:            {args.fps}")
    print(f"XYZ curves:     {'Disabled' if args.no_xyz_curves else 'Enabled'}")
    print(f"Camera height:  {args.height}px")
    print("=" * 60)
    print()
    
    create_visualization_video(
        episode_path=episode_path,
        output_path=output_path,
        fps=args.fps,
        include_xyz_curves=not args.no_xyz_curves,
        target_height=args.height,
    )


if __name__ == "__main__":
    main()
