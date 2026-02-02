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
# Basic usage - visualize all episodes
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --data_dir data/piper_pick_place

# Visualize teleop data (auto-detected from metadata)
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --data_dir data/teleop_data

# Visualize specific episodes
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --data_dir data/piper_pick_place --episodes 0 1 2

# Custom output path and fps
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --data_dir data/piper_pick_place --output data/viz_output.mp4 --fps 15

# Skip XYZ curves (faster rendering)
python scripts/scripts_piper_local/3_visualize_collected_data.py \
    --data_dir data/piper_pick_place --no_xyz_curves

=============================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

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

def load_episode_data(episode_dir: Path, is_teleop: bool = False) -> dict:
    """Load episode data from directory.
    
    Args:
        episode_dir: Path to episode directory (e.g., episode_0000/)
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
    if 'joint_angles' in data:
        result['joint_angles'] = data['joint_angles']
    else:
        result['joint_angles'] = np.zeros((result['num_timesteps'], 6), dtype=np.float32)
    
    if 'joint_torques' in data:
        result['joint_torques'] = data['joint_torques']
        # Check if we have non-zero torque data
        if np.any(np.abs(data['joint_torques']) > 1e-6):
            result['has_force_data'] = True
    else:
        result['joint_torques'] = np.zeros((result['num_timesteps'], 6), dtype=np.float32)
    
    if 'ee_force' in data:
        result['ee_force'] = data['ee_force']
        # Check if we have non-zero force data
        if np.any(np.abs(data['ee_force']) > 1e-6):
            result['has_force_data'] = True
    else:
        result['ee_force'] = np.zeros((result['num_timesteps'], 6), dtype=np.float32)
    
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


def get_episode_dirs(data_dir: Path, episode_indices: Optional[List[int]] = None) -> List[Path]:
    """Get list of episode directories.
    
    Args:
        data_dir: Root data directory containing episode folders.
        episode_indices: Optional list of specific episode indices to load.
        
    Returns:
        List of episode directory paths, sorted by episode index.
    """
    episode_dirs = sorted(data_dir.glob("episode_*"))
    
    if episode_indices is not None:
        episode_dirs = [
            d for d in episode_dirs 
            if int(d.name.split("_")[1]) in episode_indices
        ]
    
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
    if has_force_data and not is_teleop:
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
    
    # Force visualization (only for scripted data with force data)
    if has_force_data and not is_teleop and ee_force is not None and joint_torques is not None:
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

def create_visualization_video(
    data_dir: Path,
    output_path: Path,
    episode_indices: Optional[List[int]] = None,
    fps: int = 30,
    include_xyz_curves: bool = True,
    target_height: int = 480,
) -> None:
    """Create visualization video from collected data.
    
    Args:
        data_dir: Root data directory containing episode folders.
        output_path: Path to save the output video.
        episode_indices: Optional list of specific episode indices to visualize.
        fps: Frames per second for output video.
        include_xyz_curves: Whether to include XYZ curve plots.
        target_height: Target height for camera images.
    """
    # Load metadata to detect data type
    metadata = load_metadata(data_dir)
    is_teleop = is_teleop_data(metadata)
    
    if is_teleop:
        print("Detected TELEOP data (relative delta actions)")
        print("  - Actions will be integrated for trajectory visualization")
    else:
        print("Detected SCRIPTED data (absolute actions)")
    
    # Get episode directories
    episode_dirs = get_episode_dirs(data_dir, episode_indices)
    
    if len(episode_dirs) == 0:
        print(f"No episodes found in {data_dir}")
        return
    
    print(f"Found {len(episode_dirs)} episodes to visualize")
    
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
    
    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        try:
            episode_data = load_episode_data(episode_dir, is_teleop=is_teleop)
        except Exception as e:
            print(f"Error loading {episode_dir}: {e}")
            continue
        
        episode_id = episode_data['episode_id']
        num_timesteps = episode_data['num_timesteps']
        has_force = episode_data.get('has_force_data', False)
        
        force_str = " (with force data)" if has_force else ""
        print(f"  Episode {episode_id}: {num_timesteps} timesteps{force_str}")
        
        for t in range(num_timesteps):
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
        
        # Add a few blank frames between episodes for visual separation
        if episode_dir != episode_dirs[-1]:
            separator = create_placeholder_image(
                frame.shape[1], frame.shape[0],
                f"Episode {episode_id} Complete"
            )
            for _ in range(fps // 2):  # 0.5 second pause
                writer.append_data(separator)
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
  # Visualize all episodes (includes force data if available)
  python 3_visualize_collected_data.py --data_dir data/piper_pick_place
  
  # Visualize specific episodes
  python 3_visualize_collected_data.py --data_dir data/piper_pick_place --episodes 0 1 2
  
  # Without XYZ curves (faster, no force visualization)
  python 3_visualize_collected_data.py --data_dir data/piper_pick_place --no_xyz_curves

Note:
  - Scripted data (script 1) includes force data: joint_torques, ee_force
  - Teleop data (script 2) does NOT include force data
  - Force data is automatically visualized if available
"""
    )
    
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="data/piper_pick_place",
        help="Directory containing episode data (default: data/piper_pick_place)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: data/piper_pick_place_viz.mp4)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        nargs="+",
        default=None,
        help="Specific episode indices to visualize (default: all)"
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
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Load metadata for display
    metadata = load_metadata(data_dir)
    is_teleop = is_teleop_data(metadata)
    
    # Default output path
    if args.output is None:
        output_path = data_dir.parent / f"{data_dir.name}_viz.mp4"
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("Piper Data Visualization")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output video:   {output_path}")
    print(f"Data type:      {'TELEOP (relative delta)' if is_teleop else 'SCRIPTED (absolute)'}")
    print(f"Force data:     {'Not available (teleop)' if is_teleop else 'Will be visualized if present'}")
    print(f"FPS:            {args.fps}")
    print(f"XYZ curves:     {'Disabled' if args.no_xyz_curves else 'Enabled'}")
    print(f"Camera height:  {args.height}px")
    if args.episodes:
        print(f"Episodes:       {args.episodes}")
    else:
        print(f"Episodes:       All")
    print("=" * 60)
    print()
    
    create_visualization_video(
        data_dir=data_dir,
        output_path=output_path,
        episode_indices=args.episodes,
        fps=args.fps,
        include_xyz_curves=not args.no_xyz_curves,
        target_height=args.height,
    )


if __name__ == "__main__":
    main()
