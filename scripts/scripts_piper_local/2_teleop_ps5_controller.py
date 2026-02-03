#!/usr/bin/env python3
"""Script 2: PS5 Controller Teleoperation for Piper Arm.

This script allows teleoperation of the Piper robotic arm using
PS5 DualSense controller connected via Bluetooth.

=============================================================================
SETUP INSTRUCTIONS
=============================================================================
PS5 Controller Setup:
1. Put PS5 controller in pairing mode:
   - Hold PS button + Create button until LED flashes rapidly
2. On Linux, pair via Bluetooth:
   - Open Bluetooth settings and connect to "DualSense Wireless Controller"
3. Install pygame if not already installed:
   - pip install pygame

=============================================================================
PS5 CONTROLLER MAPPING
=============================================================================
| Control           | Action                                              |
|-------------------|-----------------------------------------------------|
| Left Stick Up/Down| Move end-effector in X axis (up=+X, down=-X)        |
| Left Stick L/R    | Move end-effector in Y axis (left=+Y, right=-Y)     |
| Right Stick Y     | Move end-effector in Z axis (up=+Z)                 |
| Right Stick X     | Rotate end-effector around Z (yaw)                  |
| L2/R2 Triggers    | Close/Open gripper (incremental)                    |
| L1/R1 Bumpers     | Decrease/Increase motion speed                      |
| D-pad Up/Down     | Tilt end-effector (pitch)                           |
| D-pad Left/Right  | Roll end-effector                                   |
|                   | (D-pad = 十字方向键，手柄左侧的上下左右按键)          |
| Cross (✕)         | Go to home position                                 |
| Circle (⭕) HOLD   | Gyro mode: controller tilt controls EE orientation  |
|                   |   - Controller pitch → arm RY                       |
|                   |   - Controller yaw   → arm RZ                       |
|                   |   - Controller roll  → arm RX                       |
| Triangle (△)      | Toggle gripper (open/close)                         |
| Square (▢)        | Toggle recording (with --record) / Print pose       |
|                   |   - Short press: start/stop recording               |
|                   |   - Long press (~1.5s): discard recording           |
|                   |   - 1 vibrate = started, 2 = saved, 3 = discarded   |
| Share             | Emergency stop / Resume                             |
| Options           | Re-enable arm (three-line button, top right)        |
| PS Button         | Quit program (PlayStation logo, center)             |

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage
python 2_teleop_ps5_controller.py

# Specify CAN interface
python 2_teleop_ps5_controller.py --can_interface can0

# Adjust speed and sensitivity
python 2_teleop_ps5_controller.py --speed 30 --linear_scale 0.1 --angular_scale 0.5

# With camera display
python 2_teleop_ps5_controller.py --show_camera

# Teleoperation with data recording
python 2_teleop_ps5_controller.py --record --out_dir data/teleop_data

"""

from __future__ import annotations

import argparse
import grp
import io
import json
import os
import pwd
import queue
import shutil
import subprocess
import sys
import tarfile
import time
import threading
import warnings
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# Input Group Permission Check (for PS5 controller access)
# =============================================================================

def check_and_setup_input_group():
    """Check if user is in 'input' group for controller access.
    
    If not in group, offers to add user and re-executes script with proper permissions.
    This avoids the need to manually run:
        sudo usermod -a -G input $USER
        newgrp input
    """
    username = pwd.getpwuid(os.getuid()).pw_name
    
    # Get user's current groups
    try:
        user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
        # Also add primary group
        primary_gid = pwd.getpwnam(username).pw_gid
        primary_group = grp.getgrgid(primary_gid).gr_name
        user_groups.append(primary_group)
    except KeyError:
        user_groups = []
    
    # Check if 'input' group exists
    try:
        input_gid = grp.getgrnam('input').gr_gid
    except KeyError:
        print("[Permission] 'input' group does not exist on this system")
        return  # Continue anyway, might work without it
    
    # Check if user is in input group
    if 'input' in user_groups:
        # User is in input group, but check if current process has it active
        current_groups = os.getgroups()
        if input_gid in current_groups:
            # All good, input group is active
            return
        else:
            # User is in group but current session doesn't have it active
            # Re-execute with sg to activate the group
            print("[Permission] Activating 'input' group for this session...")
            _reexec_with_input_group()
    else:
        # User is not in input group
        print("\n" + "=" * 60)
        print("[Permission] PS5 controller requires 'input' group membership")
        print("=" * 60)
        print(f"\nUser '{username}' is not in the 'input' group.")
        print("This is required for controller access without root.\n")
        
        response = input("Add user to 'input' group now? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            # Add user to input group
            print(f"\n[Permission] Running: sudo usermod -a -G input {username}")
            result = subprocess.run(
                ['sudo', 'usermod', '-a', '-G', 'input', username],
                capture_output=False
            )
            
            if result.returncode == 0:
                print("[Permission] ✓ User added to 'input' group")
                print("[Permission] Re-executing script with new group...\n")
                _reexec_with_input_group()
            else:
                print("[Permission] ✗ Failed to add user to group")
                print("[Permission] You may need to run manually:")
                print(f"    sudo usermod -a -G input {username}")
                print("    newgrp input")
                sys.exit(1)
        else:
            print("[Permission] Skipped. Controller may not work without proper permissions.")
            print("[Permission] To fix later, run:")
            print(f"    sudo usermod -a -G input {username}")
            print("    newgrp input")


def _reexec_with_input_group():
    """Re-execute the current script with 'input' group active using sg."""
    # Build the command to re-execute
    python_exe = sys.executable
    script_path = os.path.abspath(sys.argv[0])
    args = sys.argv[1:]
    
    # Use sg to run with input group
    # sg input -c "python script.py args..."
    cmd_str = f'"{python_exe}" "{script_path}"'
    if args:
        # Escape args for shell
        escaped_args = ' '.join(f'"{arg}"' for arg in args)
        cmd_str += f' {escaped_args}'
    
    # Set environment variable to prevent infinite re-exec loop
    if os.environ.get('_INPUT_GROUP_REEXEC'):
        return  # Already re-executed, don't loop
    
    os.environ['_INPUT_GROUP_REEXEC'] = '1'
    
    # Execute with sg
    os.execvp('sg', ['sg', 'input', '-c', cmd_str])


# Suppress pygame pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Set SDL environment variables before importing pygame
# This enables full DualSense support including gyro through HIDAPI
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS5", "1")

# Try to import pygame
try:
    import pygame
    from pygame import joystick
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[Error] pygame not installed. Please install: pip install pygame")

# pydualsense removed - using RawIMUHandler for gyro instead

# Try to import OpenCV for camera display
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Piper SDK
from piper_sdk import C_PiperInterface_V2

# Scipy for rotations
from scipy.spatial.transform import Rotation as R

# RoboKit
from robokit.controllers.imu_control import RawIMUHandler

# Force estimator
try:
    from rev2fwd_il.real import PiperForceEstimator
    FORCE_ESTIMATOR_AVAILABLE = True
except ImportError:
    FORCE_ESTIMATOR_AVAILABLE = False
    print("[Warning] Force estimator not available. Install rev2fwd_il package.")


# =============================================================================
# Constants and Parameters
# =============================================================================

# === Camera Settings ===
DEFAULT_FRONT_CAMERA = "Orbbec_Gemini_335L"  # Front camera name/ID
DEFAULT_WRIST_CAMERA = "Dabai_DC1"           # Wrist camera name/ID (-1 to disable)

# === Home Position (calibrated from script 0) ===
HOME_POSITION = (0.054, 0.0, 0.175)     # X, Y, Z in meters
HOME_ORIENTATION = (3.14, 1.2, 3.14)    # RX, RY, RZ in radians

# === Control Parameters ===
DEFAULT_SPEED_PERCENT = 20              # Default motion speed (0-100%)
MIN_SPEED_PERCENT = 5
MAX_SPEED_PERCENT = 50

# === Motion Limits ===
LINEAR_VEL_SCALE = 0.05                 # m per control cycle at full stick
ANGULAR_VEL_SCALE = 0.3                 # rad per control cycle at full stick

# === Gripper Parameters ===
GRIPPER_OPEN_ANGLE = 70.0               # Gripper open angle in degrees
GRIPPER_CLOSE_ANGLE = 0.0               # Gripper closed angle in degrees
GRIPPER_EFFORT = 500                    # Gripper force (0-1000)

# === Control Mode ===
MOVE_MODE = 0x00                        # Motion mode: 0x00=MOVE_P
ENABLE_TIMEOUT = 5.0                    # Arm enable timeout

# === Workspace Limits (Safety) ===
WORKSPACE_LIMITS = {
    "x_min": -0.3,  "x_max": 0.5,
    "y_min": -0.3,  "y_max": 0.3,
    "z_min": 0.05,  "z_max": 0.50,
}

# === PS5 Controller Button/Axis Mapping (pygame on Linux) ===
# Verified mapping for DualSense Wireless Controller on Linux
# Controller has: 6 axes, 13 buttons, 1 hat
class PS5Buttons:
    """PS5 DualSense button indices for pygame (Linux SDL mapping)."""
    CROSS = 0           # X button - confirmed working (go home)
    CIRCLE = 1          # O button
    TRIANGLE = 2        # Triangle button (was showing as Square behavior)
    SQUARE = 3          # Square button (was showing as Triangle behavior)
    L1 = 4              # Left bumper
    R1 = 5              # Right bumper
    L2_BTN = 6          # L2 as digital button (when fully pressed)
    R2_BTN = 7          # R2 as digital button (when fully pressed)
    SHARE = 8           # Create/Share button
    OPTIONS = 9         # Options/Menu button
    PS = 10             # PS button
    L3 = 11             # Left stick press
    R3 = 12             # Right stick press
    # Note: D-pad is on Hat 0, not individual buttons
    # Touchpad press may not be mapped


class PS5Axes:
    """PS5 DualSense axis indices for pygame (Linux SDL mapping)."""
    LEFT_X = 0          # Left stick horizontal (-1 left, +1 right)
    LEFT_Y = 1          # Left stick vertical (-1 up, +1 down)
    L2 = 2              # Left trigger (-1 released, +1 fully pressed)
    RIGHT_X = 3         # Right stick horizontal (-1 left, +1 right)
    RIGHT_Y = 4         # Right stick vertical (-1 up, +1 down)
    R2 = 5              # Right trigger (-1 released, +1 fully pressed)


# =============================================================================
# Coordinate Conversion Utilities
# =============================================================================

def euler_to_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> Tuple[float, float, float, float]:
    """Convert Euler XYZ (degrees) to quaternion (w, x, y, z)."""
    rot = R.from_euler('xyz', [rx_deg, ry_deg, rz_deg], degrees=True)
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    return (q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])  # (w, x, y, z)


def quat_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to Euler XYZ (degrees)."""
    rot = R.from_quat([qx, qy, qz, qw])  # scipy uses [x, y, z, w]
    return tuple(rot.as_euler('xyz', degrees=True))  # [rx, ry, rz]


# =============================================================================
# Data Recording
# =============================================================================

@dataclass
class EpisodeData:
    """Container for episode data (teleoperation)."""
    episode_id: int
    fixed_images: list  # List of (H, W, 3) uint8 arrays
    wrist_images: list  # List of (H, W, 3) uint8 arrays
    ee_pose: list       # List of (7,) arrays [x, y, z, qw, qx, qy, qz]
    gripper_state: list # List of floats (0=closed, 1=open, normalized)
    action: list        # List of (8,) arrays [delta_x, delta_y, delta_z, delta_qw, delta_qx, delta_qy, delta_qz, gripper_target]
    timestamp: list     # List of Unix timestamps
    # Force/torque data
    joint_angles: list = field(default_factory=list)   # List of (6,) arrays in radians
    joint_torques: list = field(default_factory=list)  # List of (6,) arrays in N·m
    ee_force: list = field(default_factory=list)       # List of (6,) arrays [Fx,Fy,Fz,Mx,My,Mz]
    success: bool = True


def backup_existing_directory(dir_path: Path) -> Optional[Path]:
    """Backup existing directory by renaming it with creation timestamp.
    
    Args:
        dir_path: Path to the directory to backup.
        
    Returns:
        Path to the backup directory if backup was created, None otherwise.
        Returns None and deletes the directory if it's empty or has no episode data.
    """
    if not dir_path.exists():
        return None
    
    # Check if directory is empty or only has metadata.json (no episode data)
    # Episode files are named like: episode_XXXX.tar.gz or episode_XXXX/ directories
    contents = list(dir_path.iterdir())
    episode_files = [f for f in contents if f.name.startswith("episode_")]
    
    if len(episode_files) == 0:
        # No episode data - delete the empty/metadata-only directory
        try:
            shutil.rmtree(str(dir_path))
            print(f"[Backup] Removed empty directory (no episode data): {dir_path}")
        except Exception as e:
            print(f"[Backup] Warning: Failed to remove empty directory {dir_path}: {e}")
        return None
    
    # Get directory creation time (or modification time as fallback)
    try:
        # On Linux, st_ctime is inode change time, st_mtime is more reliable
        # For backup naming, we use the earliest available timestamp
        stat_info = dir_path.stat()
        # Try birth time first (available on some systems)
        if hasattr(stat_info, 'st_birthtime'):
            creation_time = stat_info.st_birthtime
        else:
            # Use modification time as fallback
            creation_time = stat_info.st_mtime
        
        # Format timestamp for filename: YYYYMMDD_HHMMSS
        timestamp_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
    except Exception:
        # If we can't get the time, use current time
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup path
    backup_path = dir_path.parent / f"{dir_path.name}_backup_{timestamp_str}"
    
    # Handle case where backup already exists (add counter)
    counter = 1
    original_backup_path = backup_path
    while backup_path.exists():
        backup_path = dir_path.parent / f"{original_backup_path.name}_{counter}"
        counter += 1
    
    # Rename the directory
    try:
        shutil.move(str(dir_path), str(backup_path))
        print(f"[Backup] Existing directory backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"[Backup] Warning: Failed to backup {dir_path}: {e}")
        return None


def _save_episode_sync(episode: EpisodeData, out_dir: Path):
    """Save episode data as a compressed tar.gz archive (synchronous).
    
    The episode is saved as a single compressed archive file:
        episode_XXXX.tar.gz
    
    Archive structure:
        episode_XXXX/
            episode_data.npz      # Numeric data (poses, actions, etc.)
            fixed_cam/
                000000.png, 000001.png, ...
            wrist_cam/
                000000.png, 000001.png, ...
    
    Using tar.gz for ML datasets because:
    - Good compression ratio for mixed data (images + numeric)
    - Python standard library support (no extra dependencies)
    - Can be easily extracted for inspection
    - Widely compatible with data loading pipelines
    
    Note: For even better performance, consider tar.zst (Zstandard) which offers
    better compression ratio and faster decompression, but requires the 'zstandard' package.
    """
    import tempfile
    
    episode_name = f"episode_{episode.episode_id:04d}"
    archive_path = out_dir / f"{episode_name}.tar.gz"
    
    # Create a temporary directory for assembling the episode
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        episode_dir = temp_path / episode_name
        fixed_dir = episode_dir / "fixed_cam"
        wrist_dir = episode_dir / "wrist_cam"
        
        # Create directories
        fixed_dir.mkdir(parents=True, exist_ok=True)
        wrist_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        for i, (fixed_img, wrist_img) in enumerate(zip(episode.fixed_images, episode.wrist_images)):
            # Convert RGB to BGR for cv2
            if fixed_img is not None:
                cv2.imwrite(str(fixed_dir / f"{i:06d}.png"), cv2.cvtColor(fixed_img, cv2.COLOR_RGB2BGR))
            if wrist_img is not None:
                cv2.imwrite(str(wrist_dir / f"{i:06d}.png"), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
        
        # Convert lists to arrays
        T = len(episode.ee_pose)
        ee_pose_arr = np.array(episode.ee_pose, dtype=np.float32)
        gripper_arr = np.array(episode.gripper_state, dtype=np.float32)
        action_arr = np.array(episode.action, dtype=np.float32)
        timestamp_arr = np.array(episode.timestamp, dtype=np.float64)
        
        # Convert force data to arrays
        joint_angles_arr = np.array(episode.joint_angles, dtype=np.float32) if episode.joint_angles else np.zeros((T, 6), dtype=np.float32)
        joint_torques_arr = np.array(episode.joint_torques, dtype=np.float32) if episode.joint_torques else np.zeros((T, 6), dtype=np.float32)
        ee_force_arr = np.array(episode.ee_force, dtype=np.float32) if episode.ee_force else np.zeros((T, 6), dtype=np.float32)
        
        # DEBUG: Print force data stats before saving
        print(f"[DEBUG Save] Force data stats for episode {episode.episode_id}:")
        print(f"  joint_angles: shape={joint_angles_arr.shape}, has_nonzero={np.any(np.abs(joint_angles_arr) > 1e-6)}")
        print(f"  joint_torques: shape={joint_torques_arr.shape}, has_nonzero={np.any(np.abs(joint_torques_arr) > 1e-6)}")
        print(f"    torques range: [{joint_torques_arr.min():.4f}, {joint_torques_arr.max():.4f}]")
        print(f"  ee_force: shape={ee_force_arr.shape}, has_nonzero={np.any(np.abs(ee_force_arr) > 1e-6)}")
        print(f"    force range: [{ee_force_arr.min():.4f}, {ee_force_arr.max():.4f}]")
        
        # Save NPZ
        np.savez(
            episode_dir / "episode_data.npz",
            ee_pose=ee_pose_arr,
            gripper_state=gripper_arr,
            action=action_arr,
            timestamp=timestamp_arr,
            # Force/torque data
            joint_angles=joint_angles_arr,
            joint_torques=joint_torques_arr,
            ee_force=ee_force_arr,
            success=episode.success,
            episode_id=episode.episode_id,
            num_timesteps=T,
        )
        
        # Create compressed tar archive
        # Using gzip compression (compresslevel=6 is a good balance)
        with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
            # Add the episode directory to the archive
            tar.add(episode_dir, arcname=episode_name)
    
    # Calculate archive size for logging
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"[Save] Episode {episode.episode_id} saved to {archive_path} ({T} frames, {archive_size_mb:.2f} MB)")


class BackgroundSaver:
    """Background thread for saving episode data without blocking teleoperation."""
    
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = None
        self._running = False
        self._saved_count = 0
        self._pending_count = 0
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background saver thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._thread.start()
        print("[BackgroundSaver] Started")
    
    def _save_loop(self):
        """Background loop that processes save requests."""
        while self._running or not self._queue.empty():
            try:
                # Wait for a save request with timeout
                episode, out_dir = self._queue.get(timeout=0.5)
                
                # Perform the actual save
                try:
                    _save_episode_sync(episode, out_dir)
                    with self._lock:
                        self._saved_count += 1
                        self._pending_count -= 1
                except Exception as e:
                    print(f"[BackgroundSaver] Error saving episode {episode.episode_id}: {e}")
                    with self._lock:
                        self._pending_count -= 1
                
                self._queue.task_done()
                
            except queue.Empty:
                continue
    
    def save_episode(self, episode: EpisodeData, out_dir: Path):
        """Queue an episode for background saving.
        
        This method returns immediately without blocking.
        """
        with self._lock:
            self._pending_count += 1
        self._queue.put((episode, out_dir))
        print(f"[BackgroundSaver] Episode {episode.episode_id} queued for saving ({len(episode.ee_pose)} frames)")
    
    def stop(self, wait: bool = True, timeout: float = 30.0):
        """Stop the background saver.
        
        Args:
            wait: If True, wait for all pending saves to complete.
            timeout: Maximum time to wait for pending saves.
        """
        self._running = False
        
        if wait and self._thread is not None:
            # Wait for queue to be processed
            pending = self.get_pending_count()
            if pending > 0:
                print(f"[BackgroundSaver] Waiting for {pending} pending save(s)...")
            
            self._thread.join(timeout=timeout)
            
            if self._thread.is_alive():
                print(f"[BackgroundSaver] Warning: Timeout waiting for saves to complete")
            else:
                print(f"[BackgroundSaver] All saves completed")
    
    def get_saved_count(self) -> int:
        """Get the number of successfully saved episodes."""
        with self._lock:
            return self._saved_count
    
    def get_pending_count(self) -> int:
        """Get the number of episodes waiting to be saved."""
        with self._lock:
            return self._pending_count


# Legacy function for backward compatibility
def save_episode(episode: EpisodeData, out_dir: Path):
    """Save episode data (synchronous, for backward compatibility)."""
    _save_episode_sync(episode, out_dir)


def is_camera_enabled(camera_id: Union[int, str]) -> bool:
    """Check if camera is enabled (not -1 or "-1")."""
    if isinstance(camera_id, int):
        return camera_id >= 0
    if isinstance(camera_id, str):
        return camera_id.strip() != "-1"
    return False


# =============================================================================
# Piper Arm Controller
# =============================================================================

class PiperController:
    """High-level controller for Piper robotic arm."""
    
    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self.piper: Optional[Any] = None
        self._emergency_stop = False
        self._home_position = HOME_POSITION
        self._home_orientation = HOME_ORIENTATION
        self.connected = False
        self.enabled = False
        self._speed_percent = DEFAULT_SPEED_PERCENT
        
    def connect(self) -> bool:
        """Connect to the Piper arm."""
        try:
            print(f"[Piper] Connecting via {self.can_interface}...")
            self.piper = C_PiperInterface_V2(self.can_interface)
            self.piper.ConnectPort()
            time.sleep(0.5)
            self.connected = True
            print("[Piper] ✓ Connected")
            return True
        except Exception as e:
            print(f"[Piper] ✗ Connection failed: {e}")
            print(f"[Piper] If CAN interface is not UP, run:")
            print(f"       sudo ip link set {self.can_interface} up type can bitrate 1000000")
            return False
    
    def enable(self) -> bool:
        """Enable the arm."""
        if not self.connected or self.piper is None:
            print("[Piper] ✗ Not connected, cannot enable")
            return False
        
        try:
            print("[Piper] Enabling arm...")
            self.piper.EnableArm(7)
            self.piper.MotionCtrl_2(0x01, MOVE_MODE, self._speed_percent, 0)
            self.piper.GripperCtrl(0x01, 1000, 0x01, 0)
            
            start_time = time.time()
            while time.time() - start_time < ENABLE_TIMEOUT:
                if self._check_enabled():
                    self.enabled = True
                    print("[Piper] ✓ Enabled")
                    return True
                time.sleep(0.1)
            
            print("[Piper] ✗ Enable timeout")
            return False
        except Exception as e:
            print(f"[Piper] ✗ Enable failed: {e}")
            return False
    
    def _check_enabled(self) -> bool:
        """Check if arm is enabled."""
        try:
            status = self.piper.GetArmStatus()
            ctrl_mode = getattr(status.arm_status, 'ctrl_mode', None)
            ctrl_ok = ctrl_mode is not None and int(ctrl_mode) != 0
            
            enable_list = self.piper.GetArmEnableStatus()
            enable_ok = bool(enable_list) and all(enable_list)
            
            return ctrl_ok and enable_ok
        except:
            return False
    
    def disconnect(self):
        """Disable and disconnect from the arm."""
        if self.piper is not None and self.enabled:
            print("[Piper] Disabling arm...")
            try:
                self.piper.DisableArm(7)
                self.enabled = False
            except Exception as e:
                print(f"[Piper] Disable error: {e}")
        self.connected = False
        self.piper = None
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self._emergency_stop = True
        if self.piper is not None:
            try:
                self.piper.MotionCtrl_2(0x02, MOVE_MODE, 0, 0)
            except Exception as e:
                print(f"[Piper] Emergency stop error: {e}")
        print("[Piper] 🛑 EMERGENCY STOP ACTIVATED!")
    
    def clear_emergency_stop(self):
        """Clear emergency stop and resume."""
        self._emergency_stop = False
        if self.piper is not None:
            try:
                self.piper.MotionCtrl_2(0x01, MOVE_MODE, self._speed_percent, 0)
            except:
                pass
        print("[Piper] ✓ Emergency stop cleared, motion resumed")
    
    def set_speed(self, speed_percent: int):
        """Set motion speed percentage."""
        self._speed_percent = max(MIN_SPEED_PERCENT, min(MAX_SPEED_PERCENT, speed_percent))
        if self.piper is not None and self.enabled:
            try:
                self.piper.MotionCtrl_2(0x01, MOVE_MODE, self._speed_percent, 0)
            except:
                pass
        print(f"[Piper] Speed set to {self._speed_percent}%")
    
    def get_ee_pose_meters(self) -> Optional[Dict[str, float]]:
        """Get current end-effector pose in meters and radians."""
        if not self.connected or self.piper is None:
            return None
        
        try:
            end_pose = self.piper.GetArmEndPoseMsgs()
            
            x = end_pose.end_pose.X_axis / 1_000_000.0
            y = end_pose.end_pose.Y_axis / 1_000_000.0
            z = end_pose.end_pose.Z_axis / 1_000_000.0
            
            rx = np.deg2rad(end_pose.end_pose.RX_axis / 1000.0)
            ry = np.deg2rad(end_pose.end_pose.RY_axis / 1000.0)
            rz = np.deg2rad(end_pose.end_pose.RZ_axis / 1000.0)
            
            return {
                'x': x, 'y': y, 'z': z,
                'rx': rx, 'ry': ry, 'rz': rz,
                'rx_deg': np.rad2deg(rx), 'ry_deg': np.rad2deg(ry), 'rz_deg': np.rad2deg(rz),
            }
        except Exception as e:
            return None
    
    def move_to_pose(self, x: float, y: float, z: float,
                     rx: float, ry: float, rz: float) -> bool:
        """Move to specified pose (meters and radians)."""
        if not self.enabled or self.piper is None:
            return False
        
        if self._emergency_stop:
            return False
        
        # Safety clamp
        z = max(WORKSPACE_LIMITS["z_min"], min(WORKSPACE_LIMITS["z_max"], z))
        x = max(WORKSPACE_LIMITS["x_min"], min(WORKSPACE_LIMITS["x_max"], x))
        y = max(WORKSPACE_LIMITS["y_min"], min(WORKSPACE_LIMITS["y_max"], y))
        
        try:
            X = int(x * 1_000_000)
            Y = int(y * 1_000_000)
            Z = int(z * 1_000_000)
            RX = int(np.rad2deg(rx) * 1000)
            RY = int(np.rad2deg(ry) * 1000)
            RZ = int(np.rad2deg(rz) * 1000)
            
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            return True
        except Exception as e:
            print(f"[Piper] Move failed: {e}")
            return False
    
    def set_gripper_angle(self, angle_deg: float, effort: int = GRIPPER_EFFORT) -> bool:
        """Set gripper angle (0=close, 70=open)."""
        if not self.enabled or self.piper is None:
            return False
        
        if self._emergency_stop:
            return False
        
        try:
            angle_sdk = int(angle_deg * 1000)
            self.piper.GripperCtrl(angle_sdk, effort, 0x01, 0)
            return True
        except Exception as e:
            print(f"[Piper] Gripper control failed: {e}")
            return False
    
    def open_gripper(self) -> bool:
        """Fully open the gripper."""
        return self.set_gripper_angle(GRIPPER_OPEN_ANGLE)
    
    def close_gripper(self) -> bool:
        """Fully close the gripper."""
        return self.set_gripper_angle(GRIPPER_CLOSE_ANGLE)
    
    def get_gripper_angle(self) -> Optional[float]:
        """Get current gripper angle in degrees."""
        if not self.connected or self.piper is None:
            return None
        try:
            gripper_msgs = self.piper.GetArmGripperMsgs()
            return gripper_msgs.gripper_state.grippers_angle / 1000.0
        except:
            return None
    
    def go_to_home(self) -> bool:
        """Move to home position."""
        if self._emergency_stop:
            return False
        
        print("[Piper] Going to home position...")
        x, y, z = self._home_position
        rx, ry, rz = self._home_orientation
        return self.move_to_pose(x, y, z, rx, ry, rz)


# =============================================================================
# PS5 Controller Handler
# =============================================================================

class PS5Controller:
    """Handler for PS5 DualSense controller using pygame."""
    
    def __init__(self):
        self.joystick: Optional[pygame.joystick.JoystickType] = None
        self.connected = False
        self.deadzone = 0.1
        
        # Button states for edge detection
        self._prev_buttons = {}
        
    def initialize(self) -> bool:
        """Initialize pygame and find PS5 controller."""
        if not PYGAME_AVAILABLE:
            print("[Controller] ✗ pygame not available")
            return False
        
        pygame.init()
        pygame.joystick.init()
        
        # Print SDL2 version info for debugging
        print(f"[Controller] pygame version: {pygame.version.ver}")
        print(f"[Controller] SDL version: {pygame.version.SDL}")
        # SDL2 >= 2.0.14 is required for DualSense gyro support
        sdl_version = pygame.version.SDL
        if sdl_version < (2, 0, 14):
            print(f"[Controller] ⚠ SDL2 version {sdl_version} < 2.0.14, gyro may not work!")
        else:
            print(f"[Controller] ✓ SDL2 version {sdl_version} supports DualSense sensors")
        
        # Find connected joysticks
        num_joysticks = pygame.joystick.get_count()
        if num_joysticks == 0:
            print("[Controller] ✗ No joysticks found")
            print("[Controller] Make sure PS5 controller is connected via Bluetooth")
            return False
        
        print(f"[Controller] Found {num_joysticks} joystick(s):")
        
        # Look for PS5 controller
        for i in range(num_joysticks):
            js = pygame.joystick.Joystick(i)
            js.init()
            name = js.get_name()
            print(f"  [{i}] {name}")
            
            # Check if it's a PS5 controller
            if "dualsense" in name.lower() or "ps5" in name.lower() or "sony" in name.lower() or "wireless controller" in name.lower():
                self.joystick = js
                self.connected = True
                print(f"[Controller] ✓ Using: {name}")
                print(f"[Controller]   Buttons: {js.get_numbuttons()}, Axes: {js.get_numaxes()}, Hats: {js.get_numhats()}")
                return True
        
        # If no PS5 found, use first joystick
        if num_joysticks > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
            print(f"[Controller] ⚠ No PS5 controller found, using: {self.joystick.get_name()}")
            return True
        
        return False
    
    def update(self):
        """Process pygame events (call this each frame)."""
        pygame.event.pump()
    
    def get_axis(self, axis: int) -> float:
        """Get axis value with deadzone applied."""
        if not self.connected or self.joystick is None:
            return 0.0
        
        try:
            value = self.joystick.get_axis(axis)
            if abs(value) < self.deadzone:
                return 0.0
            # Apply deadzone and normalize
            sign = 1 if value > 0 else -1
            return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        except:
            return 0.0
    
    def get_button(self, button: int) -> bool:
        """Get button state."""
        if not self.connected or self.joystick is None:
            return False
        
        try:
            return self.joystick.get_button(button)
        except:
            return False
    
    def get_button_pressed(self, button: int) -> bool:
        """Check if button was just pressed (edge detection)."""
        current = self.get_button(button)
        prev = self._prev_buttons.get(button, False)
        self._prev_buttons[button] = current
        return current and not prev
    
    def get_trigger(self, trigger: int) -> float:
        """Get trigger value normalized to [0, 1]."""
        # Triggers are typically -1 to 1, convert to 0 to 1
        value = self.get_axis(trigger)
        return (value + 1.0) / 2.0
    
    def get_hat(self, hat: int = 0) -> Tuple[int, int]:
        """Get D-pad (hat) state."""
        if not self.connected or self.joystick is None:
            return (0, 0)
        
        try:
            return self.joystick.get_hat(hat)
        except:
            return (0, 0)
    
    def rumble(self, low_frequency: float = 0.5, high_frequency: float = 0.5, 
               duration_ms: int = 200):
        """Vibrate the controller.
        
        Args:
            low_frequency: Low frequency motor intensity (0.0 to 1.0)
            high_frequency: High frequency motor intensity (0.0 to 1.0)
            duration_ms: Duration in milliseconds
        """
        if not self.connected or self.joystick is None:
            return
        
        try:
            self.joystick.rumble(low_frequency, high_frequency, duration_ms)
        except Exception as e:
            pass  # Rumble not supported or failed
    
    def stop_rumble(self):
        """Stop any ongoing vibration."""
        if not self.connected or self.joystick is None:
            return
        try:
            self.joystick.stop_rumble()
        except Exception as e:
            pass
    
    def rumble_short(self, count: int = 1, intensity: float = 0.7, duration_ms: int = 100):
        """Short vibration pulses.
        
        Args:
            count: Number of pulses
            intensity: Motor intensity (0.0 to 1.0)
            duration_ms: Duration per pulse in milliseconds
        """
        if count <= 0:
            return
        
        def _do_rumble():
            gap_ms = 150  # Gap between pulses in milliseconds
            for i in range(count):
                # Start vibration
                self.rumble(intensity, intensity, duration_ms)
                # Wait for the vibration duration
                time.sleep(duration_ms / 1000.0)
                # Explicitly stop vibration
                self.stop_rumble()
                # Wait for gap before next pulse (if not last pulse)
                if i < count - 1:
                    time.sleep(gap_ms / 1000.0)
        
        # Run in background thread to not block
        threading.Thread(target=_do_rumble, daemon=True).start()
    
    def close(self):
        """Close pygame."""
        if self.joystick is not None:
            self.joystick.quit()
        pygame.quit()


# =============================================================================
# Camera Display (Optional)
# =============================================================================

class CameraCapture:
    """Camera capture for display and data recording."""
    
    def __init__(self, camera_id: Union[int, str], width: int = 640, height: int = 480, name: str = ""):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.name = name or str(camera_id)
        self.cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
    
    def _resolve_camera_source(self, camera_id: Union[int, str]):
        """Resolve camera identifier."""
        if isinstance(camera_id, int):
            return camera_id
        
        if isinstance(camera_id, str):
            cam_str = camera_id.strip()
            if cam_str.isdigit():
                return int(cam_str)
            
            if cam_str.startswith("/dev/"):
                return cam_str
            
            # Try to find by name in /dev/v4l/by-id/
            try:
                by_id_dir = Path("/dev/v4l/by-id")
                if by_id_dir.exists():
                    for p in by_id_dir.iterdir():
                        if cam_str in p.name and "video-index0" in p.name:
                            return str(p.resolve())
            except:
                pass
        
        return camera_id
    
    def start(self) -> bool:
        """Start camera capture."""
        if not CV2_AVAILABLE:
            return False
        
        source = self._resolve_camera_source(self.camera_id)
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"[Camera {self.name}] ✗ Failed to open: {source}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # Wait for first frame
        for _ in range(50):
            time.sleep(0.02)
            if self._latest_frame is not None:
                break
        
        print(f"[Camera {self.name}] ✓ Started ({self.width}x{self.height})")
        return True
    
    def _capture_loop(self):
        """Background capture loop."""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(0.01)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (BGR format)."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None
    
    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """Get latest frame (RGB format for saving)."""
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def stop(self):
        """Stop camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print(f"[Camera {self.name}] Stopped")


# =============================================================================
# Teleoperation Controller
# =============================================================================

class TeleoperationController:
    """Main teleoperation controller combining PS5 input and arm control.
    
    Optionally records data in the same format as script 1 (1_collect_data_piper.py).
    Action is stored as RELATIVE pose change (delta) per frame.
    """
    
    def __init__(self, piper: PiperController, 
                 ps5: PS5Controller,
                 linear_scale: float = LINEAR_VEL_SCALE,
                 angular_scale: float = ANGULAR_VEL_SCALE,
                 control_freq: float = 30.0,
                 # Data recording parameters
                 record_data: bool = False,
                 fixed_cam: Optional[CameraCapture] = None,
                 wrist_cam: Optional[CameraCapture] = None,
                 force_estimator: Optional["PiperForceEstimator"] = None,
                 imu_handler: Optional[RawIMUHandler] = None):
        self.piper = piper
        self.ps5 = ps5
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
        self._control_freq = control_freq
        self._control_dt = 1.0 / control_freq  # Time step for integration
        
        self._gripper_open = True
        self._gripper_target_angle = GRIPPER_OPEN_ANGLE  # Track target gripper angle
        self._gripper_speed = 3.0  # Degrees per control cycle when trigger pressed
        self._gyro_sensitivity = 1.5  # User-adjustable sensitivity multiplier
        self._running = False
        self._recorded_poses = []  # Store recorded poses
        
        # Gyro mode tracking
        self._gyro_mode_active = False

        # RoboKit IMU handler for gyroscope (use provided or create new)
        self._imu_handler = imu_handler if imu_handler is not None else RawIMUHandler()
        
        # Data recording
        self._record_data = record_data
        self._fixed_cam = fixed_cam
        self._wrist_cam = wrist_cam
        self._recording = False
        self._current_episode: Optional[EpisodeData] = None
        self._episode_count = 0
        self._prev_pose = None  # For computing delta action
        
        # Force estimator
        self._force_estimator = force_estimator
        
        # Long-press Square button tracking for discard
        self._square_press_start_time: Optional[float] = None
        self._square_long_press_triggered = False
        self._square_long_press_threshold = 1.5  # seconds to hold for discard
        self._square_rumble_interval = 0.15  # seconds between rumble pulses during hold
        
    def start_recording(self, episode_id: int = None):
        """Start recording a new episode."""
        if not self._record_data:
            print("[Teleop] ⚠ Data recording not enabled")
            return
        
        if episode_id is None:
            episode_id = self._episode_count
        
        self._current_episode = EpisodeData(
            episode_id=episode_id,
            fixed_images=[],
            wrist_images=[],
            ee_pose=[],
            gripper_state=[],
            action=[],
            timestamp=[],
            joint_angles=[],
            joint_torques=[],
            ee_force=[],
        )
        self._recording = True
        self._prev_pose = None
        # Increment episode count for next recording
        self._episode_count += 1
        print(f"[Teleop] 🔴 Recording episode {episode_id + 1}...")
        
    def stop_recording(self, success: bool = True) -> Optional[EpisodeData]:
        """Stop recording and return the episode data."""
        if not self._recording:
            return None
        
        self._recording = False
        episode = self._current_episode
        if episode is not None:
            episode.success = success
            print(f"[Teleop] ⬛ Episode {episode.episode_id + 1} stopped ({len(episode.ee_pose)} frames)")
        self._current_episode = None
        self._prev_pose = None
        return episode
    
    def _record_frame(self, pose: dict, target_x: float = None, target_y: float = None, 
                      target_z: float = None, target_rx: float = None, 
                      target_ry: float = None, target_rz: float = None):
        """Record a single frame of data during teleoperation.
        
        Args:
            pose: Current end-effector pose dict from piper.get_ee_pose_meters()
            target_*: Target pose values (if not provided, uses current pose values)
        """
        if not self._recording or self._current_episode is None:
            return
        
        # Use current pose if targets not specified
        if target_x is None:
            target_x = pose['x']
        if target_y is None:
            target_y = pose['y']
        if target_z is None:
            target_z = pose['z']
        if target_rx is None:
            target_rx = pose['rx']
        if target_ry is None:
            target_ry = pose['ry']
        if target_rz is None:
            target_rz = pose['rz']
        
        # Get current pose as observation
        current_pos = np.array([pose['x'], pose['y'], pose['z']])
        current_quat = np.array(euler_to_quat(pose['rx_deg'], pose['ry_deg'], pose['rz_deg']))
        ee_pose = np.concatenate([current_pos, current_quat])  # [x, y, z, qw, qx, qy, qz]
        
        # Get gripper state (normalized 0-1)
        gripper_angle = self.piper.get_gripper_angle() or self._gripper_target_angle
        gripper_state = gripper_angle / GRIPPER_OPEN_ANGLE  # Normalize to [0, 1]
        
        # Compute action as RELATIVE pose change (delta)
        if self._prev_pose is not None:
            delta_x = target_x - self._prev_pose['x']
            delta_y = target_y - self._prev_pose['y']
            delta_z = target_z - self._prev_pose['z']
            
            delta_rx = target_rx - self._prev_pose['rx']
            delta_ry = target_ry - self._prev_pose['ry']
            delta_rz = target_rz - self._prev_pose['rz']
            delta_quat = np.array(euler_to_quat(
                np.rad2deg(delta_rx), np.rad2deg(delta_ry), np.rad2deg(delta_rz)
            ))
        else:
            # First frame: no delta, set to zero
            delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
            delta_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        # Gripper target (normalized)
        gripper_target = self._gripper_target_angle / GRIPPER_OPEN_ANGLE
        
        action = np.array([delta_x, delta_y, delta_z,
                           delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3],
                           gripper_target], dtype=np.float32)
        
        # Get camera frames
        fixed_img = self._fixed_cam.get_frame_rgb() if self._fixed_cam else None
        wrist_img = self._wrist_cam.get_frame_rgb() if self._wrist_cam else None
        
        # Handle missing camera data
        if fixed_img is None:
            fixed_img = np.zeros((480, 640, 3), dtype=np.uint8)
        if wrist_img is None:
            wrist_img = np.zeros_like(fixed_img)
        
        # Get force/torque data
        joint_angles = np.zeros(6)
        joint_torques = np.zeros(6)
        ee_force = np.zeros(6)
        
        frame_idx = len(self._current_episode.ee_pose) if self._current_episode else -1
        
        if self._force_estimator is not None and self.piper.piper is not None:
            joint_state = self._force_estimator.get_joint_state_from_piper(self.piper.piper)
            if joint_state is not None:
                joint_angles, joint_torques = joint_state
                ee_force = self._force_estimator.estimate_force(
                    joint_angles, joint_torques,
                    use_filter=True,
                    use_gravity_comp=True
                )
                # DEBUG: Print force data on first few frames
                if frame_idx < 3:
                    print(f"[DEBUG Force] Frame {frame_idx}: SUCCESS")
                    print(f"  joint_angles: {joint_angles}")
                    print(f"  joint_torques: {joint_torques}")
                    print(f"  ee_force: {ee_force}")
            else:
                # DEBUG: joint_state is None - print for first few frames
                if frame_idx < 5:
                    print(f"[DEBUG Force] Frame {frame_idx}: joint_state is None!")
        else:
            # DEBUG: force_estimator not available
            if frame_idx < 3:
                print(f"[DEBUG Force] Frame {frame_idx}: force_estimator={self._force_estimator is not None}, piper.piper={self.piper.piper is not None}")
        
        # Record data
        self._current_episode.fixed_images.append(fixed_img)
        self._current_episode.wrist_images.append(wrist_img)
        self._current_episode.ee_pose.append(ee_pose)
        self._current_episode.gripper_state.append(gripper_state)
        self._current_episode.action.append(action)
        self._current_episode.timestamp.append(time.time())
        # Record force data
        self._current_episode.joint_angles.append(joint_angles)
        self._current_episode.joint_torques.append(joint_torques)
        self._current_episode.ee_force.append(ee_force)
        
        # Update previous pose for next delta computation
        self._prev_pose = {
            'x': target_x, 'y': target_y, 'z': target_z,
            'rx': target_rx, 'ry': target_ry, 'rz': target_rz,
        }
        
    def start(self):
        """Start teleoperation."""
        self._running = True
        print("\n" + "=" * 60)
        print("PS5 Teleoperation Active!")
        print("=" * 60)
        print("\nControls:")
        print("  Left Stick:    Move X/Y (up=+X, left=+Y)")
        print("  Right Stick Y: Move Z (up=+Z)")
        print("  Right Stick X: Rotate yaw")
        print("  L2/R2:         Close/Open gripper")
        print("  L1/R1:         Decrease/Increase speed")
        print("  D-pad:         Pitch/Roll")
        print("  Cross (X):     Go to home")
        print("  Circle HOLD:   Gyro mode (hold still briefly, then tilt)")
        print("  Triangle:      Toggle gripper")
        if self._record_data:
            print("  Square:        Short=Start/Stop recording, Long=Discard")
        else:
            print("  Square:        Print & record pose")
        print("  Share:         Emergency stop/Resume")
        print("  PS Button:     Quit")
        
        if self._record_data:
            print("\nData Recording:")
            print("  Square (short press): Start/Stop recording")
            print("    - 1 vibrate = started recording")
            print("    - 2 vibrates = stopped & saved")
            print("  Square (long press):  Discard current episode")
            print("    - continuous vibration while holding")
            print("    - 3 vibrates = discarded")
        print("=" * 60 + "\n")
    
    def step(self) -> bool:
        """Execute one control step.
        
        Returns:
            False if should quit, True otherwise.
        """
        return self._step_ps5()
    
    def _step_ps5(self) -> bool:
        """Execute one control step using PS5 controller.
        
        Returns:
            False if should quit, True otherwise.
        """
        self.ps5.update()
        
        # Check quit button (PS button)
        if self.ps5.get_button_pressed(PS5Buttons.PS):
            print("\n[Teleop] PS button pressed, quitting...")
            return False
        
        # Emergency stop toggle (Share button)
        if self.ps5.get_button_pressed(PS5Buttons.SHARE):
            if self.piper._emergency_stop:
                self.piper.clear_emergency_stop()
            else:
                self.piper.emergency_stop()
            return True
        
        # Skip control if emergency stopped
        if self.piper._emergency_stop:
            return True
        
        # Options button - reset/re-enable
        if self.ps5.get_button_pressed(PS5Buttons.OPTIONS):
            print("[Teleop] Resetting arm...")
            self.piper.enable()
            # Keep gripper open after re-enable (enable() sets gripper to closed state)
            self.piper.open_gripper()
            self._gripper_open = True
            self._gripper_target_angle = GRIPPER_OPEN_ANGLE
            return True
        
        # Cross button - go home
        if self.ps5.get_button_pressed(PS5Buttons.CROSS):
            self.piper.go_to_home()
            return True
        
        # Triangle button - toggle gripper (was Circle)
        if self.ps5.get_button_pressed(PS5Buttons.TRIANGLE):
            if self._gripper_open:
                self.piper.close_gripper()
                self._gripper_open = False
                self._gripper_target_angle = GRIPPER_CLOSE_ANGLE
                print("[Teleop] Gripper: CLOSE")
            else:
                self.piper.open_gripper()
                self._gripper_open = True
                self._gripper_target_angle = GRIPPER_OPEN_ANGLE
                print("[Teleop] Gripper: OPEN")
            return True
        
        # Square button - toggle recording (short press) or discard (long press)
        square_held = self.ps5.get_button(PS5Buttons.SQUARE)
        square_just_pressed = self.ps5.get_button_pressed(PS5Buttons.SQUARE)
        
        if self._record_data and self._recording:
            # Recording mode with active recording: handle long-press for discard
            if square_just_pressed:
                # Start tracking long press
                self._square_press_start_time = time.time()
                self._square_long_press_triggered = False
            
            if square_held and self._square_press_start_time is not None:
                hold_duration = time.time() - self._square_press_start_time
                
                if not self._square_long_press_triggered:
                    # Continuous vibration while holding (after short delay to distinguish from short press)
                    if hold_duration > 0.3:  # Start vibrating after 0.3s to avoid short-press confusion
                        # Rumble continuously by checking interval
                        if not hasattr(self, '_last_hold_rumble_time'):
                            self._last_hold_rumble_time = 0
                        if time.time() - self._last_hold_rumble_time > self._square_rumble_interval:
                            self.ps5.rumble(0.4, 0.4, int(self._square_rumble_interval * 1000))
                            self._last_hold_rumble_time = time.time()
                    
                    # Check if long press threshold reached
                    if hold_duration >= self._square_long_press_threshold:
                        # Discard recording
                        self._square_long_press_triggered = True
                        self._recording = False
                        self._current_episode = None
                        self._prev_pose = None
                        print("[Teleop] 🗑️ Episode discarded (long press)")
                        # Stop continuous vibration first, then three pulses
                        self.ps5.stop_rumble()
                        time.sleep(0.1)  # Brief pause before pulses
                        self.ps5.rumble_short(count=3, intensity=0.8, duration_ms=120)
            
            elif not square_held and self._square_press_start_time is not None:
                # Button released - check if it was a short press
                hold_duration = time.time() - self._square_press_start_time
                
                # Stop any ongoing rumble first
                self.ps5.stop_rumble()
                
                if not self._square_long_press_triggered and hold_duration < 0.3:
                    # Short press: stop recording and save
                    self._recording = False
                    print(f"[Teleop] ⬛ Stopped recording ({len(self._current_episode.ee_pose) if self._current_episode else 0} frames)")
                    # Vibrate twice to indicate stop
                    time.sleep(0.05)  # Brief pause to ensure previous rumble stopped
                    self.ps5.rumble_short(count=2, intensity=0.8, duration_ms=120)
                
                # Reset tracking
                self._square_press_start_time = None
                self._square_long_press_triggered = False
        
        elif self._record_data and not self._recording:
            # Recording mode but not recording: short press starts recording
            if square_just_pressed:
                self.start_recording()
                # Vibrate once to indicate start
                self.ps5.rumble_short(count=1, intensity=0.7, duration_ms=150)
        
        elif not self._record_data and square_just_pressed:
            # No recording mode: print and record pose (original behavior)
            pose = self.piper.get_ee_pose_meters()
            gripper = self.piper.get_gripper_angle()
            if pose:
                print(f"\n[Pose #{len(self._recorded_poses)}] Position: ({pose['x']:.4f}, {pose['y']:.4f}, {pose['z']:.4f}) m")
                print(f"[Pose] Rotation: ({pose['rx_deg']:.1f}°, {pose['ry_deg']:.1f}°, {pose['rz_deg']:.1f}°)")
                if gripper is not None:
                    print(f"[Pose] Gripper: {gripper:.1f}°")
                # Record the pose
                self._recorded_poses.append({
                    'x': pose['x'], 'y': pose['y'], 'z': pose['z'],
                    'rx': pose['rx'], 'ry': pose['ry'], 'rz': pose['rz'],
                    'gripper': gripper if gripper else 0.0
                })
                print(f"[Teleop] Pose recorded! Total: {len(self._recorded_poses)}")
        
        # Speed adjustment (L1/R1)
        if self.ps5.get_button_pressed(PS5Buttons.L1):
            self.piper.set_speed(self.piper._speed_percent - 5)
        if self.ps5.get_button_pressed(PS5Buttons.R1):
            self.piper.set_speed(self.piper._speed_percent + 5)
        
        # Get current pose
        pose = self.piper.get_ee_pose_meters()
        if pose is None:
            return True
        
        # Read stick inputs
        # Left stick: up/down -> X, left/right -> Y
        left_stick_y = -self.ps5.get_axis(PS5Axes.LEFT_Y)  # Inverted: up = positive
        left_stick_x = -self.ps5.get_axis(PS5Axes.LEFT_X)  # Inverted: left = positive
        right_x = self.ps5.get_axis(PS5Axes.RIGHT_X)       # Yaw rotation
        right_y = -self.ps5.get_axis(PS5Axes.RIGHT_Y)      # Z movement (inverted: up = +Z)
        
        # D-pad for pitch/roll (when Circle not held)
        dpad = self.ps5.get_hat(0)
        dpad_pitch = dpad[1]  # Up/Down
        dpad_roll = dpad[0]   # Left/Right
        
        # Circle button held - gyro mode for orientation control
        circle_held = self.ps5.get_button(PS5Buttons.CIRCLE)
        
        # Handle gyro mode activation/deactivation
        if circle_held and not self._gyro_mode_active:
            # Just entered gyro mode - reset IMU pose
            self._gyro_mode_active = True
            self._imu_handler.reset_pose() 
            # debug the vr gyro instead
            # self._quest_handler.reset_pose()

            print("[Gyro] 🎯 Entering gyro mode - IMU pose reset")
        elif not circle_held and self._gyro_mode_active:
            # Exited gyro mode
            self._gyro_mode_active = False
        
        # Debug: print gyro status when Circle is first pressed
        if circle_held and not getattr(self, '_gyro_debug_printed', False):
            print("[Teleop] 🎮 Gyro mode activated - using RawIMUHandler")
            print(f"[Teleop] ✓ IMU ready! Tilt controller to rotate end-effector")
            self._gyro_debug_printed = True
        elif not circle_held:
            self._gyro_debug_printed = False
        
        # Calculate target pose
        # Left stick up/down -> X axis, left/right -> Y axis
        target_x = pose['x'] + left_stick_y * self.linear_scale  # Up = +X
        target_y = pose['y'] + left_stick_x * self.linear_scale  # Left = +Y
        target_z = pose['z'] + right_y * self.linear_scale       # Up = +Z
        
        # Rotation control
        if circle_held:
            # Gyro mode: use RawIMUHandler for orientation control
            # get_latest_euler returns: {'euler': [rpy_real(3), rpy_rel(3)], 'quat': [4]}

            # debug the vr gyro instead
            imu_data = self._imu_handler.get_latest_euler() 
            euler = imu_data['euler']  # [roll_real, pitch_real, yaw_real, roll_rel, pitch_rel, yaw_rel] in degrees
            
            # Use relative euler angles (change since last query)
            # euler[3:6] = rpy_rel (degrees)
            roll_rel = euler[3]   # Roll change
            pitch_rel = euler[4]  # Pitch change  
            yaw_rel = euler[5]    # Yaw change

            # vr_data = self._quest_handler.get_new_data()
            
            # Convert degrees to radians and apply sensitivity
            # Mapping: controller roll -> robot RX, controller pitch -> robot RY, controller yaw -> robot RZ
            delta_rx = -np.deg2rad(roll_rel) * self._gyro_sensitivity   # Controller roll -> robot RX
            delta_ry = np.deg2rad(pitch_rel) * self._gyro_sensitivity   # Controller pitch -> robot RY
            delta_rz = np.deg2rad(yaw_rel) * self._gyro_sensitivity     # Controller yaw -> robot RZ
            
            target_rx = pose['rx'] + delta_rx
            target_ry = pose['ry'] + delta_ry
            target_rz = pose['rz'] + delta_rz
            
            # Periodic debug print (every ~1 second at 30Hz)
            if not hasattr(self, '_gyro_print_counter'):
                self._gyro_print_counter = 0
            self._gyro_print_counter += 1
            if self._gyro_print_counter % 30 == 0:
                print(f"[Gyro] rpy_rel=({roll_rel:+.1f}, {pitch_rel:+.1f}, {yaw_rel:+.1f}) deg  "
                      f"Δ=({np.rad2deg(delta_rx):+.2f}, {np.rad2deg(delta_ry):+.2f}, {np.rad2deg(delta_rz):+.2f}) deg")
        else:
            # Normal mode: D-pad for rotation
            target_rx = pose['rx'] + dpad_roll * self.angular_scale * 0.5
            target_ry = pose['ry'] + dpad_pitch * self.angular_scale * 0.5
            target_rz = pose['rz']
        
        # Apply movement if any input is active
        has_position_input = any(abs(v) > 0.01 for v in [left_stick_x, left_stick_y, right_y])
        has_rotation_input = circle_held or any(abs(v) > 0.01 for v in [dpad_pitch, dpad_roll])
        
        if has_position_input or has_rotation_input:
            self.piper.move_to_pose(target_x, target_y, target_z, target_rx, target_ry, target_rz)
        
        # Gripper control via triggers (incremental - no snap-back)
        # L2: close gripper (decrease angle), R2: open gripper (increase angle)
        l2 = self.ps5.get_trigger(PS5Axes.L2)  # Close
        r2 = self.ps5.get_trigger(PS5Axes.R2)  # Open
        
        if l2 > 0.1 or r2 > 0.1:
            # Incremental control: adjust target angle based on trigger pressure
            delta = (r2 - l2) * self._gripper_speed
            self._gripper_target_angle += delta
            # Clamp to valid range
            self._gripper_target_angle = max(GRIPPER_CLOSE_ANGLE, 
                                              min(GRIPPER_OPEN_ANGLE, self._gripper_target_angle))
            self.piper.set_gripper_angle(self._gripper_target_angle)
            self._gripper_open = self._gripper_target_angle > GRIPPER_OPEN_ANGLE / 2
        
        # === Data Recording ===
        if self._recording and self._current_episode is not None:
            # Get current pose as observation
            current_pos = np.array([pose['x'], pose['y'], pose['z']])
            current_quat = np.array(euler_to_quat(pose['rx_deg'], pose['ry_deg'], pose['rz_deg']))
            ee_pose = np.concatenate([current_pos, current_quat])  # [x, y, z, qw, qx, qy, qz]
            
            # Get gripper state (normalized 0-1)
            gripper_angle = self.piper.get_gripper_angle() or self._gripper_target_angle
            gripper_state = gripper_angle / GRIPPER_OPEN_ANGLE  # Normalize to [0, 1]
            
            # Compute action as RELATIVE pose change (delta)
            # Action format: [delta_x, delta_y, delta_z, delta_qw, delta_qx, delta_qy, delta_qz, gripper_target]
            if self._prev_pose is not None:
                delta_x = target_x - self._prev_pose['x']
                delta_y = target_y - self._prev_pose['y']
                delta_z = target_z - self._prev_pose['z']
                
                # Compute delta rotation as quaternion difference
                # q_delta = q_target * q_prev^-1  (simplified: just store delta euler converted to quat)
                delta_rx = target_rx - self._prev_pose['rx']
                delta_ry = target_ry - self._prev_pose['ry']
                delta_rz = target_rz - self._prev_pose['rz']
                delta_quat = np.array(euler_to_quat(
                    np.rad2deg(delta_rx), np.rad2deg(delta_ry), np.rad2deg(delta_rz)
                ))
            else:
                # First frame: no delta, set to zero
                delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
                delta_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            
            # Gripper target (normalized)
            gripper_target = self._gripper_target_angle / GRIPPER_OPEN_ANGLE
            
            action = np.array([delta_x, delta_y, delta_z,
                               delta_quat[0], delta_quat[1], delta_quat[2], delta_quat[3],
                               gripper_target], dtype=np.float32)
            
            # Get camera frames
            fixed_img = self._fixed_cam.get_frame_rgb() if self._fixed_cam else None
            wrist_img = self._wrist_cam.get_frame_rgb() if self._wrist_cam else None
            
            # Handle missing camera data
            if fixed_img is None:
                fixed_img = np.zeros((480, 640, 3), dtype=np.uint8)
            if wrist_img is None:
                wrist_img = np.zeros_like(fixed_img)
            
            # Get force/torque data
            joint_angles = np.zeros(6)
            joint_torques = np.zeros(6)
            ee_force = np.zeros(6)
            
            frame_idx = len(self._current_episode.ee_pose)
            
            if self._force_estimator is not None and self.piper.piper is not None:
                joint_state = self._force_estimator.get_joint_state_from_piper(self.piper.piper)
                if joint_state is not None:
                    joint_angles, joint_torques = joint_state
                    ee_force = self._force_estimator.estimate_force(
                        joint_angles, joint_torques,
                        use_filter=True,
                        use_gravity_comp=True
                    )
                    # DEBUG: Print force data on first few frames
                    if frame_idx < 3:
                        print(f"[DEBUG Force] Frame {frame_idx}: SUCCESS")
                        print(f"  joint_angles: {joint_angles}")
                        print(f"  joint_torques: {joint_torques}")
                        print(f"  ee_force: {ee_force}")
                else:
                    # DEBUG: joint_state is None - print for first few frames
                    if frame_idx < 5:
                        print(f"[DEBUG Force] Frame {frame_idx}: joint_state is None!")
            else:
                # DEBUG: force_estimator not available
                if frame_idx < 3:
                    print(f"[DEBUG Force] Frame {frame_idx}: force_estimator={self._force_estimator is not None}, piper.piper={self.piper.piper is not None}")
            
            # Record data
            self._current_episode.fixed_images.append(fixed_img)
            self._current_episode.wrist_images.append(wrist_img)
            self._current_episode.ee_pose.append(ee_pose)
            self._current_episode.gripper_state.append(gripper_state)
            self._current_episode.action.append(action)
            self._current_episode.timestamp.append(time.time())
            # Record force data
            self._current_episode.joint_angles.append(joint_angles)
            self._current_episode.joint_torques.append(joint_torques)
            self._current_episode.ee_force.append(ee_force)
            
            # Update previous pose for next delta computation
            self._prev_pose = {
                'x': target_x, 'y': target_y, 'z': target_z,
                'rx': target_rx, 'ry': target_ry, 'rz': target_rz,
            }
        
        return True
    
    def stop(self):
        """Stop teleoperation and cleanup resources."""
        self._running = False
        # Stop IMU handler subprocess
        if self._imu_handler is not None:
            self._imu_handler.stop()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Teleoperate Piper arm using PS5 DualSense controller.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic teleoperation (no data recording)
  python 2_teleop_ps5_controller.py
  
  # Teleoperation with data recording
  python 2_teleop_ps5_controller.py --record --out_dir data/teleop_data
  
  # With specific cameras
  python 2_teleop_ps5_controller.py --record --front_cam Orbbec_Gemini_335L --wrist_cam Dabai_DC1

Data Recording Controls (when --record is enabled):
  - Square (short press): Start/Stop recording (1 vibrate = start, 2 vibrates = stop & save)
  - Square (long press ~1.5s): Discard current episode (continuous vibrate while holding, 3 vibrates = discarded)
"""
    )
    
    parser.add_argument("--can_interface", "-c", type=str, default="can0",
                        help="CAN interface name (default: can0)")
    parser.add_argument("--speed", "-s", type=int, default=DEFAULT_SPEED_PERCENT,
                        help=f"Initial motion speed percentage (default: {DEFAULT_SPEED_PERCENT})")
    parser.add_argument("--linear_scale", type=float, default=LINEAR_VEL_SCALE,
                        help=f"Linear velocity scale m/cycle (default: {LINEAR_VEL_SCALE})")
    parser.add_argument("--angular_scale", type=float, default=ANGULAR_VEL_SCALE,
                        help=f"Angular velocity scale rad/cycle (default: {ANGULAR_VEL_SCALE})")
    parser.add_argument("--control_freq", type=int, default=30,
                        help="Control loop frequency Hz (default: 30)")
    parser.add_argument("--deadzone", type=float, default=0.1,
                        help="Joystick deadzone (default: 0.1)")
    parser.add_argument("--show_camera", action="store_true",
                        help="Show camera feed (requires OpenCV)")
    parser.add_argument("--camera_id", type=str, default=DEFAULT_FRONT_CAMERA,
                        help=f"Camera ID/name for display (default: {DEFAULT_FRONT_CAMERA})")
    
    # Data recording options
    parser.add_argument("--record", "-r", action="store_true",
                        help="Enable data recording mode")
    parser.add_argument("--out_dir", "-o", type=str, default="data/teleop_data",
                        help="Output directory for recorded data (default: data/teleop_data)")
    parser.add_argument("--front_cam", "-f", type=str, default=DEFAULT_FRONT_CAMERA,
                        help=f"Front camera ID/name (default: {DEFAULT_FRONT_CAMERA})")
    parser.add_argument("--wrist_cam", "-w", type=str, default=DEFAULT_WRIST_CAMERA,
                        help=f"Wrist camera ID/name, use -1 to disable (default: {DEFAULT_WRIST_CAMERA})")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Camera image width (default: 640)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Camera image height (default: 480)")
    
    return parser.parse_args()


def normalize_camera_value(value: str) -> Union[int, str]:
    """Normalize camera value from command line."""
    if value.strip() == "-1":
        return -1
    try:
        return int(value)
    except ValueError:
        return value


def main():
    # Check input group permissions before anything else
    check_and_setup_input_group()
    
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Piper PS5 Teleoperation")
    if args.record:
        print("📹 Data Recording Mode ENABLED")
    print("=" * 60)
    
    # Create output directory if recording
    out_dir = None
    if args.record:
        out_dir = Path(args.out_dir)
        
        # Backup existing directory if it exists (instead of overwriting)
        if out_dir.exists():
            backup_existing_directory(out_dir)
        
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nData will be saved to: {out_dir}")
        print(f"  Format: Compressed tar.gz archives (one per episode)")
    
    # Initialize PS5 controller
    ps5 = None
    # Initialize PS5 controller
    print("\n[Init] Initializing PS5 controller...")
    if not PYGAME_AVAILABLE:
        print("[Error] pygame not available. Please install: pip install pygame")
        sys.exit(1)
    
    ps5 = PS5Controller()
    ps5.deadzone = args.deadzone
    
    if not ps5.initialize():
        print("[Error] No PS5 controller found")
        print("[Error] Make sure PS5 controller is connected via Bluetooth")
        ps5.close()
        sys.exit(1)
    
    # =========================================================================
    # Parallel initialization: IMU calibration + Arm connection + Cameras
    # IMU calibration requires the controller to be still, but doesn't conflict
    # with arm initialization or camera setup, so we can do them in parallel.
    # =========================================================================
    import concurrent.futures
    
    imu_handler = None
    piper = None
    fixed_cam = None
    wrist_cam = None
    display_cam = None
    init_errors = []
    
    def init_imu():
        """Initialize IMU handler (includes gyroscope calibration)."""
        print("\n[Init] Initializing IMU handler (gyroscope calibration)...")
        print("[Init] ⚠ Keep the PS5 controller STILL during calibration!")
        handler = RawIMUHandler()
        print("[Init] ✓ IMU handler initialized and calibrated")
        return handler
    
    def init_arm():
        """Initialize and enable Piper arm."""
        print("\n[Init] Connecting to Piper arm...")
        arm = PiperController(args.can_interface)
        arm._speed_percent = args.speed
        
        if not arm.connect():
            raise RuntimeError("Failed to connect to Piper arm")
        
        if not arm.enable():
            arm.disconnect()
            raise RuntimeError("Failed to enable Piper arm")
        
        return arm
    
    def init_cameras():
        """Initialize cameras for recording or display."""
        cams = {"fixed": None, "wrist": None, "display": None}
        
        if args.record:
            # Front camera (required for recording)
            front_cam_id = normalize_camera_value(args.front_cam)
            print(f"\n[Init] Starting front camera: {front_cam_id}")
            cams["fixed"] = CameraCapture(front_cam_id, args.image_width, args.image_height, name="front")
            if not cams["fixed"].start():
                raise RuntimeError(f"Front camera failed to start: {front_cam_id}")
            
            # Wrist camera (optional)
            wrist_cam_id = normalize_camera_value(args.wrist_cam)
            wrist_cam_enabled = is_camera_enabled(wrist_cam_id)
            if wrist_cam_enabled:
                print(f"[Init] Starting wrist camera: {wrist_cam_id}")
                cams["wrist"] = CameraCapture(wrist_cam_id, args.image_width, args.image_height, name="wrist")
                if not cams["wrist"].start():
                    # Stop fixed cam before raising
                    if cams["fixed"]:
                        cams["fixed"].stop()
                    raise RuntimeError(f"Wrist camera failed to start: {wrist_cam_id}")
            
            # Use front camera for display if --show_camera
            if args.show_camera:
                cams["display"] = cams["fixed"]
                
        elif args.show_camera and CV2_AVAILABLE:
            # Just display camera, no recording
            print("\n[Init] Starting camera for display...")
            cams["display"] = CameraCapture(args.camera_id, name="display")
            if not cams["display"].start():
                print("[Init] ⚠ Camera failed to start, continuing without display")
                cams["display"] = None
        
        print("[Init] ✓ Cameras initialized")
        return cams
    
    # Run all initializations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        imu_future = executor.submit(init_imu)
        arm_future = executor.submit(init_arm)
        cam_future = executor.submit(init_cameras)
        
        # Wait for all to complete
        try:
            imu_handler = imu_future.result()
        except Exception as e:
            init_errors.append(f"IMU init failed: {e}")
        
        try:
            piper = arm_future.result()
        except Exception as e:
            init_errors.append(f"Arm init failed: {e}")
        
        try:
            cams = cam_future.result()
            fixed_cam = cams["fixed"]
            wrist_cam = cams["wrist"]
            display_cam = cams["display"]
        except Exception as e:
            init_errors.append(f"Camera init failed: {e}")
    
    # Check for errors
    if init_errors:
        print("\n[Error] Initialization failed:")
        for err in init_errors:
            print(f"  - {err}")
        if imu_handler:
            imu_handler.stop()
        if piper:
            piper.disconnect()
        if fixed_cam:
            fixed_cam.stop()
        if wrist_cam:
            wrist_cam.stop()
        ps5.close()
        sys.exit(1)
    
    # Go to home position
    print("\n[Init] Going to home position...")
    piper.go_to_home()
    time.sleep(2.0)
    
    # Open gripper
    piper.open_gripper()
    time.sleep(0.5)
    
    # Initialize force estimator
    force_estimator = None
    if FORCE_ESTIMATOR_AVAILABLE:
        print("\n[Init] Initializing force estimator...")
        force_estimator = PiperForceEstimator(dh_is_offset=0x01)
        # Calibrate gravity baseline at home position
        time.sleep(0.5)  # Wait for arm to settle
        if force_estimator.calibrate_gravity_from_piper(piper.piper):
            print("[Init] ✓ Force estimator gravity baseline calibrated")
        else:
            print("[Init] ✗ Force estimator calibration failed, continuing without gravity compensation")
    else:
        print("\n[Init] Force estimator not available, skipping force recording")
    
    # Create teleoperation controller
    teleop = TeleoperationController(
        piper, 
        ps5=ps5,
        linear_scale=args.linear_scale,
        angular_scale=args.angular_scale,
        control_freq=args.control_freq,
        record_data=args.record,
        fixed_cam=fixed_cam,
        wrist_cam=wrist_cam,
        force_estimator=force_estimator,
        imu_handler=imu_handler,
    )
    teleop.start()
    
    # Track saved episodes
    saved_episodes = 0
    prev_recording_state = False  # Track recording state changes
    
    # Initialize background saver for non-blocking data saving
    bg_saver = None
    if args.record:
        bg_saver = BackgroundSaver()
        bg_saver.start()
    
    # Main loop
    control_period = 1.0 / args.control_freq
    
    try:
        while True:
            loop_start = time.time()
            
            # Track recording state before step
            was_recording = teleop._recording if args.record else False
            
            # Run teleop step
            result = teleop.step()
            
            if result is False:
                # PS button pressed, quit
                break
            
            # Check if recording just stopped (state changed from True to False)
            if args.record and was_recording and not teleop._recording:
                # Recording was stopped via Square button, queue episode for background saving
                episode = teleop._current_episode
                if episode is not None and len(episode.ee_pose) > 10:  # Min 10 frames
                    episode.success = True
                    bg_saver.save_episode(episode, out_dir)  # Non-blocking
                    saved_episodes += 1
                    print(f"[Main] ✓ Episode {saved_episodes} queued for saving!")
                teleop._current_episode = None
            
            # Show camera if available
            if display_cam is not None and CV2_AVAILABLE:
                frame = display_cam.get_frame()
                if frame is not None:
                    # Add overlay text
                    pose = piper.get_ee_pose_meters()
                    if pose:
                        text = f"Pos: ({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f})m"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
                        text2 = f"Rot: ({pose['rx_deg']:.1f}, {pose['ry_deg']:.1f}, {pose['rz_deg']:.1f})deg"
                        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
                    
                    speed_text = f"Speed: {piper._speed_percent}%"
                    cv2.putText(frame, speed_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (255, 255, 0), 2)
                    
                    if piper._emergency_stop:
                        cv2.putText(frame, "EMERGENCY STOP", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.8, (0, 0, 255), 2)
                    
                    # Recording indicator
                    if args.record and teleop._recording:
                        cv2.putText(frame, "REC", (frame.shape[1] - 70, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2)
                        cv2.circle(frame, (frame.shape[1] - 80, 25), 8, (0, 0, 255), -1)
                    
                    cv2.imshow("Piper Teleop", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    
    finally:
        print("\n[Main] Cleaning up...")
        teleop.stop()
        
        # Save any remaining episode
        if args.record and teleop._recording and teleop._current_episode is not None:
            episode = teleop._current_episode
            if len(episode.ee_pose) > 10:
                episode.success = True
                bg_saver.save_episode(episode, out_dir)  # Queue for background saving
                saved_episodes += 1
        
        if not piper._emergency_stop:
            piper.open_gripper()
            time.sleep(0.5)
            piper.go_to_home()
            time.sleep(2.0)
        
        piper.disconnect()
        
        # Close controllers
        ps5.close()
        
        # Stop cameras
        if fixed_cam:
            fixed_cam.stop()
        if wrist_cam:
            wrist_cam.stop()
        if display_cam and display_cam not in [fixed_cam, wrist_cam]:
            display_cam.stop()
        
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        
        # Wait for background saves to complete before saving metadata
        if bg_saver is not None:
            bg_saver.stop(wait=True, timeout=60.0)
            # Update saved_episodes count from background saver
            saved_episodes = bg_saver.get_saved_count()
        
        # Save metadata if recording
        if args.record and out_dir:
            metadata = {
                "collection_type": "teleop",
                "control_mode": "ps5",
                "action_type": "relative_delta",
                "action_description": "action[0:3] = delta_xyz (meters), action[3:7] = delta_quat (wxyz), action[7] = gripper_target (0-1)",
                "ee_pose_description": "ee_pose[0:3] = xyz (meters), ee_pose[3:7] = quat (wxyz)",
                "gripper_state_description": "normalized 0-1 (0=closed, 1=open)",
                "num_episodes": saved_episodes,
                "control_freq": args.control_freq,
                "image_size": [args.image_height, args.image_width],
                "home_position": list(HOME_POSITION),
                "home_orientation_rad": list(HOME_ORIENTATION),
                "linear_scale": args.linear_scale,
                "angular_scale": args.angular_scale,
                "front_camera": str(args.front_cam),
                "wrist_camera": str(args.wrist_cam),
                "wrist_camera_enabled": wrist_cam is not None,
            }
            
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n[Main] Data collection complete!")
            print(f"  Episodes saved: {saved_episodes}")
            print(f"  Output directory: {out_dir}")
        
        print("[Main] Done!")


if __name__ == "__main__":
    main()
