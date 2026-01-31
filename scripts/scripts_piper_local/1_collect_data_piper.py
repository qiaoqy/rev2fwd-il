#!/usr/bin/env python3
"""Script 1: Data Collection for Piper Pick & Place Task.

This script collects Task B trajectories using a finite state machine (FSM) expert:
- Task B: Pick from **plate center** → Place at **random table position**

The collected trajectories will later be time-reversed to generate Task A training data.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage
python 1_collect_data_piper.py --num_episodes 50 --out_dir data/piper_pick_place

# Full options
python 1_collect_data_piper.py \
    --can_interface can0 \
    --num_episodes 100 \
    --out_dir data/piper_pick_place \
    --control_freq 30 \
    --image_width 640 \
    --image_height 480 \
    --fixed_cam_id 0 \
    --wrist_cam_id 1 \
    --seed 42

=============================================================================
KEYBOARD CONTROLS
=============================================================================
| Key   | Action                              |
|-------|-------------------------------------|
| SPACE | Start new episode                   |
| ESC   | Emergency stop (freeze arm)         |
| Q     | Quit and save all data              |
| R     | Reset to home position              |
| S     | Skip current episode (discard)      |

=============================================================================
FSM STATES
=============================================================================
IDLE → GO_TO_HOME → HOVER_PLATE → LOWER_GRASP → CLOSE_GRIP → LIFT_OBJECT 
     → HOVER_PLACE → LOWER_PLACE → OPEN_GRIP → LIFT_RETREAT → DONE → IDLE
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Piper SDK
from piper_sdk import C_PiperInterface_V2


# =============================================================================
# Constants and Parameters
# =============================================================================

# === Workspace Parameters ===
PLATE_CENTER = (0.0, 0.15, 0.0)       # Pick position (fixed) in meters
GRASP_HEIGHT = 0.15                    # Z height for grasping (meters)
HOVER_HEIGHT = 0.25                    # Safe height for movement (meters)
RANDOM_RANGE_X = (-0.1, 0.1)           # Random X offset from plate center
RANDOM_RANGE_Y = (-0.1, 0.1)           # Random Y offset from plate center

# === Control Parameters ===
CONTROL_FREQ = 30                      # Hz
POSITION_TOLERANCE = 0.005             # 5mm position tolerance
ORIENTATION_TOLERANCE = 2.0            # 2 degrees tolerance
GRIPPER_OPEN_POS = 70000               # Gripper open position (0.001mm = 70mm)
GRIPPER_CLOSE_POS = 0                  # Gripper closed position
GRIPPER_SPEED = 1000                   # Gripper speed
GRIPPER_EFFORT = 500                   # Gripper force (0.001 N/m)

# === Motion Parameters ===
MOTION_SPEED_PERCENT = 20              # Conservative speed (0-100%)
SETTLE_TIME = 0.5                      # Seconds to wait after reaching position
GRIPPER_WAIT_TIME = 0.5                # Seconds to wait for gripper action

# === Workspace Limits (Safety) ===
WORKSPACE_LIMITS = {
    "x_min": -0.3,  "x_max": 0.3,
    "y_min": 0.0,   "y_max": 0.4,
    "z_min": 0.05,  "z_max": 0.4,
}

# === Default Orientation (gripper pointing down) ===
# You may need to calibrate this for your setup
DEFAULT_ORIENTATION_EULER = (180.0, 0.0, 0.0)  # RX, RY, RZ in degrees


# =============================================================================
# FSM State Definitions
# =============================================================================

class FSMState(Enum):
    """Finite State Machine states for pick-and-place task."""
    IDLE = auto()
    GO_TO_HOME = auto()
    HOVER_PLATE = auto()
    LOWER_GRASP = auto()
    CLOSE_GRIP = auto()
    LIFT_OBJECT = auto()
    HOVER_PLACE = auto()
    LOWER_PLACE = auto()
    OPEN_GRIP = auto()
    LIFT_RETREAT = auto()
    DONE = auto()


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


def meters_to_sdk(value_m: float) -> int:
    """Convert meters to SDK units (0.001mm)."""
    return round(value_m * 1_000_000)  # 1m = 1,000,000 * 0.001mm


def sdk_to_meters(value_sdk: int) -> float:
    """Convert SDK units (0.001mm) to meters."""
    return value_sdk / 1_000_000.0


def degrees_to_sdk(value_deg: float) -> int:
    """Convert degrees to SDK units (0.001 degrees)."""
    return round(value_deg * 1000)


def sdk_to_degrees(value_sdk: int) -> float:
    """Convert SDK units (0.001 degrees) to degrees."""
    return value_sdk / 1000.0


# =============================================================================
# Camera Wrapper
# =============================================================================

class CameraCapture:
    """Wrapper for USB camera capture."""
    
    def __init__(self, camera_id: int, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None
        
    def start(self):
        """Start camera capture in background thread."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        # Wait for first frame
        for _ in range(50):
            if self._latest_frame is not None:
                break
            time.sleep(0.1)
        
        if self._latest_frame is None:
            raise RuntimeError(f"Camera {self.camera_id} not returning frames")
        
        print(f"[Camera {self.camera_id}] Started ({self.width}x{self.height})")
        
    def _capture_loop(self):
        """Background thread for continuous capture."""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest_frame = frame_rgb
            time.sleep(0.001)  # Small delay to prevent CPU overload
            
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame (RGB format)."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None
    
    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print(f"[Camera {self.camera_id}] Stopped")


# =============================================================================
# Piper Arm Controller
# =============================================================================

class PiperController:
    """High-level controller for Piper robotic arm."""
    
    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self.piper = None
        self._emergency_stop = False
        self._home_position = None
        
    def connect(self):
        """Connect to the Piper arm."""
        print(f"[Piper] Connecting via {self.can_interface}...")
        self.piper = C_PiperInterface_V2(self.can_interface)
        self.piper.ConnectPort()
        
        # Enable the arm
        print("[Piper] Enabling arm...")
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        
        # Set motion mode: CAN command control, MOVE L (linear), conservative speed
        self.piper.MotionCtrl_2(0x01, 0x02, MOTION_SPEED_PERCENT, 0x00)
        
        # Enable gripper
        self.piper.GripperCtrl(GRIPPER_OPEN_POS, GRIPPER_SPEED, 0x01, 0)
        
        print("[Piper] Connected and enabled!")
        
    def disconnect(self):
        """Disable and disconnect from the arm."""
        if self.piper is not None:
            print("[Piper] Disabling arm...")
            self.piper.DisablePiper()
            
    def emergency_stop(self):
        """Trigger emergency stop (freeze arm)."""
        self._emergency_stop = True
        if self.piper is not None:
            # Send current position to freeze
            pose = self.get_ee_pose_sdk()
            if pose is not None:
                self.piper.EndPoseCtrl(*pose)
        print("[Piper] EMERGENCY STOP ACTIVATED!")
        
    def clear_emergency_stop(self):
        """Clear emergency stop flag."""
        self._emergency_stop = False
        print("[Piper] Emergency stop cleared.")
        
    def get_ee_pose_sdk(self) -> Optional[Tuple[int, int, int, int, int, int]]:
        """Get current end-effector pose in SDK units."""
        if self.piper is None:
            return None
        try:
            msg = self.piper.GetArmEndPoseMsgs()
            pose = msg.end_pose
            return (pose.X_axis, pose.Y_axis, pose.Z_axis,
                    pose.RX_axis, pose.RY_axis, pose.RZ_axis)
        except Exception as e:
            print(f"[Piper] Error reading pose: {e}")
            return None
    
    def get_ee_pose_meters(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get current end-effector pose in meters and degrees."""
        pose_sdk = self.get_ee_pose_sdk()
        if pose_sdk is None:
            return None
        x, y, z = sdk_to_meters(pose_sdk[0]), sdk_to_meters(pose_sdk[1]), sdk_to_meters(pose_sdk[2])
        rx, ry, rz = sdk_to_degrees(pose_sdk[3]), sdk_to_degrees(pose_sdk[4]), sdk_to_degrees(pose_sdk[5])
        return (x, y, z, rx, ry, rz)
    
    def get_ee_pose_quat(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get current end-effector pose as position (xyz) and quaternion (wxyz)."""
        pose = self.get_ee_pose_meters()
        if pose is None:
            return None
        x, y, z, rx, ry, rz = pose
        quat = euler_to_quat(rx, ry, rz)
        return np.array([x, y, z]), np.array(quat)
    
    def get_gripper_state(self) -> Optional[float]:
        """Get gripper state normalized to [0, 1] (0=closed, 1=open)."""
        if self.piper is None:
            return None
        try:
            msg = self.piper.GetArmGripperMsgs()
            gripper_pos = msg.gripper_state.grippers_angle  # in 0.001mm
            # Normalize: GRIPPER_CLOSE_POS=0 → 0, GRIPPER_OPEN_POS=70000 → 1
            normalized = gripper_pos / GRIPPER_OPEN_POS
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            print(f"[Piper] Error reading gripper: {e}")
            return None
    
    def send_ee_pose(self, x: float, y: float, z: float, 
                     rx: float, ry: float, rz: float):
        """Send target end-effector pose (meters and degrees).
        
        Args:
            x, y, z: Position in meters
            rx, ry, rz: Orientation in degrees (Euler XYZ)
        """
        if self._emergency_stop:
            return
        
        # Apply workspace limits for safety
        x = max(WORKSPACE_LIMITS["x_min"], min(WORKSPACE_LIMITS["x_max"], x))
        y = max(WORKSPACE_LIMITS["y_min"], min(WORKSPACE_LIMITS["y_max"], y))
        z = max(WORKSPACE_LIMITS["z_min"], min(WORKSPACE_LIMITS["z_max"], z))
        
        # Convert to SDK units
        X = meters_to_sdk(x)
        Y = meters_to_sdk(y)
        Z = meters_to_sdk(z)
        RX = degrees_to_sdk(rx)
        RY = degrees_to_sdk(ry)
        RZ = degrees_to_sdk(rz)
        
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
    
    def send_ee_pose_quat(self, pos: np.ndarray, quat: np.ndarray):
        """Send target end-effector pose with quaternion orientation.
        
        Args:
            pos: Position [x, y, z] in meters
            quat: Quaternion [w, x, y, z]
        """
        rx, ry, rz = quat_to_euler(quat[0], quat[1], quat[2], quat[3])
        self.send_ee_pose(pos[0], pos[1], pos[2], rx, ry, rz)
    
    def set_gripper(self, gripper_value: float):
        """Set gripper position.
        
        Args:
            gripper_value: Normalized gripper position (0=close, 1=open)
        """
        if self._emergency_stop:
            return
        
        gripper_value = max(0.0, min(1.0, gripper_value))
        gripper_pos = int(gripper_value * GRIPPER_OPEN_POS)
        self.piper.GripperCtrl(gripper_pos, GRIPPER_SPEED, 0x01, 0)
    
    def open_gripper(self):
        """Fully open the gripper."""
        self.set_gripper(1.0)
    
    def close_gripper(self):
        """Fully close the gripper."""
        self.set_gripper(0.0)
    
    def go_to_home(self):
        """Move to home position (all joints zero)."""
        if self._emergency_stop:
            return
        print("[Piper] Going to home position...")
        # Set joint control mode
        self.piper.MotionCtrl_2(0x01, 0x01, MOTION_SPEED_PERCENT, 0x00)
        # All joints to zero
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(2.0)  # Wait for motion
        # Switch back to end pose control mode (MOVE L)
        self.piper.MotionCtrl_2(0x01, 0x02, MOTION_SPEED_PERCENT, 0x00)
        
        # Record home position
        self._home_position = self.get_ee_pose_meters()
        if self._home_position:
            print(f"[Piper] Home position: x={self._home_position[0]:.3f}, "
                  f"y={self._home_position[1]:.3f}, z={self._home_position[2]:.3f}")
    
    def is_position_reached(self, target_pos: Tuple[float, float, float], 
                            tolerance: float = POSITION_TOLERANCE) -> bool:
        """Check if current position is within tolerance of target."""
        pose = self.get_ee_pose_meters()
        if pose is None:
            return False
        current_pos = np.array(pose[:3])
        target = np.array(target_pos)
        distance = np.linalg.norm(current_pos - target)
        return distance < tolerance


# =============================================================================
# Data Recording
# =============================================================================

@dataclass
class EpisodeData:
    """Container for episode data."""
    episode_id: int
    fixed_images: list  # List of (H, W, 3) uint8 arrays
    wrist_images: list  # List of (H, W, 3) uint8 arrays
    ee_pose: list       # List of (7,) arrays [x, y, z, qw, qx, qy, qz]
    gripper_state: list # List of floats
    action: list        # List of (8,) arrays [target_ee_pose(7), gripper(1)]
    fsm_state: list     # List of FSMState enum values (as int)
    timestamp: list     # List of Unix timestamps
    place_pose: np.ndarray  # Target place position (7,)
    goal_pose: np.ndarray   # Plate center position (7,)
    success: bool = False


def save_episode(episode: EpisodeData, out_dir: Path):
    """Save episode data to NPZ file and PNG images."""
    episode_dir = out_dir / f"episode_{episode.episode_id:04d}"
    fixed_dir = episode_dir / "fixed_cam"
    wrist_dir = episode_dir / "wrist_cam"
    
    # Create directories
    fixed_dir.mkdir(parents=True, exist_ok=True)
    wrist_dir.mkdir(parents=True, exist_ok=True)
    
    # Save images
    for i, (fixed_img, wrist_img) in enumerate(zip(episode.fixed_images, episode.wrist_images)):
        # Convert RGB to BGR for cv2
        cv2.imwrite(str(fixed_dir / f"{i:06d}.png"), cv2.cvtColor(fixed_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(wrist_dir / f"{i:06d}.png"), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
    
    # Convert lists to arrays
    T = len(episode.ee_pose)
    ee_pose_arr = np.array(episode.ee_pose, dtype=np.float32)
    gripper_arr = np.array(episode.gripper_state, dtype=np.float32)
    action_arr = np.array(episode.action, dtype=np.float32)
    fsm_arr = np.array(episode.fsm_state, dtype=np.int32)
    timestamp_arr = np.array(episode.timestamp, dtype=np.float64)
    
    # Save NPZ
    np.savez(
        episode_dir / "episode_data.npz",
        ee_pose=ee_pose_arr,
        gripper_state=gripper_arr,
        action=action_arr,
        fsm_state=fsm_arr,
        timestamp=timestamp_arr,
        place_pose=episode.place_pose,
        goal_pose=episode.goal_pose,
        success=episode.success,
        episode_id=episode.episode_id,
        num_timesteps=T,
    )
    
    print(f"[Save] Episode {episode.episode_id} saved to {episode_dir} ({T} timesteps)")


# =============================================================================
# Finite State Machine
# =============================================================================

class PickPlaceFSM:
    """FSM expert for Task B: Pick from plate → Place at random."""
    
    def __init__(self, piper: PiperController, 
                 plate_center: Tuple[float, float, float] = PLATE_CENTER,
                 grasp_height: float = GRASP_HEIGHT,
                 hover_height: float = HOVER_HEIGHT,
                 orientation: Tuple[float, float, float] = DEFAULT_ORIENTATION_EULER):
        self.piper = piper
        self.plate_center = plate_center
        self.grasp_height = grasp_height
        self.hover_height = hover_height
        self.orientation = orientation  # (rx, ry, rz) in degrees
        
        self.state = FSMState.IDLE
        self.place_position = None  # Random target (x, y, z)
        self._settle_start_time = None
        self._gripper_start_time = None
        
    def sample_random_place_position(self) -> Tuple[float, float, float]:
        """Sample a random placement position."""
        x = self.plate_center[0] + random.uniform(*RANDOM_RANGE_X)
        y = self.plate_center[1] + random.uniform(*RANDOM_RANGE_Y)
        z = self.plate_center[2]  # Same table height
        return (x, y, z)
    
    def get_target_pose(self) -> Tuple[float, float, float, float, float, float]:
        """Get target pose for current state (x, y, z, rx, ry, rz)."""
        rx, ry, rz = self.orientation
        
        if self.state == FSMState.GO_TO_HOME:
            # Use home position if available, otherwise plate hover
            if self.piper._home_position is not None:
                return self.piper._home_position
            return (self.plate_center[0], self.plate_center[1], self.hover_height, rx, ry, rz)
        
        elif self.state == FSMState.HOVER_PLATE:
            return (self.plate_center[0], self.plate_center[1], self.hover_height, rx, ry, rz)
        
        elif self.state == FSMState.LOWER_GRASP:
            return (self.plate_center[0], self.plate_center[1], self.grasp_height, rx, ry, rz)
        
        elif self.state == FSMState.CLOSE_GRIP:
            return (self.plate_center[0], self.plate_center[1], self.grasp_height, rx, ry, rz)
        
        elif self.state == FSMState.LIFT_OBJECT:
            return (self.plate_center[0], self.plate_center[1], self.hover_height, rx, ry, rz)
        
        elif self.state == FSMState.HOVER_PLACE:
            return (self.place_position[0], self.place_position[1], self.hover_height, rx, ry, rz)
        
        elif self.state == FSMState.LOWER_PLACE:
            return (self.place_position[0], self.place_position[1], self.grasp_height, rx, ry, rz)
        
        elif self.state == FSMState.OPEN_GRIP:
            return (self.place_position[0], self.place_position[1], self.grasp_height, rx, ry, rz)
        
        elif self.state == FSMState.LIFT_RETREAT:
            return (self.place_position[0], self.place_position[1], self.hover_height, rx, ry, rz)
        
        else:
            # IDLE or DONE - hold current position
            pose = self.piper.get_ee_pose_meters()
            if pose:
                return pose
            return (self.plate_center[0], self.plate_center[1], self.hover_height, rx, ry, rz)
    
    def get_target_gripper(self) -> float:
        """Get target gripper state for current state (0=close, 1=open)."""
        if self.state in [FSMState.CLOSE_GRIP, FSMState.LIFT_OBJECT, 
                          FSMState.HOVER_PLACE, FSMState.LOWER_PLACE]:
            return 0.0  # Closed
        else:
            return 1.0  # Open
    
    def start_episode(self):
        """Start a new episode."""
        self.place_position = self.sample_random_place_position()
        self.state = FSMState.GO_TO_HOME
        self._settle_start_time = None
        self._gripper_start_time = None
        print(f"[FSM] Starting episode. Place position: {self.place_position}")
    
    def step(self) -> bool:
        """Execute one step of the FSM.
        
        Returns:
            True if episode is done (success or abort)
        """
        if self.state == FSMState.IDLE:
            return False
        
        if self.state == FSMState.DONE:
            return True
        
        # Get and send target pose
        target = self.get_target_pose()
        self.piper.send_ee_pose(*target)
        
        # Handle gripper
        gripper_target = self.get_target_gripper()
        self.piper.set_gripper(gripper_target)
        
        # Check transition conditions
        self._check_transition(target[:3])
        
        return self.state == FSMState.DONE
    
    def _check_transition(self, target_pos: Tuple[float, float, float]):
        """Check if we should transition to next state."""
        position_reached = self.piper.is_position_reached(target_pos, POSITION_TOLERANCE)
        
        # Handle settling time for movement states
        movement_states = [FSMState.GO_TO_HOME, FSMState.HOVER_PLATE, FSMState.LOWER_GRASP,
                          FSMState.LIFT_OBJECT, FSMState.HOVER_PLACE, FSMState.LOWER_PLACE,
                          FSMState.LIFT_RETREAT]
        
        if self.state in movement_states:
            if position_reached:
                if self._settle_start_time is None:
                    self._settle_start_time = time.time()
                elif time.time() - self._settle_start_time >= SETTLE_TIME:
                    self._transition_to_next_state()
                    self._settle_start_time = None
            else:
                self._settle_start_time = None
        
        # Handle gripper states with timing
        elif self.state == FSMState.CLOSE_GRIP:
            if self._gripper_start_time is None:
                self._gripper_start_time = time.time()
            elif time.time() - self._gripper_start_time >= GRIPPER_WAIT_TIME:
                self._transition_to_next_state()
                self._gripper_start_time = None
        
        elif self.state == FSMState.OPEN_GRIP:
            if self._gripper_start_time is None:
                self._gripper_start_time = time.time()
            elif time.time() - self._gripper_start_time >= GRIPPER_WAIT_TIME:
                self._transition_to_next_state()
                self._gripper_start_time = None
    
    def _transition_to_next_state(self):
        """Transition to the next FSM state."""
        transitions = {
            FSMState.GO_TO_HOME: FSMState.HOVER_PLATE,
            FSMState.HOVER_PLATE: FSMState.LOWER_GRASP,
            FSMState.LOWER_GRASP: FSMState.CLOSE_GRIP,
            FSMState.CLOSE_GRIP: FSMState.LIFT_OBJECT,
            FSMState.LIFT_OBJECT: FSMState.HOVER_PLACE,
            FSMState.HOVER_PLACE: FSMState.LOWER_PLACE,
            FSMState.LOWER_PLACE: FSMState.OPEN_GRIP,
            FSMState.OPEN_GRIP: FSMState.LIFT_RETREAT,
            FSMState.LIFT_RETREAT: FSMState.DONE,
        }
        
        old_state = self.state
        self.state = transitions.get(self.state, FSMState.DONE)
        print(f"[FSM] {old_state.name} → {self.state.name}")


# =============================================================================
# Keyboard Input Handler
# =============================================================================

class KeyboardHandler:
    """Non-blocking keyboard input handler."""
    
    def __init__(self):
        self.start_requested = False
        self.quit_requested = False
        self.reset_requested = False
        self.skip_requested = False
        self.emergency_stop_requested = False
        self._lock = threading.Lock()
        self._keyboard_available = False
        
        # Try to import keyboard module
        # Note: On Linux, keyboard module requires root privileges
        # On Windows, it may require admin privileges
        try:
            import keyboard
            self._keyboard_available = True
            
            # Register hotkeys
            keyboard.on_press_key('space', lambda _: self._on_space())
            keyboard.on_press_key('escape', lambda _: self._on_escape())
            keyboard.on_press_key('q', lambda _: self._on_quit())
            keyboard.on_press_key('r', lambda _: self._on_reset())
            keyboard.on_press_key('s', lambda _: self._on_skip())
            
            print("[Keyboard] Handler initialized. Controls: SPACE=start, ESC=e-stop, Q=quit, R=reset, S=skip")
        except ImportError:
            print("[Keyboard] WARNING: 'keyboard' module not available. Install with: pip install keyboard")
            print("[Keyboard] Falling back to terminal input (press Enter after each command).")
        except Exception as e:
            print(f"[Keyboard] WARNING: keyboard module failed ({e}). May need root/admin privileges.")
            print("[Keyboard] Falling back to terminal input (press Enter after each command).")
    
    def _on_space(self):
        with self._lock:
            self.start_requested = True
    
    def _on_escape(self):
        with self._lock:
            self.emergency_stop_requested = True
    
    def _on_quit(self):
        with self._lock:
            self.quit_requested = True
    
    def _on_reset(self):
        with self._lock:
            self.reset_requested = True
    
    def _on_skip(self):
        with self._lock:
            self.skip_requested = True
    
    def poll(self) -> dict:
        """Poll for keyboard events.
        
        Returns:
            Dict with event flags (and resets them).
        """
        with self._lock:
            events = {
                'start': self.start_requested,
                'quit': self.quit_requested,
                'reset': self.reset_requested,
                'skip': self.skip_requested,
                'emergency_stop': self.emergency_stop_requested,
            }
            # Reset flags
            self.start_requested = False
            self.quit_requested = False
            self.reset_requested = False
            self.skip_requested = False
            self.emergency_stop_requested = False
        return events


# =============================================================================
# Main Data Collection Loop
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Collect pick-and-place data with Piper arm.")
    
    parser.add_argument("--can_interface", type=str, default="can0",
                        help="CAN interface name (default: can0)")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to collect (default: 50)")
    parser.add_argument("--out_dir", type=str, default="data/piper_pick_place",
                        help="Output directory for collected data")
    parser.add_argument("--control_freq", type=int, default=30,
                        help="Control frequency in Hz (default: 30)")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Camera image width (default: 640)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Camera image height (default: 480)")
    parser.add_argument("--fixed_cam_id", type=int, default=0,
                        help="Fixed camera device ID (default: 0)")
    parser.add_argument("--wrist_cam_id", type=int, default=1,
                        help="Wrist camera device ID (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--plate_x", type=float, default=0.0,
                        help="Plate center X position in meters (default: 0.0)")
    parser.add_argument("--plate_y", type=float, default=0.15,
                        help="Plate center Y position in meters (default: 0.15)")
    parser.add_argument("--plate_z", type=float, default=0.0,
                        help="Plate center Z position in meters (default: 0.0)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Update plate center from args
    plate_center = (args.plate_x, args.plate_y, args.plate_z)
    
    # Initialize components
    print("=" * 60)
    print("Piper Pick & Place Data Collection")
    print("=" * 60)
    
    # Initialize keyboard handler
    keyboard_handler = KeyboardHandler()
    
    # Initialize cameras
    print("\n[Init] Starting cameras...")
    fixed_cam = CameraCapture(args.fixed_cam_id, args.image_width, args.image_height)
    wrist_cam = CameraCapture(args.wrist_cam_id, args.image_width, args.image_height)
    
    try:
        fixed_cam.start()
        wrist_cam.start()
    except RuntimeError as e:
        print(f"[Error] Camera initialization failed: {e}")
        print("Please check camera connections and device IDs.")
        return
    
    # Initialize Piper arm
    print("\n[Init] Connecting to Piper arm...")
    piper = PiperController(args.can_interface)
    
    try:
        piper.connect()
    except Exception as e:
        print(f"[Error] Piper connection failed: {e}")
        fixed_cam.stop()
        wrist_cam.stop()
        return
    
    # Go to home position
    piper.go_to_home()
    time.sleep(1.0)
    
    # Initialize FSM
    fsm = PickPlaceFSM(piper, plate_center=plate_center)
    
    # Control loop parameters
    control_period = 1.0 / args.control_freq
    
    # Episode tracking
    episode_count = 0
    current_episode: Optional[EpisodeData] = None
    recording = False
    
    # Goal pose (fixed plate center in quaternion format)
    goal_quat = euler_to_quat(*DEFAULT_ORIENTATION_EULER)
    goal_pose = np.array([plate_center[0], plate_center[1], plate_center[2],
                          goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3]], dtype=np.float32)
    
    print("\n" + "=" * 60)
    print("Ready to collect data!")
    print("Press SPACE to start an episode.")
    print("Press Q to quit.")
    print("=" * 60 + "\n")
    
    try:
        while episode_count < args.num_episodes:
            loop_start = time.time()
            
            # Poll keyboard
            events = keyboard_handler.poll()
            
            # Handle quit
            if events['quit']:
                print("\n[Main] Quit requested. Exiting...")
                break
            
            # Handle emergency stop
            if events['emergency_stop']:
                piper.emergency_stop()
                fsm.state = FSMState.IDLE
                recording = False
                current_episode = None
                continue
            
            # Handle reset
            if events['reset']:
                print("\n[Main] Reset requested. Going to home position...")
                piper.clear_emergency_stop()
                piper.go_to_home()
                fsm.state = FSMState.IDLE
                recording = False
                current_episode = None
                continue
            
            # Handle skip
            if events['skip'] and recording:
                print("\n[Main] Skip requested. Discarding current episode...")
                fsm.state = FSMState.IDLE
                recording = False
                current_episode = None
                continue
            
            # Handle start
            if events['start'] and fsm.state == FSMState.IDLE:
                print(f"\n[Main] Starting episode {episode_count}...")
                piper.clear_emergency_stop()
                fsm.start_episode()
                
                # Create episode data container
                place_quat = euler_to_quat(*DEFAULT_ORIENTATION_EULER)
                place_pose = np.array([fsm.place_position[0], fsm.place_position[1], fsm.place_position[2],
                                       place_quat[0], place_quat[1], place_quat[2], place_quat[3]], dtype=np.float32)
                
                current_episode = EpisodeData(
                    episode_id=episode_count,
                    fixed_images=[],
                    wrist_images=[],
                    ee_pose=[],
                    gripper_state=[],
                    action=[],
                    fsm_state=[],
                    timestamp=[],
                    place_pose=place_pose,
                    goal_pose=goal_pose,
                )
                recording = True
            
            # Run FSM step if not idle
            if fsm.state != FSMState.IDLE:
                done = fsm.step()
                
                # Record data if recording
                if recording and current_episode is not None:
                    # Get observations
                    fixed_img = fixed_cam.get_frame()
                    wrist_img = wrist_cam.get_frame()
                    pos, quat = piper.get_ee_pose_quat() or (np.zeros(3), np.array([1, 0, 0, 0]))
                    gripper = piper.get_gripper_state() or 0.0
                    
                    # Construct ee_pose [x, y, z, qw, qx, qy, qz]
                    ee_pose = np.concatenate([pos, quat])
                    
                    # Get action (target pose from FSM + target gripper)
                    target = fsm.get_target_pose()
                    target_quat = euler_to_quat(target[3], target[4], target[5])
                    target_gripper = fsm.get_target_gripper()
                    action = np.array([target[0], target[1], target[2],
                                       target_quat[0], target_quat[1], target_quat[2], target_quat[3],
                                       target_gripper], dtype=np.float32)
                    
                    # Record
                    if fixed_img is not None and wrist_img is not None:
                        current_episode.fixed_images.append(fixed_img)
                        current_episode.wrist_images.append(wrist_img)
                        current_episode.ee_pose.append(ee_pose)
                        current_episode.gripper_state.append(gripper)
                        current_episode.action.append(action)
                        current_episode.fsm_state.append(fsm.state.value)
                        current_episode.timestamp.append(time.time())
                
                # Handle episode completion
                if done:
                    if current_episode is not None:
                        current_episode.success = True
                        save_episode(current_episode, out_dir)
                        episode_count += 1
                        print(f"[Main] Episode {episode_count - 1} completed successfully! "
                              f"({episode_count}/{args.num_episodes})")
                    
                    fsm.state = FSMState.IDLE
                    recording = False
                    current_episode = None
                    
                    # Return to hover position
                    piper.open_gripper()
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            sleep_time = control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    
    finally:
        # Cleanup
        print("\n[Main] Cleaning up...")
        piper.open_gripper()
        time.sleep(0.5)
        piper.go_to_home()
        time.sleep(2.0)
        piper.disconnect()
        fixed_cam.stop()
        wrist_cam.stop()
        
        # Save collection metadata
        metadata = {
            "num_episodes": episode_count,
            "control_freq": args.control_freq,
            "image_size": [args.image_height, args.image_width],
            "plate_center": list(plate_center),
            "grasp_height": GRASP_HEIGHT,
            "hover_height": HOVER_HEIGHT,
            "random_range_x": list(RANDOM_RANGE_X),
            "random_range_y": list(RANDOM_RANGE_Y),
            "seed": args.seed,
        }
        
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[Main] Data collection complete!")
        print(f"  Episodes collected: {episode_count}")
        print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
