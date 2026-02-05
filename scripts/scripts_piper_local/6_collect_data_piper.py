#!/usr/bin/env python3
"""Script 6: Data Collection for Piper Pick & Place Task.

This script collects Task B trajectories using a finite state machine (FSM) expert:
- Task B: Pick from **desk center** → Place at **random table position**

The collected trajectories will later be time-reversed to generate Task A training data.

Configuration calibrated from 0_system_test.py:
- Home position: (0.054, 0.0, 0.175)m with orientation (180°, 68.8°, 180°)
- Desk center (pick): (0.25, 0.0, 0.16)m with orientation (180°, 17.2°, 180°)
- Random place offset: ±0.15m in X and Y directions

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage with default cameras (Orbbec_Gemini_335L + Dabai_DC1)
python 1_collect_data_piper.py --num_episodes 50 --out_dir data/piper_pick_place

# Specify cameras by device name
python 1_collect_data_piper.py -f Orbbec_Gemini_335L -w Dabai_DC1

# Disable wrist camera
python 1_collect_data_piper.py --wrist_cam -1

# Custom workspace parameters
python 1_collect_data_piper.py \\
    --can_interface can0 \\
    --num_episodes 100 \\
    --out_dir data/piper_pick_place \\
    --plate_x 0.25 --plate_y 0.0 \\
    --grasp_height 0.16 --hover_height 0.25 \\
    --speed 20 \\
    --seed 42

python scripts/scripts_piper_local/1_collect_data_piper.py --num_episodes 1 --out_dir data/piper_pick_place


=============================================================================
KEYBOARD CONTROLS (Terminal Input Mode)
=============================================================================
| Input         | Action                              |
|---------------|-------------------------------------|
| ENTER / start | Start new episode                   |
| e / esc       | Emergency stop (freeze arm)         |
| q / quit      | Quit and save all data              |
| r / reset     | Reset to home position              |
| s / skip      | Skip current episode (discard)      |

=============================================================================
FSM STATES
=============================================================================
IDLE → GO_TO_HOME → HOVER_PLATE → LOWER_GRASP → CLOSE_GRIP → LIFT_OBJECT 
     → HOVER_PLACE → LOWER_PLACE → OPEN_GRIP → LIFT_RETREAT → RETURN_HOME → DONE → IDLE
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Piper SDK
from piper_sdk import C_PiperInterface_V2

# Force estimator
try:
    from rev2fwd_il.real import PiperForceEstimator
    FORCE_ESTIMATOR_AVAILABLE = True
except ImportError:
    FORCE_ESTIMATOR_AVAILABLE = False
    print("[Warning] Force estimator not available. Install rev2fwd_il package.")


# =============================================================================
# Constants and Parameters (Calibrated from 0_system_test.py)
# =============================================================================

# === Home Position (calibrated from script 0) ===
HOME_POSITION = (0.054, 0.0, 0.175)     # X, Y, Z in meters
HOME_ORIENTATION = (3.14, 1.2, 3.14)    # RX, RY, RZ in radians (~180°, 68.8°, 180°)

# === Workspace Parameters (calibrated from script 0) ===
# Pick position: desk center (same as script 0)
PLATE_CENTER = (0.25, 0.0, 0.16)        # Pick position (X, Y, Z) in meters
GRASP_HEIGHT = 0.16                      # Z height for grasping (meters)
HOVER_HEIGHT = 0.25                      # Safe height for movement (meters)

# === Place offset (random range from pick position) ===
# From script 0: place_offset = (-0.15, -0.15, 0.0) as max random range
RANDOM_RANGE_X = (-0.15, 0.15)           # Random X offset from plate center
RANDOM_RANGE_Y = (-0.15, 0.15)           # Random Y offset from plate center

# === Grasp Orientation (calibrated from script 0) ===
# desk_center_orientation: (3.14, 0.3, 3.14) radians (~180°, 17.2°, 180°)
DEFAULT_ORIENTATION_RAD = (3.14, 0.3, 3.14)  # RX, RY, RZ in radians
DEFAULT_ORIENTATION_EULER = tuple(np.rad2deg(r) for r in DEFAULT_ORIENTATION_RAD)  # in degrees

# === Control Parameters ===
CONTROL_FREQ = 30                        # Hz
POSITION_TOLERANCE = 0.015               # 15mm position tolerance (from script 0)
ORIENTATION_TOLERANCE = 5.0              # 5 degrees tolerance (from script 0)
GRIPPER_OPEN_ANGLE = 70.0                # Gripper open angle in degrees
GRIPPER_CLOSE_ANGLE = 0.0                # Gripper closed angle in degrees
GRIPPER_EFFORT = 500                     # Gripper force (0-1000)
GRIPPER_TOLERANCE = 50.0                 # Gripper angle tolerance in degrees

# === Motion Parameters ===
MOTION_SPEED_PERCENT = 20                # Conservative speed (0-100%)
MOVE_MODE = 0x00                         # Motion mode: 0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L
SETTLE_TIME = 0.5                        # Seconds to wait after reaching position
GRIPPER_WAIT_TIME = 0.5                  # Seconds to wait for gripper action
MOTION_TIMEOUT = 10.0                    # Motion timeout in seconds
GRIPPER_TIMEOUT = 3.0                    # Gripper operation timeout
ENABLE_TIMEOUT = 5.0                     # Arm enable timeout

# === Workspace Limits (Safety, from script 0) ===
WORKSPACE_LIMITS = {
    "x_min": -0.3,  "x_max": 0.5,
    "y_min": -0.3,  "y_max": 0.3,
    "z_min": 0.05,  "z_max": 0.50,
}

# === Camera Settings (from script 0) ===
DEFAULT_FRONT_CAMERA = "Orbbec_Gemini_335L"  # Front camera name/ID
DEFAULT_WRIST_CAMERA = "Dabai_DC1"           # Wrist camera name/ID (-1 to disable)


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
    RETURN_HOME = auto()
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
# Camera Wrapper (Enhanced from script 0)
# =============================================================================

class CameraCapture:
    """Wrapper for USB camera capture with device name resolution."""
    
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
        self._resolved_source = None
    
    def _resolve_camera_source(self, camera_id: Union[int, str]):
        """Resolve camera identifier to VideoCapture source.
        
        Supports:
        - Integer index (e.g., 8)
        - /dev/videoX path
        - Device name substring from /dev/v4l/by-id/ (e.g., Orbbec_Gemini_335L, Dabai_DC1)
        
        For cameras with multiple video nodes (e.g., Orbbec depth cameras), 
        prioritizes video-index0 which is typically the RGB stream.
        """
        if isinstance(camera_id, int):
            return camera_id
        
        if isinstance(camera_id, str):
            cam_str = camera_id.strip()
            
            # Check if it's a numeric string
            if cam_str.isdigit():
                return int(cam_str)
            
            # Check if it's a direct path
            if cam_str.startswith("/dev/video") or cam_str.startswith("/dev/v4l/"):
                return cam_str
            
            # Try to find by name in /dev/v4l/by-id/
            try:
                from pathlib import Path
                by_id_dir = Path("/dev/v4l/by-id")
                if by_id_dir.exists():
                    matches = []
                    for p in by_id_dir.iterdir():
                        if cam_str in p.name:
                            matches.append(p)
                    
                    # Prioritize video-index0 (RGB stream), then video-index1, then others
                    # This is critical for Orbbec/Dabai cameras with multiple video nodes
                    matches.sort(key=lambda p: (
                        0 if "video-index0" in p.name else 1,
                        0 if "video-index1" in p.name else 1,
                        p.name,
                    ))
                    
                    if matches:
                        real_path = str(matches[0].resolve())
                        print(f"[Camera] Resolved '{camera_id}' -> {real_path}")
                        return real_path
            except Exception:
                pass
        
        return camera_id
    
    def _open_camera(self, source):
        """Open camera with V4L2 backend first, fallback to others."""
        # Try V4L2 backend first (preferred for Linux)
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        
        # Try Orbbec obsensor backend if available
        obsensor_backend = getattr(cv2, "CAP_OBSENSOR", None)
        if obsensor_backend is not None:
            cap = cv2.VideoCapture(source, obsensor_backend)
            if cap.isOpened():
                return cap
        
        # Fallback to CAP_ANY
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        
        return None
        
    def start(self):
        """Start camera capture in background thread."""
        self._resolved_source = self._resolve_camera_source(self.camera_id)
        self.cap = self._open_camera(self._resolved_source)
        
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id} (resolved: {self._resolved_source})")
        
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
            raise RuntimeError(f"Camera {self.name} not returning frames")
        
        print(f"[Camera {self.name}] Started ({self.width}x{self.height})")
        
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
        print(f"[Camera {self.name}] Stopped")


def is_camera_enabled(camera_id: Union[int, str]) -> bool:
    """Check if camera is enabled (not -1 or "-1")."""
    if isinstance(camera_id, int):
        return camera_id >= 0
    if isinstance(camera_id, str):
        return camera_id.strip() != "-1"
    return False


# =============================================================================
# Piper Arm Controller (Enhanced from script 0)
# =============================================================================

class PiperController:
    """High-level controller for Piper robotic arm (verified from script 0)."""
    
    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self.piper: Optional[Any] = None
        self._emergency_stop = False
        self._home_position = HOME_POSITION
        self._home_orientation = HOME_ORIENTATION
        self.connected = False
        self.enabled = False
        
    def connect(self) -> bool:
        """Connect to the Piper arm."""
        try:
            print(f"[Piper] Connecting via {self.can_interface}...")
            self.piper = C_PiperInterface_V2(self.can_interface)
            self.piper.ConnectPort()
            
            # Wait for connection to stabilize
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
        """Enable the arm (verified method from script 0)."""
        if not self.connected or self.piper is None:
            print("[Piper] ✗ Not connected, cannot enable")
            return False
        
        try:
            print("[Piper] Enabling arm...")
            
            # Send enable command (EnableArm with parameter 7)
            self.piper.EnableArm(7)
            
            # Ensure control mode (exit teaching mode)
            self._ensure_control_mode()
            
            # Set motion mode (verified from script 0)
            self.piper.MotionCtrl_2(0x01, MOVE_MODE, MOTION_SPEED_PERCENT, 0)
            
            # Enable gripper
            self.piper.GripperCtrl(0x01, 1000, 0x01, 0)
            
            # Wait for enable completion
            start_time = time.time()
            while time.time() - start_time < ENABLE_TIMEOUT:
                if self._check_enabled():
                    self.enabled = True
                    print("[Piper] ✓ Enabled")
                    return True
                time.sleep(0.1)
            
            print("[Piper] ✗ Enable timeout")
            self._print_arm_status()
            return False
        except Exception as e:
            print(f"[Piper] ✗ Enable failed: {e}")
            return False
    
    def _ensure_control_mode(self):
        """Ensure exit teaching mode and enter control mode."""
        try:
            status = self._read_arm_status()
            if not status:
                return
            ctrl_mode_val = status.get('ctrl_mode_val')
            # If in teaching mode (ctrl_mode == 0), we need to exit it
        except Exception as e:
            print(f"[Piper][Debug] Control mode recovery failed: {e}")
    
    def _check_enabled(self) -> bool:
        """Check if arm is enabled."""
        try:
            status = self._read_arm_status()
            if status is None:
                return False
            
            ctrl_mode_val = status.get('ctrl_mode_val')
            # ctrl_mode != 0 means in control mode (CAN/Ethernet etc.)
            ctrl_ok = ctrl_mode_val is not None and ctrl_mode_val != 0
            
            enable_list = self._get_enable_status()
            enable_ok = bool(enable_list) and all(enable_list)
            
            return ctrl_ok and enable_ok
        except:
            return False
    
    def _get_enable_status(self) -> Optional[list]:
        """Get joint enable status."""
        try:
            return self.piper.GetArmEnableStatus()
        except:
            return None
    
    def _read_arm_status(self) -> Optional[Dict[str, Any]]:
        """Read arm status for debugging."""
        try:
            arm_status = self.piper.GetArmStatus()
            status = arm_status.arm_status
            ctrl_mode = getattr(status, 'ctrl_mode', None)
            arm_state = getattr(status, 'arm_status', None)
            mode_feed = getattr(status, 'mode_feed', None)
            teach_status = getattr(status, 'teach_status', None)
            motion_status = getattr(status, 'motion_status', None)
            
            return {
                'ctrl_mode': ctrl_mode,
                'arm_state': arm_state,
                'mode_feed': mode_feed,
                'teach_status': teach_status,
                'motion_status': motion_status,
                'ctrl_mode_val': int(ctrl_mode) if ctrl_mode is not None else None,
                'arm_state_val': int(arm_state) if arm_state is not None else None,
                'mode_feed_val': int(mode_feed) if mode_feed is not None else None,
                'teach_status_val': int(teach_status) if teach_status is not None else None,
                'motion_status_val': int(motion_status) if motion_status is not None else None,
            }
        except:
            return None
    
    def _print_arm_status(self, prefix: str = "[Piper][Debug]"):
        """Print arm status for debugging."""
        status = self._read_arm_status()
        if not status:
            print(f"{prefix} Cannot read arm status")
            return
        print(
            f"{prefix} ctrl_mode={status.get('ctrl_mode')}({status.get('ctrl_mode_val')}) "
            f"arm_status={status.get('arm_state')}({status.get('arm_state_val')}) "
            f"mode_feed={status.get('mode_feed')}({status.get('mode_feed_val')}) "
            f"teach_status={status.get('teach_status')}({status.get('teach_status_val')}) "
            f"motion_status={status.get('motion_status')}({status.get('motion_status_val')})"
        )
        
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
        """Trigger emergency stop (freeze arm)."""
        self._emergency_stop = True
        if self.piper is not None:
            try:
                # Stop motion
                self.piper.MotionCtrl_2(0x02, MOVE_MODE, 0, 0)  # 0x02 = stop
            except Exception as e:
                print(f"[Piper] Emergency stop error: {e}")
        print("[Piper] EMERGENCY STOP ACTIVATED!")
        
    def clear_emergency_stop(self):
        """Clear emergency stop flag and re-enable."""
        self._emergency_stop = False
        if self.piper is not None:
            try:
                self.piper.MotionCtrl_2(0x01, MOVE_MODE, MOTION_SPEED_PERCENT, 0)
            except:
                pass
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
    
    def get_ee_pose_meters(self) -> Optional[Dict[str, float]]:
        """Get current end-effector pose in meters and radians (consistent with script 0)."""
        if not self.connected or self.piper is None:
            return None
        
        try:
            end_pose = self.piper.GetArmEndPoseMsgs()
            
            # SDK unit: 0.001mm -> meters
            x = end_pose.end_pose.X_axis / 1_000_000.0
            y = end_pose.end_pose.Y_axis / 1_000_000.0
            z = end_pose.end_pose.Z_axis / 1_000_000.0
            
            # SDK unit: 0.001° -> radians
            rx = np.deg2rad(end_pose.end_pose.RX_axis / 1000.0)
            ry = np.deg2rad(end_pose.end_pose.RY_axis / 1000.0)
            rz = np.deg2rad(end_pose.end_pose.RZ_axis / 1000.0)
            
            return {
                'x': x, 'y': y, 'z': z,
                'rx': rx, 'ry': ry, 'rz': rz,
                'x_mm': x * 1000, 'y_mm': y * 1000, 'z_mm': z * 1000,
                'rx_deg': np.rad2deg(rx), 'ry_deg': np.rad2deg(ry), 'rz_deg': np.rad2deg(rz),
            }
        except Exception as e:
            print(f"[Piper] Error reading pose: {e}")
            return None
    
    def get_ee_pose_tuple(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Get current end-effector pose as tuple (x, y, z, rx_deg, ry_deg, rz_deg)."""
        pose = self.get_ee_pose_meters()
        if pose is None:
            return None
        return (pose['x'], pose['y'], pose['z'], pose['rx_deg'], pose['ry_deg'], pose['rz_deg'])
    
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
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_state = gripper_msgs.gripper_state
            # SDK unit: 0.001° -> degrees
            gripper_angle = gripper_state.grippers_angle / 1000.0
            # Normalize: 0° → 0, 70° → 1
            normalized = gripper_angle / GRIPPER_OPEN_ANGLE
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            print(f"[Piper] Error reading gripper: {e}")
            return None
    
    def get_gripper_status(self) -> Optional[Dict[str, Any]]:
        """Get detailed gripper status."""
        if not self.connected or self.piper is None:
            return None
        try:
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_state = gripper_msgs.gripper_state
            return {
                'angle': gripper_state.grippers_angle / 1000.0,  # degrees
                'effort': gripper_state.grippers_effort / 1000.0,
                'code': getattr(gripper_state, 'grippers_code', None),
            }
        except Exception as e:
            print(f"[Piper] Error reading gripper status: {e}")
            return None
    
    def move_to_pose(self, x: float, y: float, z: float,
                     rx: float, ry: float, rz: float,
                     speed_percent: int = None) -> bool:
        """Move to specified pose (verified method from script 0).
        
        Args:
            x, y, z: Position in meters
            rx, ry, rz: Orientation in radians
            speed_percent: Speed percentage (1-100)
        
        Returns:
            True if command was sent successfully
        """
        if not self.enabled or self.piper is None:
            print("[Piper] ✗ Not enabled, cannot move")
            return False
        
        if self._emergency_stop:
            print("[Piper] ✗ Emergency stop active, cannot move")
            return False
        
        # Safety check
        if z < WORKSPACE_LIMITS["z_min"]:
            print(f"[Piper] ✗ Z={z:.3f}m below safety limit {WORKSPACE_LIMITS['z_min']}m")
            return False
        if z > WORKSPACE_LIMITS["z_max"]:
            print(f"[Piper] ✗ Z={z:.3f}m above safety limit {WORKSPACE_LIMITS['z_max']}m")
            return False
        
        # Clamp to workspace limits
        x = max(WORKSPACE_LIMITS["x_min"], min(WORKSPACE_LIMITS["x_max"], x))
        y = max(WORKSPACE_LIMITS["y_min"], min(WORKSPACE_LIMITS["y_max"], y))
        
        speed = speed_percent if speed_percent else MOTION_SPEED_PERCENT
        
        try:
            # Ensure control mode
            self._ensure_control_mode()
            
            # Check enable status
            enable_list = self._get_enable_status()
            if enable_list is not None and not all(enable_list):
                print("[Piper] Re-enabling joints...")
                self.piper.EnableArm(7)
                time.sleep(0.2)
            
            # Convert to SDK units: meters -> 0.001mm, radians -> 0.001°
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
    
    def send_ee_pose(self, x: float, y: float, z: float, 
                     rx: float, ry: float, rz: float):
        """Send target end-effector pose (meters and degrees, legacy API).
        
        Args:
            x, y, z: Position in meters
            rx, ry, rz: Orientation in degrees (Euler XYZ)
        """
        # Convert degrees to radians for move_to_pose
        rx_rad = np.deg2rad(rx)
        ry_rad = np.deg2rad(ry)
        rz_rad = np.deg2rad(rz)
        self.move_to_pose(x, y, z, rx_rad, ry_rad, rz_rad)
    
    def send_ee_pose_quat(self, pos: np.ndarray, quat: np.ndarray):
        """Send target end-effector pose with quaternion orientation.
        
        Args:
            pos: Position [x, y, z] in meters
            quat: Quaternion [w, x, y, z]
        """
        rx, ry, rz = quat_to_euler(quat[0], quat[1], quat[2], quat[3])
        self.send_ee_pose(pos[0], pos[1], pos[2], rx, ry, rz)
    
    def set_gripper(self, gripper_value: float):
        """Set gripper position (normalized 0-1).
        
        Args:
            gripper_value: Normalized gripper position (0=close, 1=open)
        """
        if self._emergency_stop:
            return False
        
        gripper_value = max(0.0, min(1.0, gripper_value))
        angle_deg = gripper_value * GRIPPER_OPEN_ANGLE
        return self.set_gripper_angle(angle_deg, GRIPPER_EFFORT)
    
    def set_gripper_angle(self, angle_deg: float, effort: int = 500) -> bool:
        """Set gripper angle (verified method from script 0).
        
        Args:
            angle_deg: Gripper angle in degrees (0=close, 70=open)
            effort: Gripper force (0-1000)
        """
        if not self.enabled or self.piper is None:
            return False
        
        if self._emergency_stop:
            return False
        
        try:
            angle_sdk = int(angle_deg * 1000)  # degrees -> 0.001°
            self.piper.GripperCtrl(angle_sdk, effort, 0x01, 0)
            return True
        except Exception as e:
            print(f"[Piper] Gripper control failed: {e}")
            return False
    
    def open_gripper(self) -> bool:
        """Fully open the gripper."""
        return self.set_gripper_angle(GRIPPER_OPEN_ANGLE, GRIPPER_EFFORT)
    
    def close_gripper(self) -> bool:
        """Fully close the gripper."""
        return self.set_gripper_angle(GRIPPER_CLOSE_ANGLE, GRIPPER_EFFORT)
    
    def go_to_home(self) -> bool:
        """Move to calibrated home position (verified from script 0)."""
        if self._emergency_stop:
            return False
        if not self.enabled:
            return False
        
        print("[Piper] Going to home position...")
        x, y, z = self._home_position
        rx, ry, rz = self._home_orientation
        
        if not self.move_to_pose(x, y, z, rx, ry, rz):
            return False
        
        # Wait for motion to complete
        target = {'x': x, 'y': y, 'z': z, 'rx_deg': np.rad2deg(rx), 'ry_deg': np.rad2deg(ry), 'rz_deg': np.rad2deg(rz)}
        reached = self.wait_until_pose(target, MOTION_TIMEOUT, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)
        
        if reached:
            print(f"[Piper] Home position reached: ({x:.3f}, {y:.3f}, {z:.3f})m")
        return reached
    
    def wait_until_pose(self, target: Dict[str, float], timeout_s: float,
                        pos_tol_m: float, rot_tol_deg: float) -> bool:
        """Wait until end-effector reaches target pose."""
        start = time.time()
        while time.time() - start < timeout_s:
            if self._emergency_stop:
                return False
            
            pose = self.get_ee_pose_meters()
            if pose is None:
                time.sleep(0.05)
                continue
            
            # Check position
            pos_err = np.sqrt(
                (pose['x'] - target['x'])**2 +
                (pose['y'] - target['y'])**2 +
                (pose['z'] - target['z'])**2
            )
            
            # Check orientation
            rot_err = max(
                abs(pose['rx_deg'] - target['rx_deg']),
                abs(pose['ry_deg'] - target['ry_deg']),
                abs(pose['rz_deg'] - target['rz_deg'])
            )
            
            if pos_err < pos_tol_m and rot_err < rot_tol_deg:
                return True
            
            time.sleep(0.05)
        
        return False
    
    def wait_until_gripper(self, target_angle_deg: float, timeout_s: float,
                           tol_deg: float) -> bool:
        """Wait until gripper reaches target angle."""
        start = time.time()
        while time.time() - start < timeout_s:
            if self._emergency_stop:
                return False
            
            status = self.get_gripper_status()
            if status is None:
                time.sleep(0.05)
                continue
            
            if abs(status['angle'] - target_angle_deg) < tol_deg:
                return True
            
            time.sleep(0.05)
        
        return False
    
    def is_position_reached(self, target_pos: Tuple[float, float, float], 
                            tolerance: float = POSITION_TOLERANCE) -> bool:
        """Check if current position is within tolerance of target."""
        pose = self.get_ee_pose_meters()
        if pose is None:
            return False
        current_pos = np.array([pose['x'], pose['y'], pose['z']])
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
    # Force/torque data
    joint_angles: list = field(default_factory=list)   # List of (6,) arrays in radians
    joint_torques: list = field(default_factory=list)  # List of (6,) arrays in N·m
    ee_force: list = field(default_factory=list)       # List of (6,) arrays [Fx,Fy,Fz,Mx,My,Mz]
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
    
    # Convert force data to arrays
    joint_angles_arr = np.array(episode.joint_angles, dtype=np.float32) if episode.joint_angles else np.zeros((T, 6), dtype=np.float32)
    joint_torques_arr = np.array(episode.joint_torques, dtype=np.float32) if episode.joint_torques else np.zeros((T, 6), dtype=np.float32)
    ee_force_arr = np.array(episode.ee_force, dtype=np.float32) if episode.ee_force else np.zeros((T, 6), dtype=np.float32)
    
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
        # Force/torque data
        joint_angles=joint_angles_arr,
        joint_torques=joint_torques_arr,
        ee_force=ee_force_arr,
        success=episode.success,
        episode_id=episode.episode_id,
        num_timesteps=T,
    )
    
    print(f"[Save] Episode {episode.episode_id} saved to {episode_dir} ({T} timesteps)")


# =============================================================================
# Finite State Machine (Updated for script 0 compatibility)
# =============================================================================

class PickPlaceFSM:
    """FSM expert for Task B: Pick from plate → Place at random.
    
    Updated to use radians for orientation (consistent with script 0).
    """
    
    def __init__(self, piper: PiperController, 
                 plate_center: Tuple[float, float, float] = PLATE_CENTER,
                 grasp_height: float = GRASP_HEIGHT,
                 hover_height: float = HOVER_HEIGHT,
                 orientation_rad: Tuple[float, float, float] = DEFAULT_ORIENTATION_RAD):
        self.piper = piper
        self.plate_center = plate_center
        self.grasp_height = grasp_height
        self.hover_height = hover_height
        self.orientation_rad = orientation_rad  # (rx, ry, rz) in radians
        
        self.state = FSMState.IDLE
        self.place_position = None  # Random target (x, y, z)
        self._settle_start_time = None
        self._gripper_start_time = None
        
    def sample_random_place_position(self) -> Tuple[float, float, float]:
        """Sample a random placement position."""
        x = self.plate_center[0] + random.uniform(*RANDOM_RANGE_X)
        y = self.plate_center[1] + random.uniform(*RANDOM_RANGE_Y)
        z = self.grasp_height  # Use grasp height for Z
        return (x, y, z)
    
    def get_target_pose_rad(self) -> Tuple[float, float, float, float, float, float]:
        """Get target pose for current state (x, y, z, rx, ry, rz) in meters and radians."""
        rx, ry, rz = self.orientation_rad
        
        if self.state == FSMState.GO_TO_HOME:
            # Use calibrated home position
            hx, hy, hz = self.piper._home_position
            hrx, hry, hrz = self.piper._home_orientation
            return (hx, hy, hz, hrx, hry, hrz)
        
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
        
        elif self.state == FSMState.RETURN_HOME:
            # Return to calibrated home position
            hx, hy, hz = self.piper._home_position
            hrx, hry, hrz = self.piper._home_orientation
            return (hx, hy, hz, hrx, hry, hrz)
        
        else:
            # IDLE or DONE - hold current position
            pose = self.piper.get_ee_pose_meters()
            if pose:
                return (pose['x'], pose['y'], pose['z'], pose['rx'], pose['ry'], pose['rz'])
            return (self.plate_center[0], self.plate_center[1], self.hover_height, rx, ry, rz)
    
    def get_target_pose(self) -> Tuple[float, float, float, float, float, float]:
        """Get target pose for current state (x, y, z, rx, ry, rz) in meters and degrees.
        
        For backward compatibility with legacy code.
        """
        pose_rad = self.get_target_pose_rad()
        return (
            pose_rad[0], pose_rad[1], pose_rad[2],
            np.rad2deg(pose_rad[3]), np.rad2deg(pose_rad[4]), np.rad2deg(pose_rad[5])
        )
    
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
        print(f"[FSM] Starting episode. Place position: ({self.place_position[0]:.3f}, {self.place_position[1]:.3f}, {self.place_position[2]:.3f})m")
    
    def step(self) -> bool:
        """Execute one step of the FSM.
        
        Returns:
            True if episode is done (success or abort)
        """
        if self.state == FSMState.IDLE:
            return False
        
        if self.state == FSMState.DONE:
            return True
        
        # Get target pose in radians and send
        target_rad = self.get_target_pose_rad()
        self.piper.move_to_pose(*target_rad)
        
        # Handle gripper
        gripper_target = self.get_target_gripper()
        self.piper.set_gripper(gripper_target)
        
        # Check transition conditions
        self._check_transition(target_rad[:3])
        
        return self.state == FSMState.DONE
    
    def _check_transition(self, target_pos: Tuple[float, float, float]):
        """Check if we should transition to next state."""
        position_reached = self.piper.is_position_reached(target_pos, POSITION_TOLERANCE)
        
        # Handle settling time for movement states
        movement_states = [FSMState.GO_TO_HOME, FSMState.HOVER_PLATE, FSMState.LOWER_GRASP,
                          FSMState.LIFT_OBJECT, FSMState.HOVER_PLACE, FSMState.LOWER_PLACE,
                          FSMState.LIFT_RETREAT, FSMState.RETURN_HOME]
        
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
            FSMState.LIFT_RETREAT: FSMState.RETURN_HOME,
            FSMState.RETURN_HOME: FSMState.DONE,
        }
        
        old_state = self.state
        self.state = transitions.get(self.state, FSMState.DONE)
        print(f"[FSM] {old_state.name} → {self.state.name}")


# =============================================================================
# Keyboard Input Handler (Updated from script 0)
# =============================================================================

class KeyboardHandler:
    """Keyboard input handler using terminal input mode (verified from script 0)."""
    
    def __init__(self):
        self.start_requested = False
        self.quit_requested = False
        self.reset_requested = False
        self.skip_requested = False
        self.emergency_stop_requested = False
        self._lock = threading.Lock()
        self._keyboard_available = False
        self._running = True
        self._last_key: Optional[str] = None
    
    def start(self):
        """Start keyboard listening using terminal input mode."""
        print("[Keyboard] Using terminal input mode - type command and press Enter")
        print("[Keyboard] Commands: SPACE=start, ESC=e-stop, Q=quit, R=reset, S=skip")
        self._keyboard_available = True
        
        def input_thread():
            while self._running:
                try:
                    user_input = input().strip().lower()
                    with self._lock:
                        self._last_key = user_input
                        # Map common inputs to commands
                        if user_input in ['', ' ', 'space', 'start']:
                            self.start_requested = True
                        elif user_input in ['q', 'quit', 'exit']:
                            self.quit_requested = True
                        elif user_input in ['r', 'reset']:
                            self.reset_requested = True
                        elif user_input in ['s', 'skip']:
                            self.skip_requested = True
                        elif user_input in ['esc', 'escape', 'stop', 'e']:
                            self.emergency_stop_requested = True
                except EOFError:
                    break
                except Exception:
                    break
        
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()
    
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
    
    def stop(self):
        """Stop keyboard listening."""
        self._running = False


# =============================================================================
# Main Data Collection Loop
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect pick-and-place data with Piper arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default cameras
  python 1_collect_data_piper.py --num_episodes 50 --out_dir data/piper_pick_place
  
  # Specify camera by name (recommended)
  python 1_collect_data_piper.py --front_cam Orbbec_Gemini_335L --wrist_cam Dabai_DC1
  
  # Disable wrist camera
  python 1_collect_data_piper.py --wrist_cam -1
  
  # Custom plate center position
  python 1_collect_data_piper.py --plate_x 0.25 --plate_y 0.0 --grasp_height 0.16
"""
    )
    
    # Hardware settings
    parser.add_argument("--can_interface", "-c", type=str, default="can0",
                        help="CAN interface name (default: can0)")
    parser.add_argument("--front_cam", "-f", type=str, default=DEFAULT_FRONT_CAMERA,
                        help=f"Front camera ID/name (default: {DEFAULT_FRONT_CAMERA})")
    parser.add_argument("--wrist_cam", "-w", type=str, default=DEFAULT_WRIST_CAMERA,
                        help=f"Wrist camera ID/name, use -1 to disable (default: {DEFAULT_WRIST_CAMERA})")
    
    # Data collection settings
    parser.add_argument("--num_episodes", "-n", type=int, default=50,
                        help="Number of episodes to collect (default: 50)")
    parser.add_argument("--out_dir", "-o", type=str, default="data/piper_pick_place",
                        help="Output directory for collected data")
    parser.add_argument("--control_freq", type=int, default=CONTROL_FREQ,
                        help=f"Control frequency in Hz (default: {CONTROL_FREQ})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    # Image settings
    parser.add_argument("--image_width", type=int, default=640,
                        help="Camera image width (default: 640)")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Camera image height (default: 480)")
    
    # Workspace parameters (calibrated from script 0)
    parser.add_argument("--plate_x", type=float, default=PLATE_CENTER[0],
                        help=f"Pick position X in meters (default: {PLATE_CENTER[0]})")
    parser.add_argument("--plate_y", type=float, default=PLATE_CENTER[1],
                        help=f"Pick position Y in meters (default: {PLATE_CENTER[1]})")
    parser.add_argument("--grasp_height", type=float, default=GRASP_HEIGHT,
                        help=f"Grasp Z height in meters (default: {GRASP_HEIGHT})")
    parser.add_argument("--hover_height", type=float, default=HOVER_HEIGHT,
                        help=f"Hover Z height in meters (default: {HOVER_HEIGHT})")
    parser.add_argument("--speed", "-s", type=int, default=MOTION_SPEED_PERCENT,
                        help=f"Motion speed percentage (default: {MOTION_SPEED_PERCENT})")
    
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
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize camera values
    front_cam_id = normalize_camera_value(args.front_cam)
    wrist_cam_id = normalize_camera_value(args.wrist_cam)
    
    # Update parameters from args
    plate_center = (args.plate_x, args.plate_y, args.grasp_height)
    grasp_height = args.grasp_height
    hover_height = args.hover_height
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Piper Pick & Place Data Collection")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  CAN Interface: {args.can_interface}")
    print(f"  Front Camera:  {front_cam_id}")
    print(f"  Wrist Camera:  {wrist_cam_id}" + (" (disabled)" if not is_camera_enabled(wrist_cam_id) else ""))
    print(f"  Output Dir:    {out_dir}")
    print(f"  Episodes:      {args.num_episodes}")
    print(f"  Control Freq:  {args.control_freq} Hz")
    print(f"  Motion Speed:  {args.speed}%")
    print(f"\nWorkspace Parameters:")
    print(f"  Pick Position: ({args.plate_x:.3f}, {args.plate_y:.3f})m")
    print(f"  Grasp Height:  {grasp_height:.3f}m")
    print(f"  Hover Height:  {hover_height:.3f}m")
    print(f"  Random Range:  X={RANDOM_RANGE_X}, Y={RANDOM_RANGE_Y}")
    print()
    
    # Initialize keyboard handler
    keyboard_handler = KeyboardHandler()
    keyboard_handler.start()
    
    # Initialize cameras
    print("\n[Init] Starting cameras...")
    fixed_cam = CameraCapture(front_cam_id, args.image_width, args.image_height, name="front")
    wrist_cam = None
    wrist_cam_enabled = is_camera_enabled(wrist_cam_id)
    
    try:
        fixed_cam.start()
        if wrist_cam_enabled:
            wrist_cam = CameraCapture(wrist_cam_id, args.image_width, args.image_height, name="wrist")
            try:
                wrist_cam.start()
            except RuntimeError as e:
                print(f"[Warning] Wrist camera initialization failed: {e}")
                print("[Warning] Continuing without wrist camera...")
                wrist_cam = None
                wrist_cam_enabled = False
        else:
            print("[Camera] Wrist camera disabled")
    except RuntimeError as e:
        print(f"[Error] Front camera initialization failed: {e}")
        print("Please check camera connections and device IDs.")
        keyboard_handler.stop()
        return
    
    # Initialize Piper arm (using verified method from script 0)
    print("\n[Init] Connecting to Piper arm...")
    piper = PiperController(args.can_interface)
    
    try:
        if not piper.connect():
            print("[Error] Piper connection failed")
            keyboard_handler.stop()
            fixed_cam.stop()
            if wrist_cam:
                wrist_cam.stop()
            return
        
        if not piper.enable():
            print("[Error] Piper enable failed")
            keyboard_handler.stop()
            fixed_cam.stop()
            if wrist_cam:
                wrist_cam.stop()
            return
    except Exception as e:
        print(f"[Error] Piper initialization failed: {e}")
        keyboard_handler.stop()
        fixed_cam.stop()
        if wrist_cam:
            wrist_cam.stop()
        return
    
    # Go to home position
    print("\n[Init] Going to home position...")
    piper.go_to_home()
    time.sleep(1.0)
    
    # Open gripper
    piper.open_gripper()
    time.sleep(0.5)
    
    # Initialize FSM with calibrated parameters
    fsm = PickPlaceFSM(
        piper,
        plate_center=plate_center,
        grasp_height=grasp_height,
        hover_height=hover_height,
        orientation_rad=DEFAULT_ORIENTATION_RAD
    )
    
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
    
    # Control loop parameters
    control_period = 1.0 / args.control_freq
    
    # Episode tracking
    episode_count = 0
    current_episode: Optional[EpisodeData] = None
    recording = False
    
    # Goal pose (fixed plate center in quaternion format)
    goal_quat = euler_to_quat(*DEFAULT_ORIENTATION_EULER)
    goal_pose = np.array([plate_center[0], plate_center[1], grasp_height,
                          goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3]], dtype=np.float32)
    
    print("\n" + "=" * 60)
    print("Ready to collect data!")
    print("Press ENTER or type 'start' to start an episode.")
    print("Type 'q' to quit, 'r' to reset, 's' to skip, 'e' for emergency stop.")
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
                    wrist_img = wrist_cam.get_frame() if wrist_cam else None
                    
                    # Get pose using the new dict-based method
                    pose_dict = piper.get_ee_pose_meters()
                    if pose_dict is not None:
                        pos = np.array([pose_dict['x'], pose_dict['y'], pose_dict['z']])
                        quat = euler_to_quat(pose_dict['rx_deg'], pose_dict['ry_deg'], pose_dict['rz_deg'])
                        quat = np.array(quat)
                    else:
                        pos = np.zeros(3)
                        quat = np.array([1, 0, 0, 0])
                    
                    gripper = piper.get_gripper_state() or 0.0
                    
                    # Construct ee_pose [x, y, z, qw, qx, qy, qz]
                    ee_pose = np.concatenate([pos, quat])
                    
                    # Get action (target pose from FSM + target gripper)
                    target = fsm.get_target_pose()  # returns (x, y, z, rx_deg, ry_deg, rz_deg)
                    target_quat = euler_to_quat(target[3], target[4], target[5])
                    target_gripper = fsm.get_target_gripper()
                    action = np.array([target[0], target[1], target[2],
                                       target_quat[0], target_quat[1], target_quat[2], target_quat[3],
                                       target_gripper], dtype=np.float32)
                    
                    # Get force/torque data
                    joint_angles = np.zeros(6)
                    joint_torques = np.zeros(6)
                    ee_force = np.zeros(6)
                    
                    if force_estimator is not None and piper.piper is not None:
                        joint_state = force_estimator.get_joint_state_from_piper(piper.piper)
                        if joint_state is not None:
                            joint_angles, joint_torques = joint_state
                            ee_force = force_estimator.estimate_force(
                                joint_angles, joint_torques,
                                use_filter=True,
                                use_gravity_comp=True
                            )
                    
                    # Record (handle optional wrist camera)
                    if fixed_img is not None:
                        current_episode.fixed_images.append(fixed_img)
                        # Use placeholder if no wrist camera
                        if wrist_img is not None:
                            current_episode.wrist_images.append(wrist_img)
                        else:
                            current_episode.wrist_images.append(np.zeros_like(fixed_img))
                        current_episode.ee_pose.append(ee_pose)
                        current_episode.gripper_state.append(gripper)
                        current_episode.action.append(action)
                        current_episode.fsm_state.append(fsm.state.value)
                        current_episode.timestamp.append(time.time())
                        # Record force data
                        current_episode.joint_angles.append(joint_angles)
                        current_episode.joint_torques.append(joint_torques)
                        current_episode.ee_force.append(ee_force)
                
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
        keyboard_handler.stop()
        
        if piper.enabled and not piper._emergency_stop:
            piper.open_gripper()
            time.sleep(0.5)
            piper.go_to_home()
            time.sleep(2.0)
        
        piper.disconnect()
        fixed_cam.stop()
        if wrist_cam:
            wrist_cam.stop()
        
        # Save collection metadata
        metadata = {
            "num_episodes": episode_count,
            "control_freq": args.control_freq,
            "image_size": [args.image_height, args.image_width],
            "plate_center": list(plate_center),
            "grasp_height": grasp_height,
            "hover_height": hover_height,
            "random_range_x": list(RANDOM_RANGE_X),
            "random_range_y": list(RANDOM_RANGE_Y),
            "home_position": list(HOME_POSITION),
            "home_orientation_rad": list(HOME_ORIENTATION),
            "grasp_orientation_rad": list(DEFAULT_ORIENTATION_RAD),
            "seed": args.seed,
            "front_camera": str(front_cam_id),
            "wrist_camera": str(wrist_cam_id),
            "wrist_camera_enabled": wrist_cam_enabled,
            "force_estimation_enabled": FORCE_ESTIMATOR_AVAILABLE and force_estimator is not None,
        }
        
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[Main] Data collection complete!")
        print(f"  Episodes collected: {episode_count}")
        print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
