#!/usr/bin/env python3
"""Script 2: PS5 Controller Teleoperation for Piper Arm.

This script allows teleoperation of the Piper robotic arm using a PS5 DualSense
controller connected via Bluetooth.

=============================================================================
SETUP INSTRUCTIONS
=============================================================================
1. Put PS5 controller in pairing mode:
   - Hold PS button + Create button until LED flashes rapidly
2. On Linux, pair via Bluetooth:
   - Open Bluetooth settings and connect to "DualSense Wireless Controller"
3. Install pygame if not already installed:
   - pip install pygame

=============================================================================
CONTROLLER MAPPING
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
| Square (▢)        | Print current pose and record position              |
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

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
import warnings
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

# Try to import pydualsense for gyro support
try:
    from pydualsense import pydualsense
    PYDUALSENSE_AVAILABLE = True
except ImportError:
    PYDUALSENSE_AVAILABLE = False

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
    success: bool = True


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
    
    # Save NPZ
    np.savez(
        episode_dir / "episode_data.npz",
        ee_pose=ee_pose_arr,
        gripper_state=gripper_arr,
        action=action_arr,
        timestamp=timestamp_arr,
        success=episode.success,
        episode_id=episode.episode_id,
        num_timesteps=T,
    )
    
    print(f"[Save] Episode {episode.episode_id} saved to {episode_dir} ({T} timesteps)")


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
    """Handler for PS5 DualSense controller using pygame + pydualsense for gyro."""
    
    def __init__(self):
        self.joystick: Optional[pygame.joystick.JoystickType] = None
        self.connected = False
        self.deadzone = 0.1
        
        # pydualsense for gyro/accelerometer
        self._dualsense = None
        self._gyro_available = False
        
        # Gyro calibration (offset to subtract)
        self._gyro_offset = [0.0, 0.0, 0.0]  # [pitch, yaw, roll] baseline
        self._gyro_calibrated = False
        self._gyro_dynamic_deadzone = 1.5  # Will be set during calibration (rad/s)
        
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
                # Check for gyro support
                if js.get_numaxes() >= 9:
                    print(f"[Controller]   ✓ Extended axes detected (may include gyro)")
                else:
                    print(f"[Controller]   ⚠ Only {js.get_numaxes()} axes - gyro may not be exposed")
                    print(f"[Controller]   Tip: Try setting SDL_JOYSTICK_HIDAPI_PS5=1 environment variable")
                
                # Initialize pydualsense for gyro if available
                self._init_pydualsense()
                return True
        
        # If no PS5 found, use first joystick
        if num_joysticks > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
            print(f"[Controller] ⚠ No PS5 controller found, using: {self.joystick.get_name()}")
            return True
        
        return False
    
    def _init_pydualsense(self):
        """Initialize pydualsense for gyro support."""
        if not PYDUALSENSE_AVAILABLE:
            print("[Controller] pydualsense not installed - gyro disabled")
            print("[Controller] To enable gyro: pip install pydualsense")
            return
        
        try:
            self._dualsense = pydualsense()
            self._dualsense.init()
            self._gyro_available = True
            print("[Controller] ✓ pydualsense initialized - gyro enabled!")
            
            # Auto-calibrate gyro
            self.calibrate_gyro()
        except Exception as e:
            print(f"[Controller] ⚠ pydualsense init failed: {e}")
            print("[Controller] Gyro will not be available")
            self._dualsense = None
            self._gyro_available = False
    
    def calibrate_gyro(self, samples: int = 100):
        """Calibrate gyro by measuring baseline angular velocity offset when stationary.
        
        The gyroscope returns ANGULAR VELOCITY (rate of rotation), not angle.
        When stationary, ideal reading is 0, but real sensors have bias/offset.
        This calibration measures that bias to subtract it during operation.
        
        Note: Large offset values (e.g., 8000) in raw units are normal - 
        they represent small angular velocity bias (~500 deg/s worth of LSB,
        but actual bias is much smaller due to sensor characteristics).
        
        Args:
            samples: Number of samples to average for calibration.
        """
        if not self._gyro_available or self._dualsense is None:
            return
        
        print("[Gyro] Calibrating angular velocity bias... keep controller COMPLETELY still!")
        print("[Gyro] (Best results with USB connection)")
        time.sleep(0.5)  # Give user time to stabilize
        
        readings = []
        
        for i in range(samples):
            try:
                # Read RAW values directly (no scaling)
                pitch = self._dualsense.state.gyro.Pitch
                yaw = self._dualsense.state.gyro.Yaw
                roll = self._dualsense.state.gyro.Roll
                readings.append((pitch, yaw, roll))
                time.sleep(0.025)  # 40Hz sampling, ~2.5 seconds total
            except:
                pass
        
        if len(readings) < 10:
            print("[Gyro] ✗ Calibration failed - not enough samples")
            return
        
        # Use median instead of mean to reject outliers
        readings = np.array(readings)
        self._gyro_offset[0] = np.median(readings[:, 0])
        self._gyro_offset[1] = np.median(readings[:, 1])
        self._gyro_offset[2] = np.median(readings[:, 2])
        
        # Compute std to estimate noise level
        std_pitch = np.std(readings[:, 0])
        std_yaw = np.std(readings[:, 1])
        std_roll = np.std(readings[:, 2])
        max_std = max(std_pitch, std_yaw, std_roll)
        
        # Set deadzone based on noise (15 sigma for safety)
        # USB: std ~10, so deadzone ~200 raw units
        # Bluetooth: std ~4000, so deadzone ~60000 raw units
        self._gyro_dynamic_deadzone = max(200.0, max_std * 15)
        
        self._gyro_calibrated = True
        
        print(f"[Gyro] ✓ Calibrated! Offset: pitch={self._gyro_offset[0]:.0f}, "
              f"yaw={self._gyro_offset[1]:.0f}, roll={self._gyro_offset[2]:.0f} (raw)")
        print(f"[Gyro] Noise std: {max_std:.1f}, deadzone: {self._gyro_dynamic_deadzone:.0f} (raw)")
        
        if max_std > 500:
            print(f"[Gyro] ⚠ HIGH NOISE ({max_std:.0f})! Use USB cable for better results.")
        elif max_std < 50:
            print(f"[Gyro] ✓ Low noise - good connection!")
    
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
    
    def get_gyro_raw(self, debug: bool = False) -> Optional[Tuple[float, float, float]]:
        """Get calibrated gyroscope ANGULAR VELOCITY data (pitch, yaw, roll).
        
        IMPORTANT: Returns ANGULAR VELOCITY (rate of rotation), NOT angle!
        Values are in RAW units with bias offset removed and deadzone applied.
        To convert to deg/s: raw_value / 16.4 (approx DualSense sensitivity)
        To get angle change: angular_velocity * dt
        
        Typical range: -10000 to +10000 raw units for moderate rotation speed.
        
        Returns:
            Tuple of (pitch, yaw, roll) angular velocities in raw units, or None.
        """
        if not self._gyro_available or self._dualsense is None:
            if debug:
                print("[Gyro] ✗ pydualsense not available")
            return None
        
        try:
            # Read RAW values and subtract calibrated offset
            gyro_pitch = self._dualsense.state.gyro.Pitch - self._gyro_offset[0]
            gyro_yaw = self._dualsense.state.gyro.Yaw - self._gyro_offset[1]
            gyro_roll = self._dualsense.state.gyro.Roll - self._gyro_offset[2]
            
            # Clamp extreme values (likely noise spikes)
            max_valid = 20000.0  # Max reasonable rotation
            gyro_pitch = np.clip(gyro_pitch, -max_valid, max_valid)
            gyro_yaw = np.clip(gyro_yaw, -max_valid, max_valid)
            gyro_roll = np.clip(gyro_roll, -max_valid, max_valid)
            
            # Apply dynamic deadzone from calibration
            dz = self._gyro_dynamic_deadzone
            if abs(gyro_pitch) < dz:
                gyro_pitch = 0.0
            if abs(gyro_yaw) < dz:
                gyro_yaw = 0.0
            if abs(gyro_roll) < dz:
                gyro_roll = 0.0
            
            if debug:
                print(f"[Gyro] pitch={gyro_pitch:+8.0f}  yaw={gyro_yaw:+8.0f}  roll={gyro_roll:+8.0f} (raw)")
            
            return (gyro_pitch, gyro_yaw, gyro_roll)
        except Exception as e:
            if debug:
                print(f"[Gyro] Error reading: {e}")
            return None
    
    def get_gyro(self, debug: bool = False) -> Optional[Tuple[float, float, float]]:
        """Get raw gyroscope data. Alias for get_gyro_raw for compatibility."""
        return self.get_gyro_raw(debug)
    
    def close(self):
        """Close pygame and pydualsense."""
        if self._dualsense is not None:
            try:
                self._dualsense.close()
            except:
                pass
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
    
    def __init__(self, piper: PiperController, ps5: PS5Controller,
                 linear_scale: float = LINEAR_VEL_SCALE,
                 angular_scale: float = ANGULAR_VEL_SCALE,
                 control_freq: float = 30.0,
                 # Data recording parameters
                 record_data: bool = False,
                 fixed_cam: Optional[CameraCapture] = None,
                 wrist_cam: Optional[CameraCapture] = None):
        self.piper = piper
        self.ps5 = ps5
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
        self._control_freq = control_freq
        self._control_dt = 1.0 / control_freq  # Time step for integration
        
        self._gripper_open = True
        self._gripper_target_angle = GRIPPER_OPEN_ANGLE  # Track target gripper angle
        self._gripper_speed = 3.0  # Degrees per control cycle when trigger pressed
        # Gyro returns ANGULAR VELOCITY (rate of rotation), not angle!
        # DualSense gyro: ~16.4 LSB per deg/s, so raw_value / 16.4 ≈ deg/s
        # Convert to rad/s: (raw / 16.4) * (π/180) = raw * 0.00106
        # Then multiply by dt to get angle change: Δθ = ω * dt
        self._gyro_lsb_per_deg_s = 16.4  # DualSense gyro sensitivity
        self._gyro_sensitivity = 1.5  # User-adjustable sensitivity multiplier
        self._running = False
        self._recorded_poses = []  # Store recorded poses
        
        # Low-pass filter for gyro (exponential moving average)
        self._gyro_alpha = 0.15  # Smoothing factor (higher = more responsive, lower = smoother)
        self._gyro_filtered = [0.0, 0.0, 0.0]  # [pitch, yaw, roll]
        
        # Dynamic gyro calibration: re-calibrate when entering gyro mode
        self._gyro_mode_active = False
        self._gyro_recal_samples = []  # Samples for dynamic recalibration
        self._gyro_recal_frames = 15   # Number of frames to collect for recalibration
        
        # Data recording
        self._record_data = record_data
        self._fixed_cam = fixed_cam
        self._wrist_cam = wrist_cam
        self._recording = False
        self._current_episode: Optional[EpisodeData] = None
        self._episode_count = 0
        self._prev_pose = None  # For computing delta action
        
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
        )
        self._recording = True
        self._prev_pose = None
        print(f"[Teleop] 🔴 Recording episode {episode_id}...")
        
    def stop_recording(self, success: bool = True) -> Optional[EpisodeData]:
        """Stop recording and return the episode data."""
        if not self._recording:
            return None
        
        self._recording = False
        episode = self._current_episode
        if episode is not None:
            episode.success = success
            self._episode_count += 1
            print(f"[Teleop] ⬛ Episode {episode.episode_id} stopped ({len(episode.ee_pose)} frames)")
        self._current_episode = None
        self._prev_pose = None
        return episode
        
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
        print("  Square:        Print & record pose")
        print("  Share:         Emergency stop/Resume")
        print("  PS Button:     Quit")
        if self._record_data:
            print("\nData Recording:")
            print("  L3 (Left Stick Press):  Start/Stop recording episode")
            print("  R3 (Right Stick Press): Discard current episode")
        print("=" * 60 + "\n")
    
    def step(self) -> bool:
        """Execute one control step.
        
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
        
        # Square button - print and record pose (was Triangle)
        if self.ps5.get_button_pressed(PS5Buttons.SQUARE):
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
            return True
        
        # Speed adjustment (L1/R1)
        if self.ps5.get_button_pressed(PS5Buttons.L1):
            self.piper.set_speed(self.piper._speed_percent - 5)
        if self.ps5.get_button_pressed(PS5Buttons.R1):
            self.piper.set_speed(self.piper._speed_percent + 5)
        
        # Recording controls (L3/R3)
        if self._record_data:
            # L3: Start/Stop recording
            if self.ps5.get_button_pressed(PS5Buttons.L3):
                if self._recording:
                    # Stop recording (episode will be saved by main loop)
                    self._recording = False
                    print(f"[Teleop] ⬛ Stopped recording ({len(self._current_episode.ee_pose) if self._current_episode else 0} frames)")
                else:
                    self.start_recording()
            
            # R3: Discard current episode
            if self.ps5.get_button_pressed(PS5Buttons.R3):
                if self._recording:
                    self._recording = False
                    self._current_episode = None
                    self._prev_pose = None
                    print("[Teleop] 🗑️ Episode discarded")
        
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
        
        # Handle gyro mode activation/deactivation with dynamic recalibration
        if circle_held and not self._gyro_mode_active:
            # Just entered gyro mode - start recalibration
            self._gyro_mode_active = True
            self._gyro_recal_samples = []
            self._gyro_filtered = [0.0, 0.0, 0.0]  # Reset filter
            print("[Gyro] 🎯 Entering gyro mode - hold still for calibration...")
        elif not circle_held and self._gyro_mode_active:
            # Exited gyro mode
            self._gyro_mode_active = False
            self._gyro_recal_samples = []
        
        # Debug: print gyro status when Circle is first pressed
        if circle_held and not getattr(self, '_gyro_debug_printed', False):
            print("[Teleop] 🎮 Gyro mode activated - checking gyro availability...")
            gyro = self.ps5.get_gyro(debug=True)  # Print debug info
            if gyro is None:
                print("[Teleop] ⚠ Gyro not available!")
                print("[Teleop] To enable gyro: pip install pydualsense libhidapi-hidraw0")
                print("[Teleop] Fallback: using right stick X for yaw rotation")
            else:
                print(f"[Teleop] ✓ Gyro working! Tilt controller to rotate end-effector")
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
            # Gyro mode: use controller's gyroscope for orientation
            # IMPORTANT: Gyro returns ANGULAR VELOCITY (deg/s in raw units), not angle!
            # We need to integrate: Δangle = angular_velocity × dt
            gyro = self.ps5.get_gyro_raw()
            if gyro is not None:
                gyro_pitch, gyro_yaw, gyro_roll = gyro
                
                # Dynamic recalibration: collect samples for first N frames
                # This compensates for gyro drift since initial calibration
                if len(self._gyro_recal_samples) < self._gyro_recal_frames:
                    self._gyro_recal_samples.append((gyro_pitch, gyro_yaw, gyro_roll))
                    if len(self._gyro_recal_samples) == self._gyro_recal_frames:
                        # Calculate and apply dynamic offset correction
                        samples = np.array(self._gyro_recal_samples)
                        recal_offset = np.median(samples, axis=0)
                        # Update the PS5 controller's offset directly
                        self.ps5._gyro_offset[0] += recal_offset[0]
                        self.ps5._gyro_offset[1] += recal_offset[1]
                        self.ps5._gyro_offset[2] += recal_offset[2]
                        print(f"[Gyro] ✓ Recalibrated! Drift correction: "
                              f"({recal_offset[0]:+.0f}, {recal_offset[1]:+.0f}, {recal_offset[2]:+.0f})")
                    # Skip control during calibration
                    target_rx = pose['rx']
                    target_ry = pose['ry']
                    target_rz = pose['rz']
                else:
                    # Normal gyro control after recalibration
                    # Apply low-pass filter (exponential moving average)
                    self._gyro_filtered[0] = self._gyro_alpha * gyro_pitch + (1 - self._gyro_alpha) * self._gyro_filtered[0]
                    self._gyro_filtered[1] = self._gyro_alpha * gyro_yaw + (1 - self._gyro_alpha) * self._gyro_filtered[1]
                    self._gyro_filtered[2] = self._gyro_alpha * gyro_roll + (1 - self._gyro_alpha) * self._gyro_filtered[2]
                    
                    # Use filtered values (still in raw units = angular velocity)
                    filtered_pitch = self._gyro_filtered[0]
                    filtered_yaw = self._gyro_filtered[1]
                    filtered_roll = self._gyro_filtered[2]
                    
                    # Convert raw gyro (angular velocity) to angle change:
                    # 1. raw_value / 16.4 = deg/s
                    # 2. deg/s * (π/180) = rad/s  
                    # 3. rad/s * dt = rad (angle change)
                    # Combined: raw * (1/16.4) * (π/180) * dt * sensitivity
                    gyro_to_rad = (1.0 / self._gyro_lsb_per_deg_s) * (np.pi / 180.0) * self._control_dt * self._gyro_sensitivity
                    
                    # Map controller gyro to arm rotation (incremental)
                    # Controller pitch (tilt forward/back) -> arm RX or RY
                    # Controller yaw (rotate horizontally) -> arm RZ
                    # Controller roll (tilt left/right) -> arm RX or RY
                    delta_rx = filtered_pitch * gyro_to_rad
                    delta_ry = -filtered_roll * gyro_to_rad  # Negated for correct direction
                    delta_rz = filtered_yaw * gyro_to_rad
                    
                    target_rx = pose['rx'] + delta_rx
                    target_ry = pose['ry'] + delta_ry
                    target_rz = pose['rz'] + delta_rz
                    
                    # Periodic debug print (every ~1 second at 30Hz)
                    if not hasattr(self, '_gyro_print_counter'):
                        self._gyro_print_counter = 0
                    self._gyro_print_counter += 1
                    if self._gyro_print_counter % 30 == 0:
                        # Show angular velocity in deg/s for intuition
                        vel_pitch = filtered_pitch / self._gyro_lsb_per_deg_s
                        vel_yaw = filtered_yaw / self._gyro_lsb_per_deg_s
                        vel_roll = filtered_roll / self._gyro_lsb_per_deg_s
                        print(f"[Gyro] vel=({vel_pitch:+.1f}, {vel_yaw:+.1f}, {vel_roll:+.1f}) deg/s  "
                              f"Δ=({np.rad2deg(delta_rx):+.2f}, {np.rad2deg(delta_ry):+.2f}, {np.rad2deg(delta_rz):+.2f}) deg")
            else:
                # Gyro not available, use right stick X for yaw as fallback
                target_rx = pose['rx']
                target_ry = pose['ry']
                target_rz = pose['rz'] + right_x * self.angular_scale  # Fallback: right stick for yaw
        else:
            # Normal mode: D-pad and right stick for rotation
            target_rx = pose['rx'] + dpad_roll * self.angular_scale * 0.5
            target_ry = pose['ry'] + dpad_pitch * self.angular_scale * 0.5
            target_rz = pose['rz'] + right_x * self.angular_scale
        
        # Apply movement if any input is active
        has_position_input = any(abs(v) > 0.01 for v in [left_stick_x, left_stick_y, right_y])
        has_rotation_input = circle_held or any(abs(v) > 0.01 for v in [right_x, dpad_pitch, dpad_roll])
        
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
            
            # Record data
            self._current_episode.fixed_images.append(fixed_img)
            self._current_episode.wrist_images.append(wrist_img)
            self._current_episode.ee_pose.append(ee_pose)
            self._current_episode.gripper_state.append(gripper_state)
            self._current_episode.action.append(action)
            self._current_episode.timestamp.append(time.time())
            
            # Update previous pose for next delta computation
            self._prev_pose = {
                'x': target_x, 'y': target_y, 'z': target_z,
                'rx': target_rx, 'ry': target_ry, 'rz': target_rz,
            }
        
        return True
    
    def stop(self):
        """Stop teleoperation."""
        self._running = False


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
  - L3 (Left Stick Press):  Start/Stop recording episode
  - R3 (Right Stick Press): Discard current episode
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
    args = parse_args()
    
    if not PYGAME_AVAILABLE:
        print("[Error] pygame is required. Install with: pip install pygame")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Piper PS5 Teleoperation")
    if args.record:
        print("📹 Data Recording Mode ENABLED")
    print("=" * 60)
    
    # Create output directory if recording
    out_dir = None
    if args.record:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nData will be saved to: {out_dir}")
    
    # Initialize PS5 controller
    print("\n[Init] Initializing PS5 controller...")
    ps5 = PS5Controller()
    ps5.deadzone = args.deadzone
    
    if not ps5.initialize():
        print("[Error] Failed to initialize controller")
        sys.exit(1)
    
    # Initialize Piper arm
    print("\n[Init] Connecting to Piper arm...")
    piper = PiperController(args.can_interface)
    piper._speed_percent = args.speed
    
    if not piper.connect():
        print("[Error] Failed to connect to Piper arm")
        ps5.close()
        sys.exit(1)
    
    if not piper.enable():
        print("[Error] Failed to enable Piper arm")
        piper.disconnect()
        ps5.close()
        sys.exit(1)
    
    # Go to home position
    print("\n[Init] Going to home position...")
    piper.go_to_home()
    time.sleep(2.0)
    
    # Open gripper
    piper.open_gripper()
    time.sleep(0.5)
    
    # Initialize cameras for recording
    fixed_cam = None
    wrist_cam = None
    display_cam = None  # For --show_camera option
    
    if args.record:
        # Front camera (required for recording)
        front_cam_id = normalize_camera_value(args.front_cam)
        print(f"\n[Init] Starting front camera: {front_cam_id}")
        fixed_cam = CameraCapture(front_cam_id, args.image_width, args.image_height, name="front")
        if not fixed_cam.start():
            print("[Warning] Front camera failed to start")
            fixed_cam = None
        
        # Wrist camera (optional)
        wrist_cam_id = normalize_camera_value(args.wrist_cam)
        wrist_cam_enabled = is_camera_enabled(wrist_cam_id)
        if wrist_cam_enabled:
            print(f"[Init] Starting wrist camera: {wrist_cam_id}")
            wrist_cam = CameraCapture(wrist_cam_id, args.image_width, args.image_height, name="wrist")
            if not wrist_cam.start():
                print("[Warning] Wrist camera failed to start")
                wrist_cam = None
        
        # Use front camera for display if --show_camera
        if args.show_camera:
            display_cam = fixed_cam
    elif args.show_camera and CV2_AVAILABLE:
        # Just display camera, no recording
        print("\n[Init] Starting camera for display...")
        display_cam = CameraCapture(args.camera_id, name="display")
        if not display_cam.start():
            print("[Init] ⚠ Camera failed to start, continuing without display")
            display_cam = None
    
    # Create teleoperation controller
    teleop = TeleoperationController(
        piper, ps5,
        linear_scale=args.linear_scale,
        angular_scale=args.angular_scale,
        control_freq=args.control_freq,
        record_data=args.record,
        fixed_cam=fixed_cam,
        wrist_cam=wrist_cam,
    )
    teleop.start()
    
    # Track saved episodes
    saved_episodes = 0
    prev_recording_state = False  # Track recording state changes
    
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
                # Recording was stopped via L3, save the episode
                episode = teleop._current_episode
                if episode is not None and len(episode.ee_pose) > 10:  # Min 10 frames
                    episode.success = True
                    save_episode(episode, out_dir)
                    saved_episodes += 1
                    print(f"[Main] ✓ Episode {saved_episodes} saved!")
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
                save_episode(episode, out_dir)
                saved_episodes += 1
        
        if not piper._emergency_stop:
            piper.open_gripper()
            time.sleep(0.5)
            piper.go_to_home()
            time.sleep(2.0)
        
        piper.disconnect()
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
        
        # Save metadata if recording
        if args.record and out_dir:
            metadata = {
                "collection_type": "teleop",
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
