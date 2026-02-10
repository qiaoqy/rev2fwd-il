#!/usr/bin/env python3
"""Script 8: Evaluate DiT Flow Policy on Real Piper Arm.

This script evaluates a trained DiT Flow policy (Diffusion Transformer with
Flow Matching) on the real Piper robotic arm.  It is the evaluation counterpart
of script 7 (7_train_ditflow.py), analogous to how script 5 evaluates the
standard Diffusion Policy trained by script 4.

=============================================================================
KEY DESIGN PRINCIPLES
=============================================================================
1. Actions are RELATIVE deltas (not absolute poses)
   - Policy outputs: [delta_x, delta_y, delta_z, delta_qw, delta_qx, delta_qy,
     delta_qz, gripper_target]
   - Delta position: added to current position in meters
   - Delta quaternion: converted to euler degrees, then to radians, added to
     current euler radians
   - Gripper target: [0, 1] mapped to [0, GRIPPER_OPEN_ANGLE] degrees

2. Observation format matches training (script 7):
   - observation.image: (3, H, W) from front camera (Orbbec_Gemini_335L)
   - observation.wrist_image: (3, H, W) from wrist camera (Dabai_DC1)
   - observation.state: [x, y, z, qw, qx, qy, qz] (7D) or + gripper (8D)
   - Image size read from model config, NOT hardcoded

3. Hardware interface matches teleop (script 1):
   - PiperController: dict-based get_ee_pose_meters(), move_to_pose(radians)
   - CameraCapture: string camera name support via /dev/v4l/by-id
   - PS5 controller for human interaction (start/stop/e-stop)
   - Motion mode: MOVE_P (0x00), not MOVE_L
   - Robust arm enable with motor status checking

4. PS5 controller interaction:
   - Square: Start/Stop episode
   - Cross: Go home / Abort episode
   - Triangle: Toggle gripper manually (overrides policy during episode)
   - Share: Emergency stop / Resume
   - Options: Re-enable arm
   - PS Button: Quit

=============================================================================
DITFLOW vs DIFFUSION POLICY
=============================================================================
DiT Flow uses:
- Transformer architecture instead of U-Net (better scaling)
- Flow Matching objective instead of DDPM (faster inference)
- AdaLN-Zero conditioning for time step (more stable training)
- num_inference_steps: flow integration steps (default ~100)

=============================================================================
DEPENDENCIES
=============================================================================
Requires:
  - lerobot >= 0.4.2
  - lerobot_policy_ditflow >= 0.1.0:
        pip install git+https://github.com/danielsanjosepro/lerobot_policy_ditflow.git

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation
python scripts/scripts_piper_local/8_eval_ditflow_piper.py \
    --checkpoint /media/qiyuan/SSDQQY/runs/ditflow_piper_teleop_B_0206/checkpoints/checkpoints/200000/pretrained_model

# Full options
python scripts/scripts_piper_local/8_eval_ditflow_piper.py \
    --checkpoint /media/qiyuan/SSDQQY/runs/ditflow_piper_teleop_B_0206/checkpoints/checkpoints/200000/pretrained_model \
    --can_interface can0 \
    --num_episodes 10 \
    --max_steps 500 \
    --control_freq 20 \
    --device cuda:0 \
    --show_camera

# Override inference steps (fewer = faster but noisier)
python scripts/scripts_piper_local/8_eval_ditflow_piper.py \
    --checkpoint /media/qiyuan/SSDQQY/runs/ditflow_piper_teleop_B_0206/checkpoints/checkpoints/200000/pretrained_model \
    --num_inference_steps 50

# Override action steps
python scripts/scripts_piper_local/8_eval_ditflow_piper.py \
    --checkpoint /media/qiyuan/SSDQQY/runs/ditflow_piper_teleop_B_0206/checkpoints/checkpoints/200000/pretrained_model \
    --n_action_steps 4 --num_inference_steps 100

python scripts/scripts_piper_local/8_eval_ditflow_piper.py \
    --checkpoint /media/qiyuan/SSDQQY/runs/ditflow_piper_0210_A/checkpoints/checkpoints/050000/pretrained_model \
    --n_action_steps 16 --num_inference_steps 100
=============================================================================
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Try to import OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Warning] OpenCV not installed. Camera display disabled.")

# Suppress pygame pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Set SDL environment variables before importing pygame
os.environ.setdefault("SDL_JOYSTICK_HIDAPI_PS5", "1")

# Try to import pygame
try:
    import pygame
    from pygame import joystick
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[Warning] pygame not installed. Falling back to stdin control.")

# Piper SDK
from piper_sdk import C_PiperInterface_V2


# =============================================================================
# Constants (matching script 1 exactly)
# =============================================================================

# === Camera Settings ===
DEFAULT_FRONT_CAMERA = "Orbbec_Gemini_335L"
DEFAULT_WRIST_CAMERA = "Dabai_DC1"

# === Home Position (calibrated from script 0) ===
HOME_POSITION = (0.054, 0.0, 0.175)     # X, Y, Z in meters
HOME_ORIENTATION = (3.14, 1.2, 3.14)    # RX, RY, RZ in radians
START_POSITION = (0.2888, 0.0010, 0.2542)     # X, Y, Z in meters
START_ORIENTATION = (-3.1206, 0.2391, -3.1261)  # RX, RY, RZ in radians

# === Control Parameters ===
DEFAULT_SPEED_PERCENT = 20
MIN_SPEED_PERCENT = 5
MAX_SPEED_PERCENT = 50

# === Gripper Parameters ===
GRIPPER_OPEN_ANGLE = 70.0               # Gripper open angle in degrees
GRIPPER_CLOSE_ANGLE = 0.0               # Gripper closed angle in degrees
GRIPPER_EFFORT = 500                    # Gripper force (0-1000)

# === Control Mode ===
MOVE_MODE = 0x00                        # Motion mode: 0x00=MOVE_P (same as teleop)
ENABLE_TIMEOUT = 5.0                    # Arm enable timeout

# === Workspace Limits (Safety) ===
WORKSPACE_LIMITS = {
    "x_min": -0.3,  "x_max": 0.5,
    "y_min": -0.3,  "y_max": 0.3,
    "z_min": 0.05,  "z_max": 0.50,
}

# === PS5 Controller Button/Axis Mapping ===
class PS5Buttons:
    """PS5 DualSense button indices for pygame (Linux SDL mapping)."""
    CROSS = 0
    CIRCLE = 1
    TRIANGLE = 2
    SQUARE = 3
    L1 = 4
    R1 = 5
    L2_BTN = 6
    R2_BTN = 7
    SHARE = 8
    OPTIONS = 9
    PS = 10
    L3 = 11
    R3 = 12


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
    return tuple(rot.as_euler('xyz', degrees=True))  # [rx, ry, rz] in degrees


def is_camera_enabled(camera_id: Union[int, str]) -> bool:
    """Check if camera is enabled (not -1 or "-1")."""
    if isinstance(camera_id, int):
        return camera_id >= 0
    if isinstance(camera_id, str):
        return camera_id.strip() != "-1"
    return False


# =============================================================================
# Camera Capture (matching script 1)
# =============================================================================

class CameraCapture:
    """Camera capture with string name support via /dev/v4l/by-id lookup."""

    def __init__(self, camera_id: Union[int, str], width: int = 640, height: int = 480, name: str = ""):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.name = name or str(camera_id)
        self.cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_timestamp = 0.0
        self._consecutive_failures = 0
        self._running = False
        self._thread = None

    def _resolve_camera_source(self, camera_id: Union[int, str]):
        """Resolve camera identifier (supports string names via /dev/v4l/by-id)."""
        if isinstance(camera_id, int):
            return camera_id

        if isinstance(camera_id, str):
            try:
                return int(camera_id)
            except ValueError:
                pass

            by_id_dir = Path("/dev/v4l/by-id")
            if by_id_dir.exists():
                for entry in sorted(by_id_dir.iterdir()):
                    if camera_id in entry.name and "video-index0" in entry.name:
                        resolved = entry.resolve()
                        print(f"[Camera] Resolved '{camera_id}' -> {resolved}")
                        return str(resolved)

            print(f"[Camera] Warning: Could not resolve '{camera_id}', trying as device path")
            return camera_id

        return camera_id

    def start(self) -> bool:
        """Start camera capture."""
        source = self._resolve_camera_source(self.camera_id)
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            print(f"[Camera {self.name}] Failed to open: {source}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        for _ in range(50):
            if self._latest_frame is not None:
                break
            time.sleep(0.1)

        if self._latest_frame is None:
            print(f"[Camera {self.name}] Warning: No frames after 5s")
            return False

        print(f"[Camera {self.name}] Started ({self.width}x{self.height})")
        return True

    def _capture_loop(self):
        """Background capture loop."""
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._frame_timestamp = time.time()
                    self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
            time.sleep(0.001)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (RGB format)."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """Get latest frame in RGB format (alias)."""
        return self.get_frame()

    def get_frame_age(self) -> float:
        """Get age of latest frame in seconds."""
        with self._lock:
            if self._frame_timestamp > 0:
                return time.time() - self._frame_timestamp
        return float('inf')

    def is_healthy(self, max_age: float = 1.0) -> bool:
        """Check if camera is producing fresh frames."""
        return self.get_frame_age() < max_age

    def stop(self):
        """Stop camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print(f"[Camera {self.name}] Stopped")


# =============================================================================
# Piper Arm Controller (matching script 1 interface)
# =============================================================================

class PiperController:
    """High-level controller for Piper robotic arm."""

    def __init__(self, can_interface: str = "can0"):
        self.can_interface = can_interface
        self.piper: Optional[Any] = None
        self._emergency_stop = False
        self._home_position = HOME_POSITION
        self._home_orientation = HOME_ORIENTATION
        self._start_position = START_POSITION
        self._start_orientation = START_ORIENTATION
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
            print("[Piper] Connected!")
            return True
        except Exception as e:
            print(f"[Piper] Connection failed: {e}")
            print(f"[Piper] Hint: Make sure CAN interface is up. Run:")
            print(f"  sudo ip link set {self.can_interface} up type can bitrate 1000000")
            return False

    def enable(self) -> bool:
        """Enable the arm with robust checking."""
        if not self.connected or self.piper is None:
            print("[Piper] Not connected!")
            return False

        try:
            print("[Piper] Enabling arm...")
            for attempt in range(3):
                self.piper.EnableArm(7)
                time.sleep(0.1)

            self.piper.MotionCtrl_2(0x01, MOVE_MODE, self._speed_percent, 0)
            self.piper.GripperCtrl(0x01, 1000, 0x01, 0)

            start_time = time.time()
            while time.time() - start_time < ENABLE_TIMEOUT:
                arm_ok, gripper_ok = self._check_motor_enable_status()
                if arm_ok and gripper_ok:
                    self.enabled = True
                    print("[Piper] Arm enabled successfully!")
                    return True
                elif arm_ok:
                    self.piper.GripperCtrl(0x01, 1000, 0x01, 0)
                else:
                    self.piper.EnableArm(7)
                time.sleep(0.1)

            print("[Piper] Enable timeout!")
            self._print_enable_status()
            return False

        except Exception as e:
            print(f"[Piper] Enable error: {e}")
            return False

    def _check_motor_enable_status(self) -> Tuple[bool, bool]:
        """Check if arm motors and gripper are enabled."""
        try:
            arm_msgs = self.piper.GetArmLowSpdInfoMsgs()
            arm_enabled = (
                arm_msgs.motor_1.foc_status.driver_enable_status and
                arm_msgs.motor_2.foc_status.driver_enable_status and
                arm_msgs.motor_3.foc_status.driver_enable_status and
                arm_msgs.motor_4.foc_status.driver_enable_status and
                arm_msgs.motor_5.foc_status.driver_enable_status and
                arm_msgs.motor_6.foc_status.driver_enable_status
            )
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_enabled = gripper_msgs.gripper_state.foc_status.driver_enable_status
            return arm_enabled, gripper_enabled
        except Exception:
            return False, False

    def _print_enable_status(self):
        """Print detailed enable status for debugging."""
        try:
            arm_msgs = self.piper.GetArmLowSpdInfoMsgs()
            print("[Debug] Motor enable status:")
            for i in range(1, 7):
                motor = getattr(arm_msgs, f"motor_{i}")
                status = motor.foc_status.driver_enable_status
                print(f"  Motor {i}: {'✓' if status else '✗'}")
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_status = gripper_msgs.gripper_state.foc_status.driver_enable_status
            print(f"  Gripper: {'✓' if gripper_status else '✗'}")
        except Exception as e:
            print(f"[Debug] Failed to read enable status: {e}")

    def disconnect(self):
        """Disable and disconnect from the arm."""
        if self.piper is not None and self.enabled:
            try:
                self.piper.DisableArm(7)
                self.enabled = False
            except Exception as e:
                print(f"[Piper] Disable error: {e}")
        self.connected = False
        self.piper = None
        print("[Piper] Disconnected")

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
            except Exception:
                pass
        print("[Piper] ✓ Emergency stop cleared, motion resumed")

    def set_speed(self, speed_percent: int):
        """Set motion speed percentage."""
        self._speed_percent = max(MIN_SPEED_PERCENT, min(MAX_SPEED_PERCENT, speed_percent))
        if self.piper is not None and self.enabled:
            try:
                self.piper.MotionCtrl_2(0x01, MOVE_MODE, self._speed_percent, 0)
            except Exception:
                pass
        print(f"[Piper] Speed set to {self._speed_percent}%")

    def get_ee_pose_meters(self) -> Optional[Dict[str, float]]:
        """Get current end-effector pose in meters and radians."""
        if not self.connected or self.piper is None:
            return None
        try:
            msg = self.piper.GetArmEndPoseMsgs()
            pose = msg.end_pose
            x = pose.X_axis / 1_000_000.0
            y = pose.Y_axis / 1_000_000.0
            z = pose.Z_axis / 1_000_000.0
            rx_deg = pose.RX_axis / 1000.0
            ry_deg = pose.RY_axis / 1000.0
            rz_deg = pose.RZ_axis / 1000.0
            return {
                'x': x, 'y': y, 'z': z,
                'rx': np.deg2rad(rx_deg),
                'ry': np.deg2rad(ry_deg),
                'rz': np.deg2rad(rz_deg),
                'rx_deg': rx_deg,
                'ry_deg': ry_deg,
                'rz_deg': rz_deg,
            }
        except Exception as e:
            print(f"[Piper] Error reading pose: {e}")
            return None

    def move_to_pose(self, x: float, y: float, z: float,
                     rx: float, ry: float, rz: float) -> bool:
        """Move to specified pose (meters and radians)."""
        if not self.enabled or self.piper is None:
            return False
        if self._emergency_stop:
            return False
        x = max(WORKSPACE_LIMITS["x_min"], min(WORKSPACE_LIMITS["x_max"], x))
        y = max(WORKSPACE_LIMITS["y_min"], min(WORKSPACE_LIMITS["y_max"], y))
        z = max(WORKSPACE_LIMITS["z_min"], min(WORKSPACE_LIMITS["z_max"], z))
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
            angle_deg = max(0.0, min(GRIPPER_OPEN_ANGLE, angle_deg))
            angle_sdk = int(angle_deg * 1000)
            self.piper.GripperCtrl(angle_sdk, effort, 0x01, 0)
            return True
        except Exception as e:
            print(f"[Piper] Gripper control failed: {e}")
            return False

    def open_gripper(self) -> bool:
        return self.set_gripper_angle(GRIPPER_OPEN_ANGLE)

    def close_gripper(self) -> bool:
        return self.set_gripper_angle(GRIPPER_CLOSE_ANGLE)

    def get_gripper_angle(self) -> Optional[float]:
        """Get current gripper angle in degrees."""
        if not self.connected or self.piper is None:
            return None
        try:
            gripper_msgs = self.piper.GetArmGripperMsgs()
            return gripper_msgs.gripper_state.grippers_angle / 1000.0
        except Exception:
            return None

    def go_to_home(self) -> bool:
        if self._emergency_stop:
            return False
        print("[Piper] Going to home position...")
        x, y, z = self._home_position
        rx, ry, rz = self._home_orientation
        return self.move_to_pose(x, y, z, rx, ry, rz)

    def go_to_start(self) -> bool:
        if self._emergency_stop:
            return False
        print("[Piper] Going to start position...")
        x, y, z = self._start_position
        rx, ry, rz = self._start_orientation
        return self.move_to_pose(x, y, z, rx, ry, rz)

    def is_arm_ready(self) -> bool:
        if not self.enabled or self.piper is None:
            return False
        try:
            arm_ok, gripper_ok = self._check_motor_enable_status()
            if not (arm_ok and gripper_ok):
                return False
            status = self.piper.GetArmStatus()
            ctrl_mode = getattr(status.arm_status, 'ctrl_mode', 0)
            return int(ctrl_mode) != 0
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_arm_ready():
                return True
            time.sleep(0.1)
        print("[Piper] Arm not ready after timeout")
        self._print_enable_status()
        return False


# =============================================================================
# PS5 Controller Handler (matching script 1)
# =============================================================================

class PS5Controller:
    """Handler for PS5 DualSense controller using pygame."""

    def __init__(self):
        self.joystick: Optional[Any] = None
        self.connected = False
        self._prev_buttons = {}

    def initialize(self) -> bool:
        if not PYGAME_AVAILABLE:
            return False
        pygame.init()
        pygame.joystick.init()
        num_joysticks = pygame.joystick.get_count()
        if num_joysticks == 0:
            print("[PS5] No controllers found")
            return False
        for i in range(num_joysticks):
            js = pygame.joystick.Joystick(i)
            js.init()
            name = js.get_name().lower()
            if "dualsense" in name or "ps5" in name or "054c:0ce6" in name:
                self.joystick = js
                self.connected = True
                print(f"[PS5] Found: {js.get_name()}")
                return True
        if num_joysticks > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.connected = True
            print(f"[PS5] Using first gamepad: {self.joystick.get_name()}")
            return True
        return False

    def update(self):
        pygame.event.pump()

    def get_button(self, button: int) -> bool:
        if not self.connected or self.joystick is None:
            return False
        try:
            return self.joystick.get_button(button)
        except Exception:
            return False

    def get_button_pressed(self, button: int) -> bool:
        current = self.get_button(button)
        prev = self._prev_buttons.get(button, False)
        self._prev_buttons[button] = current
        return current and not prev

    def rumble(self, low_frequency: float = 0.5, high_frequency: float = 0.5,
               duration_ms: int = 200):
        if not self.connected or self.joystick is None:
            return
        try:
            self.joystick.rumble(low_frequency, high_frequency, duration_ms)
        except Exception:
            pass

    def rumble_short(self, count: int = 1, intensity: float = 0.7, duration_ms: int = 100):
        def _do_rumble():
            for i in range(count):
                self.rumble(intensity, intensity, duration_ms)
                time.sleep(duration_ms / 1000.0 + 0.05)
        threading.Thread(target=_do_rumble, daemon=True).start()

    def close(self):
        if PYGAME_AVAILABLE:
            pygame.quit()


class StdinController:
    """Fallback controller using stdin."""

    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()
        print("[Input] Stdin controller active. Commands: s=start/stop, h=home, g=gripper, e=estop, q=quit")

    def _input_loop(self):
        while True:
            try:
                line = input().strip().lower()
                with self._lock:
                    self._pressed.add(line)
            except (EOFError, KeyboardInterrupt):
                break

    def poll(self) -> dict:
        with self._lock:
            pressed = self._pressed.copy()
            self._pressed.clear()
        return {
            'start_stop': 's' in pressed,
            'home': 'h' in pressed,
            'gripper': 'g' in pressed,
            'emergency_stop': 'e' in pressed,
            'quit': 'q' in pressed,
        }


# =============================================================================
# Video Recording
# =============================================================================

class VideoRecorder:
    """Record video from camera frames.

    The recorder uses the *actual* camera frame dimensions so that the
    saved video faithfully represents what the cameras see.  When two
    camera views are combined side-by-side, each view keeps its native
    aspect ratio (both are resized to the same height before
    concatenation).
    """

    def __init__(self, output_path: str, fps: int = 20, width: int = 640, height: int = 480):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame (RGB format) at its native resolution."""
        self.frames.append(frame.copy())

    def add_combined_frame(self, fixed_frame: np.ndarray, wrist_frame: Optional[np.ndarray]):
        """Add a combined frame with both camera views side-by-side.

        Each view is resized to a common height (the taller of the two)
        while preserving its aspect ratio, then concatenated horizontally.
        """
        if wrist_frame is not None:
            # Use the taller frame's height as the target
            target_h = max(fixed_frame.shape[0], wrist_frame.shape[0])
            # Resize each frame to target_h keeping aspect ratio
            fixed_scale = target_h / fixed_frame.shape[0]
            fixed_w = int(fixed_frame.shape[1] * fixed_scale)
            wrist_scale = target_h / wrist_frame.shape[0]
            wrist_w = int(wrist_frame.shape[1] * wrist_scale)
            fixed_resized = cv2.resize(fixed_frame, (fixed_w, target_h))
            wrist_resized = cv2.resize(wrist_frame, (wrist_w, target_h))
            combined = np.concatenate([fixed_resized, wrist_resized], axis=1)
        else:
            combined = fixed_frame.copy()
        self.frames.append(combined)

    def save(self):
        if not self.frames:
            print("[Video] No frames to save!")
            return
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            self.output_path, fps=self.fps, codec='libx264',
            quality=8, pixelformat='yuv420p',
        )
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        print(f"[Video] Saved {len(self.frames)} frames to {self.output_path}")

    def clear(self):
        self.frames = []


# =============================================================================
# Trajectory Comparison Plot
# =============================================================================

def generate_trajectory_plot(
    actual_positions: List[np.ndarray],
    integrated_positions: List[np.ndarray],
    episode_id: int,
    width: int = 960,
    height: int = 720,
    dpi: int = 100,
) -> np.ndarray:
    """Generate a comparison plot of actual EE trajectory vs action-integrated trajectory."""
    actual = np.array(actual_positions)
    integrated = np.array(integrated_positions)
    steps = np.arange(len(actual))

    labels = ['X (m)', 'Y (m)', 'Z (m)']
    colors_actual = ['#1f77b4', '#2ca02c', '#9467bd']
    colors_cmd = ['#ff7f0e', '#d62728', '#e377c2']

    fig, axes = plt.subplots(3, 1, figsize=(width / dpi, height / dpi), dpi=dpi,
                             sharex=True)
    fig.suptitle(f'Episode {episode_id}: Commanded (integral) vs Actual Trajectory',
                 fontsize=13, fontweight='bold')

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(steps, actual[:, i], color=colors_actual[i], linewidth=1.5,
                label='Actual EE pos')
        ax.plot(steps, integrated[:, i], color=colors_cmd[i], linewidth=1.5,
                linestyle='--', label='Action integral (commanded)')
        ax.fill_between(steps, actual[:, i], integrated[:, i],
                        alpha=0.15, color=colors_cmd[i])
        ax.set_ylabel(label, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Step', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)

    return img


# =============================================================================
# DiT Flow Policy Loading
# =============================================================================

def load_policy_config(pretrained_dir: str) -> Dict[str, Any]:
    """Load policy configuration from checkpoint.

    Returns dict with: has_wrist, image_shape, state_dim, action_dim,
    n_obs_steps, n_action_steps, policy_type, raw_config
    """
    config_path = Path(pretrained_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    input_features = config_dict.get("input_features", {})
    output_features = config_dict.get("output_features", {})

    has_wrist = "observation.wrist_image" in input_features

    image_shape = None
    if "observation.image" in input_features:
        image_shape = tuple(input_features["observation.image"]["shape"])

    state_dim = None
    if "observation.state" in input_features:
        state_dim = input_features["observation.state"]["shape"][0]

    action_dim = None
    if "action" in output_features:
        action_dim = output_features["action"]["shape"][0]

    policy_type = config_dict.get("type", "unknown")

    return {
        "has_wrist": has_wrist,
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_obs_steps": config_dict.get("n_obs_steps", 2),
        "n_action_steps": config_dict.get("n_action_steps", 8),
        "policy_type": policy_type,
        "raw_config": config_dict,
    }


def load_ditflow_policy(
    pretrained_dir: str,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
) -> Tuple[Any, Any, Any, int, int]:
    """Load LeRobot DiT Flow policy from checkpoint.

    This imports lerobot_policy_ditflow which registers the 'ditflow' policy
    type via @PreTrainedConfig.register_subclass("ditflow").

    Returns:
        (policy, preprocessor, postprocessor, num_inference_steps, n_action_steps)
    """
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.factory import make_pre_post_processors

    # Import lerobot_policy_ditflow to register the ditflow policy type
    from lerobot_policy_ditflow import DiTFlowConfig, DiTFlowPolicy

    pretrained_path = Path(pretrained_dir)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"

    print(f"[Policy] Loading DiT Flow config from {config_path}...", flush=True)
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Remove 'type' field if present (not a constructor arg)
    if "type" in config_dict:
        del config_dict["type"]

    # Parse input_features
    input_features_raw = config_dict.get("input_features", {})
    input_features = {}
    for key, val in input_features_raw.items():
        feat_type = FeatureType[val["type"]] if isinstance(val["type"], str) else val["type"]
        input_features[key] = PolicyFeature(type=feat_type, shape=tuple(val["shape"]))
    config_dict["input_features"] = input_features

    # Parse output_features
    output_features_raw = config_dict.get("output_features", {})
    output_features = {}
    for key, val in output_features_raw.items():
        feat_type = FeatureType[val["type"]] if isinstance(val["type"], str) else val["type"]
        output_features[key] = PolicyFeature(type=feat_type, shape=tuple(val["shape"]))
    config_dict["output_features"] = output_features

    # Parse normalization_mapping
    if "normalization_mapping" in config_dict:
        norm_mapping_raw = config_dict["normalization_mapping"]
        norm_mapping = {}
        for key, val in norm_mapping_raw.items():
            norm_mode = NormalizationMode[val] if isinstance(val, str) else val
            norm_mapping[key] = norm_mode
        config_dict["normalization_mapping"] = norm_mapping

    # Convert lists to tuples for fields that expect tuples
    for field_name in ["crop_shape", "optimizer_betas"]:
        if field_name in config_dict and isinstance(config_dict[field_name], list):
            config_dict[field_name] = tuple(config_dict[field_name])

    # Override parameters if specified
    if num_inference_steps is not None:
        config_dict["num_inference_steps"] = num_inference_steps
        print(f"[Policy] Overriding num_inference_steps = {num_inference_steps}")

    if n_action_steps is not None:
        horizon = config_dict.get("horizon", 16)
        if n_action_steps > horizon:
            n_action_steps = horizon
        config_dict["n_action_steps"] = n_action_steps
        print(f"[Policy] Overriding n_action_steps = {n_action_steps}")

    # Filter out unknown fields not accepted by DiTFlowConfig
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(DiTFlowConfig)}
    unknown_keys = set(config_dict.keys()) - valid_fields
    if unknown_keys:
        print(f"[Policy] Ignoring unknown config keys: {unknown_keys}", flush=True)
        for k in unknown_keys:
            del config_dict[k]

    # Create config and policy
    print(f"[Policy] Creating DiTFlowConfig...", flush=True)
    cfg = DiTFlowConfig(**config_dict)

    print(f"[Policy] Creating DiTFlowPolicy model...", flush=True)
    policy = DiTFlowPolicy(cfg)

    # Load weights
    print(f"[Policy] Loading model weights...", flush=True)
    state_dict = load_file(model_path)

    # Use strict=False to handle version mismatches in lerobot_policy_ditflow.
    # Newer versions wrap linear1/linear2 inside an nn.Sequential called `mlp`,
    # creating alias keys (mlp.0 == linear1, mlp.3 == linear2).  Checkpoints
    # trained with the older version only have linear1/linear2 keys.  Because
    # the underlying nn.Linear objects are shared, loading linear1/linear2
    # automatically populates mlp.0/mlp.3 — so missing mlp.* keys are safe.
    result = policy.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        # Only allow missing keys that are mlp.N aliases of linear1/linear2
        unexpected_missing = [
            k for k in result.missing_keys
            if not ((".mlp.0." in k or ".mlp.3." in k) and "decoder.layers" in k)
        ]
        if unexpected_missing:
            raise RuntimeError(
                f"[Policy] Unexpected missing keys in state_dict:\n"
                f"  {unexpected_missing}\n"
                f"This usually means lerobot_policy_ditflow version mismatch."
            )
        print(f"[Policy] Note: {len(result.missing_keys)} aliased mlp keys auto-populated "
              f"via shared linear1/linear2 parameters (safe).", flush=True)
    if result.unexpected_keys:
        print(f"[Policy] Warning: unexpected keys in checkpoint: {result.unexpected_keys}",
              flush=True)

    # Move to device
    print(f"[Policy] Moving to device {device}...", flush=True)
    policy = policy.to(device)
    policy.eval()

    # Load preprocessor and postprocessor
    print(f"[Policy] Loading preprocessor/postprocessor...", flush=True)
    preprocessor_overrides = {"device_processor": {"device": device}}
    postprocessor_overrides = {"device_processor": {"device": device}}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    actual_inference_steps = cfg.num_inference_steps or 100
    actual_n_action_steps = cfg.n_action_steps

    print(f"[Policy] DiT Flow loaded! (inference_steps={actual_inference_steps}, "
          f"n_action_steps={actual_n_action_steps})", flush=True)

    return policy, preprocessor, postprocessor, actual_inference_steps, actual_n_action_steps


# =============================================================================
# Observation Building (matching training format exactly)
# =============================================================================

def build_observation(
    fixed_image: np.ndarray,
    wrist_image: Optional[np.ndarray],
    pose: Dict[str, float],
    policy_config: Dict,
) -> Dict[str, torch.Tensor]:
    """Build observation dict matching the training data format.

    Returns torch tensors matching the format that LeRobot's dataset __getitem__
    produces.  Images are float32 (1, C, H, W) in [0, 1].  State is float32
    (1, state_dim).  All tensors include a leading batch dimension.
    """
    image_shape = policy_config.get("image_shape", (3, 240, 320))
    target_h, target_w = image_shape[1], image_shape[2]

    obs = {}

    # Fixed camera image
    fixed_resized = cv2.resize(fixed_image, (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)
    fixed_tensor = torch.from_numpy(fixed_resized).permute(2, 0, 1).float() / 255.0
    obs["observation.image"] = fixed_tensor.unsqueeze(0)

    # Wrist camera image
    if policy_config.get("has_wrist", False) and wrist_image is not None:
        wrist_resized = cv2.resize(wrist_image, (target_w, target_h),
                                   interpolation=cv2.INTER_LINEAR)
        wrist_tensor = torch.from_numpy(wrist_resized).permute(2, 0, 1).float() / 255.0
        obs["observation.wrist_image"] = wrist_tensor.unsqueeze(0)

    # State vector
    current_quat = np.array(euler_to_quat(pose['rx_deg'], pose['ry_deg'], pose['rz_deg']))
    ee_pose = np.array([pose['x'], pose['y'], pose['z'],
                        current_quat[0], current_quat[1], current_quat[2], current_quat[3]],
                       dtype=np.float32)

    state_dim = policy_config.get("state_dim", 8)
    if state_dim == 8:
        gripper_angle = pose.get('gripper_angle', GRIPPER_OPEN_ANGLE)
        gripper_normalized = gripper_angle / GRIPPER_OPEN_ANGLE
        state = np.concatenate([ee_pose, [gripper_normalized]]).astype(np.float32)
    else:
        state = ee_pose

    obs["observation.state"] = torch.from_numpy(state).float().unsqueeze(0)

    return obs


# =============================================================================
# Action Application (RELATIVE delta)
# =============================================================================

def apply_delta_action(
    pose: Dict[str, float],
    action: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    """Apply a RELATIVE delta action to the current pose.

    Args:
        pose: Current EE pose dict from PiperController.get_ee_pose_meters()
        action: 8D action array [delta_x, delta_y, delta_z,
                delta_qw, delta_qx, delta_qy, delta_qz, gripper_target]

    Returns:
        (target_x, target_y, target_z, target_rx, target_ry, target_rz, gripper_deg)
        where position is in meters, rotation in radians, gripper in degrees [0, 70]
    """
    delta_x, delta_y, delta_z = action[0], action[1], action[2]
    delta_qw, delta_qx, delta_qy, delta_qz = action[3], action[4], action[5], action[6]
    gripper_target = action[7]

    # Apply position delta (meters)
    target_x = pose['x'] + delta_x
    target_y = pose['y'] + delta_y
    target_z = pose['z'] + delta_z

    # Apply rotation delta
    delta_euler_deg = quat_to_euler(delta_qw, delta_qx, delta_qy, delta_qz)
    delta_rx_rad = np.deg2rad(delta_euler_deg[0])
    delta_ry_rad = np.deg2rad(delta_euler_deg[1])
    delta_rz_rad = np.deg2rad(delta_euler_deg[2])

    target_rx = pose['rx'] + delta_rx_rad
    target_ry = pose['ry'] + delta_ry_rad
    target_rz = pose['rz'] + delta_rz_rad

    # Gripper: normalized [0, 1] -> degrees [0, GRIPPER_OPEN_ANGLE]
    gripper_deg = float(np.clip(gripper_target, 0.0, 1.0)) * GRIPPER_OPEN_ANGLE

    return target_x, target_y, target_z, target_rx, target_ry, target_rz, gripper_deg


# =============================================================================
# Evaluation Results
# =============================================================================

@dataclass
class EpisodeResult:
    episode_id: int
    steps: int
    duration: float
    emergency_stopped: bool = False
    notes: str = ""


@dataclass
class EvaluationResults:
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def avg_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(ep.steps for ep in self.episodes) / len(self.episodes)

    @property
    def avg_duration(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(ep.duration for ep in self.episodes) / len(self.episodes)

    def to_dict(self) -> Dict:
        return {
            "num_episodes": self.num_episodes,
            "avg_steps": self.avg_steps,
            "avg_duration": self.avg_duration,
            "episodes": [
                {
                    "id": ep.episode_id,
                    "steps": ep.steps,
                    "duration": ep.duration,
                    "emergency_stopped": ep.emergency_stopped,
                    "notes": ep.notes,
                }
                for ep in self.episodes
            ],
        }

    def save(self, output_path: str):
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Results] Saved to {output_path}")


# =============================================================================
# Episode Runner
# =============================================================================

def run_episode(
    piper: PiperController,
    fixed_cam: CameraCapture,
    wrist_cam: Optional[CameraCapture],
    policy,
    preprocessor,
    postprocessor,
    policy_config: Dict,
    episode_id: int,
    max_steps: int,
    n_action_steps: int,
    control_freq: int,
    device: str,
    video_recorder: Optional[VideoRecorder] = None,
    ps5: Optional["PS5Controller"] = None,
    stdin_ctrl: Optional["StdinController"] = None,
    show_camera: bool = False,
) -> EpisodeResult:
    """Run a single evaluation episode.

    The episode runs until max_steps or until the user presses Square to stop.
    Actions are RELATIVE deltas applied to the current pose.

    PS5 controls during episode:
        Square:   Stop episode (finish normally)
        Share:    Emergency stop / Resume
        Cross:    Abort episode and go home
        Triangle: Toggle gripper manually (overrides policy gripper)
    """
    print(f"\n{'='*60}")
    print(f"[Episode {episode_id}] Starting...")
    print(f"{'='*60}")

    # Reset policy state (clear action queue)
    policy.reset()

    control_period = 1.0 / control_freq
    step = 0
    episode_start_time = time.time()
    user_stopped = False
    gripper_manual_override = None

    # Trajectory tracking for comparison plot
    actual_positions = []
    integrated_positions = []
    integrated_xyz = None

    log_freq = max(1, control_freq)

    print(f"[Episode {episode_id}] Running inference loop (max {max_steps} steps, {control_freq}Hz)...")
    if ps5:
        print(f"  Square=Stop | Share=E-Stop | Cross=Abort | Triangle=Toggle Gripper")

    while step < max_steps:
        loop_start = time.time()

        # =================================================================
        # Poll controller events
        # =================================================================
        if ps5:
            ps5.update()

            if ps5.get_button_pressed(PS5Buttons.SQUARE):
                print(f"[Episode {episode_id}] User stopped at step {step}")
                ps5.rumble_short(2)
                user_stopped = True
                break

            if ps5.get_button_pressed(PS5Buttons.SHARE):
                if piper._emergency_stop:
                    piper.clear_emergency_stop()
                    ps5.rumble_short(2)
                else:
                    piper.emergency_stop()
                    ps5.rumble_short(3, intensity=1.0)

            if ps5.get_button_pressed(PS5Buttons.CROSS):
                print(f"[Episode {episode_id}] Aborted at step {step}, going home...")
                ps5.rumble_short(1)
                piper.go_to_home()
                return EpisodeResult(
                    episode_id=episode_id,
                    steps=step,
                    duration=time.time() - episode_start_time,
                    notes="aborted_by_user",
                )

            if ps5.get_button_pressed(PS5Buttons.TRIANGLE):
                if gripper_manual_override is None:
                    current_gripper = piper.get_gripper_angle() or 0.0
                    if current_gripper > GRIPPER_OPEN_ANGLE / 2:
                        gripper_manual_override = GRIPPER_CLOSE_ANGLE
                        piper.close_gripper()
                    else:
                        gripper_manual_override = GRIPPER_OPEN_ANGLE
                        piper.open_gripper()
                    print(f"  [Gripper] Manual override: {gripper_manual_override:.0f}deg")
                else:
                    if gripper_manual_override > GRIPPER_OPEN_ANGLE / 2:
                        gripper_manual_override = GRIPPER_CLOSE_ANGLE
                        piper.close_gripper()
                    else:
                        gripper_manual_override = GRIPPER_OPEN_ANGLE
                        piper.open_gripper()
                    print(f"  [Gripper] Manual toggle: {gripper_manual_override:.0f}deg")
                ps5.rumble_short(1)

            if ps5.get_button_pressed(PS5Buttons.OPTIONS):
                print("[Episode] Re-enabling arm...")
                piper.enable()
                ps5.rumble_short(1)

        elif stdin_ctrl:
            events = stdin_ctrl.poll()
            if events.get('start_stop', False):
                print(f"[Episode {episode_id}] User stopped at step {step}")
                user_stopped = True
                break
            if events.get('emergency_stop', False):
                if piper._emergency_stop:
                    piper.clear_emergency_stop()
                else:
                    piper.emergency_stop()
            if events.get('home', False):
                piper.go_to_home()
                return EpisodeResult(
                    episode_id=episode_id,
                    steps=step,
                    duration=time.time() - episode_start_time,
                    notes="aborted_by_user",
                )
            if events.get('gripper', False):
                current_gripper = piper.get_gripper_angle() or 0.0
                if current_gripper > GRIPPER_OPEN_ANGLE / 2:
                    piper.close_gripper()
                    gripper_manual_override = GRIPPER_CLOSE_ANGLE
                else:
                    piper.open_gripper()
                    gripper_manual_override = GRIPPER_OPEN_ANGLE

        # =================================================================
        # Check emergency stop
        # =================================================================
        if piper._emergency_stop:
            time.sleep(0.1)
            continue

        # =================================================================
        # Get current observations
        # =================================================================
        fixed_img = fixed_cam.get_frame()
        wrist_img = wrist_cam.get_frame() if wrist_cam else None
        pose = piper.get_ee_pose_meters()

        if fixed_img is None or pose is None:
            time.sleep(control_period)
            continue

        gripper_angle = piper.get_gripper_angle()
        if gripper_angle is None:
            gripper_angle = GRIPPER_OPEN_ANGLE
        pose['gripper_angle'] = gripper_angle

        if video_recorder is not None:
            video_recorder.add_combined_frame(fixed_img, wrist_img)

        # =================================================================
        # Run policy inference
        # =================================================================
        obs = build_observation(
            fixed_image=fixed_img,
            wrist_image=wrist_img,
            pose=pose,
            policy_config=policy_config,
        )

        obs_normalized = preprocessor(obs)

        with torch.no_grad():
            action_tensor = policy.select_action(obs_normalized)

        action_tensor = postprocessor(action_tensor)

        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy()
        else:
            action = np.asarray(action_tensor)

        if action.ndim == 2:
            action = action[0]

        if step % log_freq == 0:
            print(f"  Step {step}: action={action[:3].round(4)}, "
                  f"pos=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f})", flush=True)

        # =================================================================
        # Track trajectory for comparison plot
        # =================================================================
        current_xyz = np.array([pose['x'], pose['y'], pose['z']])
        actual_positions.append(current_xyz.copy())

        if integrated_xyz is None:
            integrated_xyz = current_xyz.copy()

        integrated_xyz = integrated_xyz + np.array([action[0], action[1], action[2]])
        integrated_positions.append(integrated_xyz.copy())

        # =================================================================
        # Apply action to robot
        # =================================================================
        target_x, target_y, target_z, target_rx, target_ry, target_rz, gripper_deg = \
            apply_delta_action(pose, action)

        piper.move_to_pose(target_x, target_y, target_z, target_rx, target_ry, target_rz)

        if gripper_manual_override is not None:
            piper.set_gripper_angle(gripper_manual_override)
        else:
            piper.set_gripper_angle(gripper_deg)

        # =================================================================
        # Camera display (optional)
        # =================================================================
        if show_camera and CV2_AVAILABLE:
            display_frame = fixed_img.copy()
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

            cv2.putText(display_frame, f"EP {episode_id} | Step {step}/{max_steps} [DiTFlow]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame,
                        f"Pos: ({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f})m",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame,
                        f"Rot: ({pose['rx_deg']:.1f}, {pose['ry_deg']:.1f}, {pose['rz_deg']:.1f})deg",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            delta_pos = np.linalg.norm(action[:3])
            cv2.putText(display_frame,
                        f"Delta: pos={delta_pos:.4f}m  grip={action[7]:.2f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            grip_text = f"Gripper: {gripper_angle:.0f}deg"
            if gripper_manual_override is not None:
                grip_text += " [MANUAL]"
            cv2.putText(display_frame, grip_text,
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            if piper._emergency_stop:
                cv2.putText(display_frame, "E-STOP",
                            (display_frame.shape[1] // 2 - 50, display_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if wrist_img is not None:
                wrist_display = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                h = display_frame.shape[0]
                wrist_resized = cv2.resize(wrist_display,
                                           (int(wrist_display.shape[1] * h / wrist_display.shape[0]), h))
                display_frame = np.concatenate([display_frame, wrist_resized], axis=1)

            cv2.imshow("Piper DiTFlow Eval", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                user_stopped = True
                break

        step += 1

        if step % (log_freq * 5) == 0:
            elapsed = time.time() - episode_start_time
            fps = step / elapsed if elapsed > 0 else 0
            print(f"  Step {step}/{max_steps} | {elapsed:.1f}s | {fps:.1f}Hz | "
                  f"pos=({pose['x']:.3f},{pose['y']:.3f},{pose['z']:.3f})", flush=True)

        elapsed = time.time() - loop_start
        sleep_time = control_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    duration = time.time() - episode_start_time
    status = "stopped by user" if user_stopped else "completed"
    print(f"[Episode {episode_id}] {status}: {step} steps in {duration:.1f}s ({step/duration:.1f}Hz)")

    # =================================================================
    # Generate trajectory comparison plot and attach to video
    # =================================================================
    if len(actual_positions) > 1 and video_recorder is not None:
        try:
            plot_img = generate_trajectory_plot(
                actual_positions=actual_positions,
                integrated_positions=integrated_positions,
                episode_id=episode_id,
                width=max(960, video_recorder.width),
                height=max(720, video_recorder.height),
            )
            plot_resized = cv2.resize(plot_img,
                                      (video_recorder.frames[0].shape[1],
                                       video_recorder.frames[0].shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            plot_hold_frames = video_recorder.fps * 5
            for _ in range(plot_hold_frames):
                video_recorder.frames.append(plot_resized)
            print(f"[Episode {episode_id}] Trajectory plot appended to video ({plot_hold_frames} frames)")

            plot_save_path = Path(video_recorder.output_path).with_suffix('.trajectory.png')
            imageio.imwrite(str(plot_save_path), plot_img)
            print(f"[Episode {episode_id}] Trajectory plot saved to {plot_save_path}")
        except Exception as e:
            print(f"[Episode {episode_id}] Warning: Failed to generate trajectory plot: {e}")

    return EpisodeResult(
        episode_id=episode_id,
        steps=step,
        duration=duration,
        notes="user_stopped" if user_stopped else "",
    )


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DiT Flow policy on Piper arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python 8_eval_ditflow_piper.py \\
      --checkpoint /media/qiyuan/SSDQQY/ditflow_piper_teleop_B_0206/checkpoints/checkpoints/200000/pretrained_model

  # With specific cameras and GPU
  python 8_eval_ditflow_piper.py \\
      --checkpoint /path/to/pretrained_model \\
      --front_cam Orbbec_Gemini_335L --wrist_cam Dabai_DC1 \\
      --device cuda:0

  # Fewer inference steps (faster but noisier)
  python 8_eval_ditflow_piper.py \\
      --checkpoint /path/to/pretrained_model \\
      --num_inference_steps 50

PS5 Controller:
  Square:    Start/Stop episode
  Cross:     Go to home position / Abort episode
  Triangle:  Toggle gripper manually (overrides policy during episode)
  Share:     Emergency stop / Resume
  Options:   Re-enable arm (three-line button, top right)
  PS Button: Quit (PlayStation logo, center)
"""
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model directory")
    parser.add_argument("--can_interface", "-c", type=str, default="can0",
                        help="CAN interface name (default: can0)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode (default: 1000)")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Action steps per inference (None=use model config)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Flow integration steps (None=use model config, typically 100)")
    parser.add_argument("--control_freq", type=int, default=20,
                        help="Control frequency in Hz (default: 20)")
    parser.add_argument("--speed", "-s", type=int, default=DEFAULT_SPEED_PERCENT,
                        help=f"Motion speed percentage (default: {DEFAULT_SPEED_PERCENT})")

    # Camera options
    parser.add_argument("--front_cam", "-f", type=str, default=DEFAULT_FRONT_CAMERA,
                        help=f"Front camera ID/name (default: {DEFAULT_FRONT_CAMERA})")
    parser.add_argument("--wrist_cam", "-w", type=str, default=DEFAULT_WRIST_CAMERA,
                        help=f"Wrist camera ID/name, -1 to disable (default: {DEFAULT_WRIST_CAMERA})")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Camera capture width (default: 320)")
    parser.add_argument("--image_height", type=int, default=270,
                        help="Camera capture height (default: 270)")
    parser.add_argument("--show_camera", action="store_true",
                        help="Show camera feed with overlay during evaluation")

    # Output options
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: auto-generated under checkpoint run dir with date)")
    parser.add_argument("--no_record_video", action="store_true",
                        help="Disable video recording (video is recorded by default)")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


def normalize_camera_value(value: str) -> Union[int, str]:
    if value.strip() == "-1":
        return -1
    try:
        return int(value)
    except ValueError:
        return value


def main():
    args = parse_args()

    args.record_video = not args.no_record_video

    # Auto-derive output directory from checkpoint path if not specified
    if args.out_dir is None:
        checkpoint_path = Path(args.checkpoint).resolve()
        run_dir = checkpoint_path
        while run_dir.name != '' and run_dir.name != run_dir.root:
            if run_dir.name == 'checkpoints' and run_dir.parent.name != 'checkpoints':
                run_dir = run_dir.parent
                break
            run_dir = run_dir.parent
        else:
            run_dir = checkpoint_path.parent

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = run_dir / f"eval_ditflow_{date_str}"
    else:
        out_dir = Path(args.out_dir)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Piper DiT Flow Policy Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Num episodes:  {args.num_episodes}")
    print(f"  Max steps:     {args.max_steps}")
    print(f"  Control freq:  {args.control_freq} Hz")
    print(f"  Device:        {args.device}")
    print(f"  Output dir:    {out_dir}")
    print(f"  Record video:  {args.record_video}")
    print("=" * 60)

    # =========================================================================
    # Load policy config first to determine camera requirements
    # =========================================================================
    print("\n[Init] Loading policy configuration...")
    policy_config = load_policy_config(args.checkpoint)
    print(f"  policy_type:    {policy_config['policy_type']}")
    print(f"  has_wrist:      {policy_config['has_wrist']}")
    print(f"  image_shape:    {policy_config['image_shape']}")
    print(f"  state_dim:      {policy_config['state_dim']}")
    print(f"  action_dim:     {policy_config['action_dim']}")
    print(f"  n_obs_steps:    {policy_config['n_obs_steps']}")
    print(f"  n_action_steps: {policy_config['n_action_steps']}")

    if policy_config['policy_type'] != 'ditflow':
        print(f"\n[Warning] Expected policy type 'ditflow' but got '{policy_config['policy_type']}'.")
        print(f"  This script is designed for DiT Flow policies (trained by script 7).")
        print(f"  For Diffusion Policy (trained by script 4), use script 5 instead.")
        resp = input("  Continue anyway? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            return

    # =========================================================================
    # Initialize PS5 controller
    # =========================================================================
    ps5 = None
    stdin_ctrl = None

    if PYGAME_AVAILABLE:
        ps5 = PS5Controller()
        if not ps5.initialize():
            print("[Warning] No PS5 controller found, falling back to stdin")
            ps5.close()
            ps5 = None
            stdin_ctrl = StdinController()
    else:
        stdin_ctrl = StdinController()

    # =========================================================================
    # Parallel initialization: Arm + Cameras + Policy
    # =========================================================================
    piper = None
    fixed_cam = None
    wrist_cam = None
    policy = None
    preprocessor = None
    postprocessor = None
    init_errors = []
    n_action_steps_loaded = args.n_action_steps or policy_config['n_action_steps']

    def init_arm():
        arm = PiperController(args.can_interface)
        arm._speed_percent = args.speed
        if not arm.connect():
            raise RuntimeError("Arm connection failed")
        if not arm.enable():
            raise RuntimeError("Arm enable failed")
        return arm

    def init_cameras():
        cams = {"fixed": None, "wrist": None}

        front_cam_id = normalize_camera_value(args.front_cam)
        if is_camera_enabled(front_cam_id):
            cam = CameraCapture(front_cam_id, 640, 480,
                               name="front")
            if cam.start():
                cams["fixed"] = cam
            else:
                raise RuntimeError(f"Front camera failed: {front_cam_id}")
        else:
            raise RuntimeError("Front camera is required!")

        if policy_config['has_wrist']:
            wrist_cam_id = normalize_camera_value(args.wrist_cam)
            if is_camera_enabled(wrist_cam_id):
                cam = CameraCapture(wrist_cam_id, 640, 480,
                                   name="wrist")
                if cam.start():
                    cams["wrist"] = cam
                else:
                    raise RuntimeError(f"Wrist camera failed: {wrist_cam_id}")
            else:
                print("[Warning] Policy expects wrist camera but it's disabled (-1)")

        return cams

    print("\n[Init] Starting parallel initialization (arm + cameras)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        arm_future = executor.submit(init_arm)
        cam_future = executor.submit(init_cameras)

        # Load policy on MAIN THREAD (CUDA must be initialized on main thread)
        print("[Init] Loading DiT Flow policy on main thread...", flush=True)
        try:
            policy, preprocessor, postprocessor, _, n_action_steps_loaded = \
                load_ditflow_policy(
                    pretrained_dir=args.checkpoint,
                    device=args.device,
                    num_inference_steps=args.num_inference_steps,
                    n_action_steps=args.n_action_steps,
                )
        except Exception as e:
            init_errors.append(f"Policy: {e}")
            import traceback
            traceback.print_exc()

        try:
            piper = arm_future.result(timeout=30)
        except Exception as e:
            init_errors.append(f"Arm: {e}")

        try:
            cams = cam_future.result(timeout=30)
            fixed_cam = cams["fixed"]
            wrist_cam = cams["wrist"]
        except Exception as e:
            init_errors.append(f"Cameras: {e}")

    def cleanup_and_exit(exit_code: int = 1):
        print(f"\n[Cleanup] Shutting down (exit_code={exit_code})...")
        try:
            if piper:
                piper.disconnect()
        except Exception:
            pass
        try:
            if fixed_cam:
                fixed_cam.stop()
        except Exception:
            pass
        try:
            if wrist_cam:
                wrist_cam.stop()
        except Exception:
            pass
        try:
            if ps5:
                ps5.close()
        except Exception:
            pass
        try:
            if CV2_AVAILABLE:
                cv2.destroyAllWindows()
        except Exception:
            pass
        print("[Cleanup] Done. Exiting.")
        os._exit(exit_code)

    if init_errors:
        print("\n[Error] Initialization failed:")
        for err in init_errors:
            print(f"  - {err}")
        cleanup_and_exit(1)

    if args.n_action_steps is None:
        args.n_action_steps = n_action_steps_loaded

    if not piper.wait_until_ready(timeout=10.0):
        print("[Error] Arm not ready. Try power cycling.")
        cleanup_and_exit(1)

    # Go to home position first (safe retracted pose)
    print("\n[Init] Going to home position first...")
    piper.go_to_home()
    time.sleep(2.0)
    piper.open_gripper()
    time.sleep(0.5)

    # Then go to start position for first episode
    print("[Init] Going to start position...")
    piper.go_to_start()
    time.sleep(2.0)

    # =========================================================================
    # Main interaction loop
    # =========================================================================
    print("\n" + "=" * 60)
    print("Ready for DiT Flow evaluation!")
    print("=" * 60)
    if ps5:
        print("  Square:    Start/Stop episode")
        print("  Cross:     Go home / Abort episode")
        print("  Triangle:  Toggle gripper (overrides policy during episode)")
        print("  Share:     Emergency stop / Resume")
        print("  Options:   Re-enable arm")
        print("  PS Button: Quit")
    else:
        print("  s = Start/Stop episode")
        print("  h = Go to home position")
        print("  g = Toggle gripper manually")
        print("  e = Emergency stop")
        print("  q = Quit")
    print(f"\n  Policy type: DiT Flow")
    print(f"  Action mode: RELATIVE delta (8D)")
    print(f"  N action steps: {args.n_action_steps}")
    print(f"  Show camera: {args.show_camera}")
    print("=" * 60 + "\n")

    results = EvaluationResults()
    episode_id = 0
    in_episode = False
    gripper_open = True

    try:
        while episode_id < args.num_episodes:
            start_stop = False
            go_home = False
            toggle_gripper = False
            e_stop = False
            quit_prog = False
            re_enable = False

            if ps5:
                ps5.update()
                start_stop = ps5.get_button_pressed(PS5Buttons.SQUARE)
                go_home = ps5.get_button_pressed(PS5Buttons.CROSS)
                toggle_gripper = ps5.get_button_pressed(PS5Buttons.TRIANGLE)
                e_stop = ps5.get_button_pressed(PS5Buttons.SHARE)
                quit_prog = ps5.get_button_pressed(PS5Buttons.PS)
                re_enable = ps5.get_button_pressed(PS5Buttons.OPTIONS)
            elif stdin_ctrl:
                events = stdin_ctrl.poll()
                start_stop = events.get('start_stop', False)
                go_home = events.get('home', False)
                toggle_gripper = events.get('gripper', False)
                e_stop = events.get('emergency_stop', False)
                quit_prog = events.get('quit', False)

            if quit_prog:
                print("\n[Main] Quit requested.")
                break

            if e_stop:
                if piper._emergency_stop:
                    piper.clear_emergency_stop()
                    if ps5:
                        ps5.rumble_short(2)
                else:
                    piper.emergency_stop()
                    if ps5:
                        ps5.rumble_short(3, intensity=1.0)
                time.sleep(0.5)
                continue

            if re_enable:
                print("[Main] Re-enabling arm...")
                piper.enable()
                piper.set_speed(args.speed)
                if ps5:
                    ps5.rumble_short(1)
                continue

            if go_home and not in_episode:
                piper.go_to_home()
                time.sleep(1.0)
                if ps5:
                    ps5.rumble_short(1)
                continue

            if toggle_gripper and not in_episode:
                if gripper_open:
                    piper.close_gripper()
                    gripper_open = False
                else:
                    piper.open_gripper()
                    gripper_open = True
                if ps5:
                    ps5.rumble_short(1)
                continue

            if start_stop:
                if not in_episode:
                    in_episode = True
                    if ps5:
                        ps5.rumble_short(1)

                    video_recorder = None
                    if args.record_video:
                        video_path = str(out_dir / f"episode_{episode_id:03d}.mp4")
                        video_recorder = VideoRecorder(
                            video_path,
                            fps=args.control_freq,
                            width=640,
                            height=480,
                        )

                    result = run_episode(
                        piper=piper,
                        fixed_cam=fixed_cam,
                        wrist_cam=wrist_cam,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        policy_config=policy_config,
                        episode_id=episode_id,
                        max_steps=args.max_steps,
                        n_action_steps=args.n_action_steps,
                        control_freq=args.control_freq,
                        device=args.device,
                        video_recorder=video_recorder,
                        ps5=ps5,
                        stdin_ctrl=stdin_ctrl,
                        show_camera=args.show_camera,
                    )

                    in_episode = False

                    if video_recorder is not None:
                        video_recorder.save()

                    results.episodes.append(result)
                    episode_id += 1

                    if ps5:
                        ps5.rumble_short(2)

                    print(f"\n[Progress] {episode_id}/{args.num_episodes} episodes completed")
                    print(f"  Avg steps: {results.avg_steps:.0f}")
                    print(f"  Avg duration: {results.avg_duration:.1f}s")

                    piper.go_to_start()
                    time.sleep(1.0)
                    piper.open_gripper()
                    gripper_open = True

                    print("\nPress Square for next episode (or PS to quit)...")

            # Camera display during idle
            if args.show_camera and CV2_AVAILABLE and not in_episode:
                frame = fixed_cam.get_frame()
                if frame is not None:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(display_frame, f"IDLE [DiTFlow] | EP {episode_id}/{args.num_episodes}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    pose = piper.get_ee_pose_meters()
                    if pose:
                        cv2.putText(display_frame,
                                    f"Pos: ({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f})m",
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, "Press Square to start",
                                (10, display_frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    if piper._emergency_stop:
                        cv2.putText(display_frame, "E-STOP",
                                    (display_frame.shape[1] // 2 - 50, display_frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("Piper DiTFlow Eval", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Main] Quit from camera window.")
                        break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        # Cleanup: always go home before disconnecting
        print("\n[Main] Cleaning up...")

        try:
            if piper and piper.enabled:
                if piper._emergency_stop:
                    piper.clear_emergency_stop()
                    time.sleep(0.5)
                piper.open_gripper()
                time.sleep(0.5)
                print("[Main] Returning to home position before shutdown...")
                piper.go_to_home()
                time.sleep(3.0)
        except Exception as e:
            print(f"[Main] Error returning to home: {e}")

        # Disconnect arm properly (disables motors)
        try:
            if piper:
                piper.disconnect()
        except Exception:
            pass

        # Close controllers
        try:
            if ps5:
                ps5.close()
        except Exception:
            pass

        # Stop cameras
        try:
            if fixed_cam:
                fixed_cam.stop()
        except Exception:
            pass
        try:
            if wrist_cam:
                wrist_cam.stop()
        except Exception:
            pass

        if CV2_AVAILABLE:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if results.num_episodes > 0:
            try:
                results.save(str(out_dir / "eval_results.json"))
            except Exception:
                pass

        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Policy type:    DiT Flow")
        print(f"  Total episodes: {results.num_episodes}")
        print(f"  Avg steps:      {results.avg_steps:.0f}")
        print(f"  Avg duration:   {results.avg_duration:.1f}s")
        print("=" * 60)
        print("\n[Main] Done!")


if __name__ == "__main__":
    main()
