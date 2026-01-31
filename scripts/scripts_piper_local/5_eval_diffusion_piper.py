#!/usr/bin/env python3
"""Script 5: Evaluate Diffusion Policy on Real Piper Arm.

This script evaluates a trained diffusion policy on the real Piper robotic arm:
- Task A: Pick from **random table position** → Place at **plate center**

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation
python 5_eval_diffusion_piper.py \
    --checkpoint runs/piper_diffusion/checkpoints/last/pretrained_model \
    --num_episodes 10

# Full options
python 5_eval_diffusion_piper.py \
    --checkpoint runs/piper_diffusion/checkpoints/last/pretrained_model \
    --can_interface can0 \
    --num_episodes 20 \
    --max_steps 500 \
    --n_action_steps 8 \
    --control_freq 30 \
    --out_dir runs/piper_diffusion/eval_videos \
    --record_video \
    --seed 42

=============================================================================
INFERENCE PIPELINE
=============================================================================
At each inference step (every n_action_steps):
1. Capture images from fixed and wrist cameras
2. Read current EE pose and gripper state
3. Preprocess observations (normalize)
4. Run diffusion policy to predict action chunk (8 steps)
5. Post-process actions (unnormalize)
6. Execute actions on the robot

=============================================================================
SUCCESS CRITERIA
=============================================================================
An episode is successful if:
1. Gripper closes on the object (grasp detected)
2. Object placed within 5cm of plate center
3. No emergency stop triggered
4. Episode completed within max_steps
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Piper SDK
from piper_sdk import C_PiperInterface_V2


# =============================================================================
# Constants and Parameters
# =============================================================================

# === Workspace Parameters ===
PLATE_CENTER = (0.0, 0.15, 0.0)       # Goal position (fixed) in meters
GRASP_HEIGHT = 0.15                    # Z height for grasping (meters)
HOVER_HEIGHT = 0.25                    # Safe height for movement (meters)
SUCCESS_RADIUS = 0.05                  # 5cm success radius

# === Control Parameters ===
CONTROL_FREQ = 30                      # Hz
GRIPPER_OPEN_POS = 70000               # Gripper open position (0.001mm = 70mm)
GRIPPER_CLOSE_POS = 0                  # Gripper closed position
GRIPPER_SPEED = 1000                   # Gripper speed
GRIPPER_EFFORT = 500                   # Gripper force (0.001 N/m)

# === Motion Parameters ===
MOTION_SPEED_PERCENT = 20              # Conservative speed (0-100%)

# === Workspace Limits (Safety) ===
WORKSPACE_LIMITS = {
    "x_min": -0.3,  "x_max": 0.3,
    "y_min": 0.0,   "y_max": 0.4,
    "z_min": 0.05,  "z_max": 0.4,
}

# === Default Orientation ===
DEFAULT_ORIENTATION_EULER = (180.0, 0.0, 0.0)  # RX, RY, RZ in degrees


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
    return round(value_m * 1_000_000)


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
    """Wrapper for USB camera capture with background thread."""
    
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest_frame = frame_rgb
            time.sleep(0.001)
            
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
            pose = self.get_ee_pose_sdk()
            if pose is not None:
                self.piper.EndPoseCtrl(*pose)
        print("[Piper] EMERGENCY STOP ACTIVATED!")
        
    def clear_emergency_stop(self):
        """Clear emergency stop flag."""
        self._emergency_stop = False
        print("[Piper] Emergency stop cleared.")
    
    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is active."""
        return self._emergency_stop
        
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
            gripper_pos = msg.gripper_state.grippers_angle
            normalized = gripper_pos / GRIPPER_OPEN_POS
            return max(0.0, min(1.0, normalized))
        except Exception as e:
            print(f"[Piper] Error reading gripper: {e}")
            return None
    
    def send_ee_pose(self, x: float, y: float, z: float, 
                     rx: float, ry: float, rz: float):
        """Send target end-effector pose (meters and degrees)."""
        if self._emergency_stop:
            return
        
        # Apply workspace limits for safety
        x = max(WORKSPACE_LIMITS["x_min"], min(WORKSPACE_LIMITS["x_max"], x))
        y = max(WORKSPACE_LIMITS["y_min"], min(WORKSPACE_LIMITS["y_max"], y))
        z = max(WORKSPACE_LIMITS["z_min"], min(WORKSPACE_LIMITS["z_max"], z))
        
        X = meters_to_sdk(x)
        Y = meters_to_sdk(y)
        Z = meters_to_sdk(z)
        RX = degrees_to_sdk(rx)
        RY = degrees_to_sdk(ry)
        RZ = degrees_to_sdk(rz)
        
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
    
    def send_ee_pose_quat(self, pos: np.ndarray, quat: np.ndarray):
        """Send target end-effector pose with quaternion orientation."""
        rx, ry, rz = quat_to_euler(quat[0], quat[1], quat[2], quat[3])
        self.send_ee_pose(pos[0], pos[1], pos[2], rx, ry, rz)
    
    def set_gripper(self, gripper_value: float):
        """Set gripper position (0=close, 1=open)."""
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
        self.piper.MotionCtrl_2(0x01, 0x01, MOTION_SPEED_PERCENT, 0x00)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(2.0)
        self.piper.MotionCtrl_2(0x01, 0x02, MOTION_SPEED_PERCENT, 0x00)
        
        self._home_position = self.get_ee_pose_meters()
        if self._home_position:
            print(f"[Piper] Home position: x={self._home_position[0]:.3f}, "
                  f"y={self._home_position[1]:.3f}, z={self._home_position[2]:.3f}")


# =============================================================================
# Policy Loading
# =============================================================================

def load_policy_config(pretrained_dir: str) -> Dict[str, Any]:
    """Load policy configuration from checkpoint."""
    config_path = Path(pretrained_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    input_features = config_dict.get("input_features", {})
    output_features = config_dict.get("output_features", {})
    
    # Check for wrist camera
    has_wrist = "observation.wrist_image" in input_features
    
    # Get image shape from table camera
    image_shape = None
    if "observation.image" in input_features:
        image_shape = tuple(input_features["observation.image"]["shape"])
    
    # Get state dimension
    state_dim = None
    if "observation.state" in input_features:
        state_dim = input_features["observation.state"]["shape"][0]
    
    # Get action dimension
    action_dim = None
    if "action" in output_features:
        action_dim = output_features["action"]["shape"][0]
    
    return {
        "has_wrist": has_wrist,
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "raw_config": config_dict,
    }


def load_diffusion_policy(
    pretrained_dir: str,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
) -> Tuple[Any, Any, Any, int, int]:
    """Load LeRobot diffusion policy from checkpoint.
    
    Returns:
        Tuple of (policy, preprocessor, postprocessor, num_inference_steps, n_action_steps)
    """
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = Path(pretrained_dir)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"
    
    print(f"[Policy] Loading config from {config_path}...", flush=True)
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Remove 'type' field if present
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
    
    # Convert lists to tuples
    for field_name in ["crop_shape", "optimizer_betas", "down_dims"]:
        if field_name in config_dict and isinstance(config_dict[field_name], list):
            config_dict[field_name] = tuple(config_dict[field_name])
    
    # Override parameters if specified
    if num_inference_steps is not None:
        num_train_timesteps = config_dict.get("num_train_timesteps", 100)
        if num_inference_steps > num_train_timesteps:
            num_inference_steps = num_train_timesteps
        config_dict["num_inference_steps"] = num_inference_steps
    
    if n_action_steps is not None:
        horizon = config_dict.get("horizon", 16)
        if n_action_steps > horizon:
            n_action_steps = horizon
        config_dict["n_action_steps"] = n_action_steps
    
    # Create config and policy
    print(f"[Policy] Creating DiffusionConfig...", flush=True)
    cfg = DiffusionConfig(**config_dict)
    
    print(f"[Policy] Creating DiffusionPolicy model...", flush=True)
    policy = DiffusionPolicy(cfg)
    
    # Load weights
    print(f"[Policy] Loading model weights...", flush=True)
    state_dict = load_file(model_path)
    policy.load_state_dict(state_dict)
    
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
    
    actual_inference_steps = cfg.num_inference_steps or cfg.num_train_timesteps
    actual_n_action_steps = cfg.n_action_steps
    
    print(f"[Policy] Loaded! (num_inference_steps={actual_inference_steps}, "
          f"n_action_steps={actual_n_action_steps})", flush=True)
    
    return policy, preprocessor, postprocessor, actual_inference_steps, actual_n_action_steps


# =============================================================================
# Video Recording
# =============================================================================

class VideoRecorder:
    """Record video from camera frames."""
    
    def __init__(self, output_path: str, fps: int = 30, width: int = 640, height: int = 480):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.writer = None
        self.frames = []
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame (RGB format)."""
        # Resize if needed
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        self.frames.append(frame.copy())
    
    def add_combined_frame(self, fixed_frame: np.ndarray, wrist_frame: np.ndarray):
        """Add a combined frame with both camera views."""
        # Resize frames
        fixed_resized = cv2.resize(fixed_frame, (self.width // 2, self.height // 2))
        wrist_resized = cv2.resize(wrist_frame, (self.width // 2, self.height // 2))
        
        # Combine horizontally
        combined = np.concatenate([fixed_resized, wrist_resized], axis=1)
        self.frames.append(combined)
    
    def save(self):
        """Save recorded frames to video file."""
        if not self.frames:
            print("[Video] No frames to save!")
            return
        
        # Ensure directory exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine output size from first frame
        h, w = self.frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        
        for frame in self.frames:
            # Convert RGB to BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        
        writer.release()
        print(f"[Video] Saved {len(self.frames)} frames to {self.output_path}")
        
    def clear(self):
        """Clear recorded frames."""
        self.frames = []


# =============================================================================
# Evaluation Results
# =============================================================================

@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""
    episode_id: int
    success: bool
    steps: int
    final_distance: float
    grasp_detected: bool = False
    emergency_stopped: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.episode_id,
            "success": self.success,
            "steps": self.steps,
            "final_dist": self.final_distance,
            "grasp_detected": self.grasp_detected,
            "emergency_stopped": self.emergency_stopped,
        }


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    episodes: List[EpisodeResult] = field(default_factory=list)
    
    @property
    def num_episodes(self) -> int:
        return len(self.episodes)
    
    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for ep in self.episodes if ep.success) / len(self.episodes)
    
    @property
    def avg_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(ep.steps for ep in self.episodes) / len(self.episodes)
    
    def to_dict(self) -> Dict:
        return {
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }
    
    def save(self, output_path: str):
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Results] Saved to {output_path}")


# =============================================================================
# Keyboard Handler
# =============================================================================

class KeyboardHandler:
    """Non-blocking keyboard input handler."""
    
    def __init__(self):
        self.start_requested = False
        self.quit_requested = False
        self.emergency_stop_requested = False
        self._lock = threading.Lock()
        
        try:
            import keyboard
            self._keyboard_available = True
            keyboard.on_press_key('space', lambda _: self._on_space())
            keyboard.on_press_key('escape', lambda _: self._on_escape())
            keyboard.on_press_key('q', lambda _: self._on_quit())
            print("[Keyboard] Handler initialized. Controls: SPACE=start, ESC=e-stop, Q=quit")
        except ImportError:
            self._keyboard_available = False
            print("[Keyboard] WARNING: 'keyboard' module not available.")
    
    def _on_space(self):
        with self._lock:
            self.start_requested = True
    
    def _on_escape(self):
        with self._lock:
            self.emergency_stop_requested = True
    
    def _on_quit(self):
        with self._lock:
            self.quit_requested = True
    
    def poll(self) -> dict:
        with self._lock:
            events = {
                'start': self.start_requested,
                'quit': self.quit_requested,
                'emergency_stop': self.emergency_stop_requested,
            }
            self.start_requested = False
            self.quit_requested = False
            self.emergency_stop_requested = False
        return events


# =============================================================================
# Observation Builder
# =============================================================================

def build_observation(
    fixed_image: np.ndarray,
    wrist_image: Optional[np.ndarray],
    ee_pose: np.ndarray,
    gripper_state: float,
    policy_config: Dict,
    device: str,
    target_image_size: Tuple[int, int] = (128, 128),
) -> Dict[str, torch.Tensor]:
    """Build observation dictionary for policy inference.
    
    Args:
        fixed_image: Fixed camera RGB image (H, W, 3) uint8
        wrist_image: Wrist camera RGB image (H, W, 3) uint8, or None
        ee_pose: End-effector pose [x, y, z, qw, qx, qy, qz]
        gripper_state: Gripper position normalized [0, 1]
        policy_config: Policy configuration dict
        device: PyTorch device
        target_image_size: Target image size (H, W)
        
    Returns:
        Dictionary of observation tensors
        
    Note:
        Image is converted to float [0, 1] range here. The preprocessor from
        LeRobot should handle further normalization (e.g., ImageNet mean/std).
        If you see strange behavior, check if preprocessor expects uint8 or float.
    """
    obs = {}
    
    # Process fixed camera image
    # Resize to policy input size
    fixed_resized = cv2.resize(fixed_image, (target_image_size[1], target_image_size[0]))
    # Convert to tensor: (H, W, 3) uint8 -> (3, H, W) float [0, 1]
    # NOTE: LeRobot preprocessor may expect different format - check if issues occur
    fixed_tensor = torch.from_numpy(fixed_resized).permute(2, 0, 1).float() / 255.0
    obs["observation.image"] = fixed_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Process wrist camera image if policy expects it
    if policy_config.get("has_wrist", False) and wrist_image is not None:
        wrist_resized = cv2.resize(wrist_image, (target_image_size[1], target_image_size[0]))
        wrist_tensor = torch.from_numpy(wrist_resized).permute(2, 0, 1).float() / 255.0
        obs["observation.wrist_image"] = wrist_tensor.unsqueeze(0).to(device)
    
    # Build state vector based on policy's expected state_dim
    state_dim = policy_config.get("state_dim", 7)
    
    if state_dim == 7:
        # Only EE pose
        state = ee_pose
    elif state_dim == 8:
        # EE pose + gripper
        state = np.concatenate([ee_pose, [gripper_state]])
    else:
        # Default to EE pose
        state = ee_pose
    
    state_tensor = torch.from_numpy(state).float()
    obs["observation.state"] = state_tensor.unsqueeze(0).to(device)
    
    return obs


def extract_action(
    action_tensor: torch.Tensor,
    action_idx: int = 0,
) -> Tuple[np.ndarray, float]:
    """Extract EE pose and gripper action from policy output.
    
    Args:
        action_tensor: Action tensor from policy (1, horizon, action_dim) or (1, action_dim)
        action_idx: Index of action step to extract (for action chunking)
        
    Returns:
        Tuple of (ee_pose [x,y,z,qw,qx,qy,qz], gripper_value)
    """
    # Handle action chunking (multiple action steps)
    if action_tensor.dim() == 3:
        # Shape: (1, horizon, action_dim)
        action = action_tensor[0, action_idx].cpu().numpy()
    else:
        # Shape: (1, action_dim)
        action = action_tensor[0].cpu().numpy()
    
    # Extract components
    # Assuming action format: [x, y, z, qw, qx, qy, qz, gripper]
    if len(action) >= 8:
        ee_pose = action[:7]
        gripper = action[7]
    else:
        # Fallback if no gripper in action
        ee_pose = action[:7] if len(action) >= 7 else np.zeros(7)
        gripper = 0.5
    
    return ee_pose, float(gripper)


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy on Piper arm.")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model directory")
    parser.add_argument("--can_interface", type=str, default="can0",
                        help="CAN interface name")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Action steps per inference (None=use model config)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Diffusion denoising steps (None=use model config)")
    parser.add_argument("--control_freq", type=int, default=30,
                        help="Control frequency in Hz")
    parser.add_argument("--out_dir", type=str, default="eval_results",
                        help="Output directory for results and videos")
    parser.add_argument("--record_video", action="store_true",
                        help="Record video of each episode")
    parser.add_argument("--fixed_cam_id", type=int, default=0,
                        help="Fixed camera device ID")
    parser.add_argument("--wrist_cam_id", type=int, default=1,
                        help="Wrist camera device ID")
    parser.add_argument("--image_width", type=int, default=640,
                        help="Camera capture width")
    parser.add_argument("--image_height", type=int, default=480,
                        help="Camera capture height")
    parser.add_argument("--policy_image_size", type=int, default=128,
                        help="Image size expected by policy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device (cuda/cpu)")
    parser.add_argument("--plate_x", type=float, default=0.0,
                        help="Plate center X position")
    parser.add_argument("--plate_y", type=float, default=0.15,
                        help="Plate center Y position")
    parser.add_argument("--plate_z", type=float, default=0.0,
                        help="Plate center Z position")
    
    return parser.parse_args()


def run_episode(
    piper: PiperController,
    fixed_cam: CameraCapture,
    wrist_cam: CameraCapture,
    policy,
    preprocessor,
    postprocessor,
    policy_config: Dict,
    episode_id: int,
    max_steps: int,
    n_action_steps: int,
    control_freq: int,
    device: str,
    plate_center: Tuple[float, float, float],
    policy_image_size: int,
    video_recorder: Optional[VideoRecorder] = None,
) -> EpisodeResult:
    """Run a single evaluation episode.
    
    Returns:
        EpisodeResult with episode statistics
    """
    print(f"\n[Episode {episode_id}] Starting...")
    
    # Reset policy state
    policy.reset()
    
    control_period = 1.0 / control_freq
    step = 0
    action_chunk = None
    action_chunk_idx = 0
    grasp_detected = False
    
    # Move to initial position (hover above plate)
    piper.open_gripper()
    rx, ry, rz = DEFAULT_ORIENTATION_EULER
    piper.send_ee_pose(plate_center[0], plate_center[1], HOVER_HEIGHT, rx, ry, rz)
    time.sleep(2.0)  # Wait for arm to reach position
    
    print(f"[Episode {episode_id}] Running inference loop...")
    
    while step < max_steps:
        loop_start = time.time()
        
        # Check emergency stop
        if piper.is_emergency_stopped():
            print(f"[Episode {episode_id}] Emergency stop!")
            return EpisodeResult(
                episode_id=episode_id,
                success=False,
                steps=step,
                final_distance=float('inf'),
                emergency_stopped=True,
            )
        
        # Get observations
        fixed_img = fixed_cam.get_frame()
        wrist_img = wrist_cam.get_frame()
        pose_result = piper.get_ee_pose_quat()
        gripper_state = piper.get_gripper_state() or 0.5
        
        if fixed_img is None or pose_result is None:
            time.sleep(control_period)
            continue
        
        pos, quat = pose_result
        ee_pose = np.concatenate([pos, quat])
        
        # Record video frame
        if video_recorder is not None and fixed_img is not None:
            if wrist_img is not None:
                video_recorder.add_combined_frame(fixed_img, wrist_img)
            else:
                video_recorder.add_frame(fixed_img)
        
        # Run inference if needed (every n_action_steps or at start)
        if action_chunk is None or action_chunk_idx >= n_action_steps:
            # Build observation
            obs = build_observation(
                fixed_image=fixed_img,
                wrist_image=wrist_img,
                ee_pose=ee_pose,
                gripper_state=gripper_state,
                policy_config=policy_config,
                device=device,
                target_image_size=(policy_image_size, policy_image_size),
            )
            
            # Preprocess
            obs_normalized = preprocessor(obs)
            
            # Run policy inference
            with torch.no_grad():
                action_dict = policy.select_action(obs_normalized)
            
            # Postprocess (unnormalize)
            action_dict = postprocessor(action_dict)
            
            # Extract action chunk
            action_chunk = action_dict["action"]  # Shape: (1, horizon, action_dim) or similar
            action_chunk_idx = 0
            
            if step % 50 == 0:
                print(f"[Episode {episode_id}] Step {step}: Ran inference", flush=True)
        
        # Extract current action from chunk
        target_ee_pose, target_gripper = extract_action(action_chunk, action_chunk_idx)
        action_chunk_idx += 1
        
        # Execute action on robot
        target_quat = target_ee_pose[3:7]
        target_pos = target_ee_pose[:3]
        piper.send_ee_pose_quat(target_pos, target_quat)
        piper.set_gripper(target_gripper)
        
        # Detect grasp (gripper closing while near object)
        if target_gripper < 0.3 and gripper_state < 0.5:
            grasp_detected = True
        
        # Check success condition
        current_xy = pos[:2]
        goal_xy = np.array(plate_center[:2])
        distance = np.linalg.norm(current_xy - goal_xy)
        
        # Success: grasped object and within success radius of goal while placing
        if grasp_detected and distance < SUCCESS_RADIUS and target_gripper > 0.7:
            print(f"[Episode {episode_id}] SUCCESS at step {step}! Distance: {distance:.3f}m")
            return EpisodeResult(
                episode_id=episode_id,
                success=True,
                steps=step,
                final_distance=distance,
                grasp_detected=grasp_detected,
            )
        
        step += 1
        
        # Maintain control frequency
        elapsed = time.time() - loop_start
        sleep_time = control_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Episode ended without success
    pose_result = piper.get_ee_pose_quat()
    if pose_result:
        pos, _ = pose_result
        final_distance = np.linalg.norm(pos[:2] - np.array(plate_center[:2]))
    else:
        final_distance = float('inf')
    
    print(f"[Episode {episode_id}] TIMEOUT at step {step}. Final distance: {final_distance:.3f}m")
    return EpisodeResult(
        episode_id=episode_id,
        success=False,
        steps=step,
        final_distance=final_distance,
        grasp_detected=grasp_detected,
    )


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plate_center = (args.plate_x, args.plate_y, args.plate_z)
    
    print("=" * 60)
    print("Piper Diffusion Policy Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Control freq: {args.control_freq} Hz")
    print(f"Output dir: {args.out_dir}")
    print("=" * 60)
    
    # Initialize keyboard handler
    keyboard_handler = KeyboardHandler()
    
    # Load policy config first to check requirements
    print("\n[Init] Loading policy configuration...")
    policy_config = load_policy_config(args.checkpoint)
    print(f"  has_wrist: {policy_config['has_wrist']}")
    print(f"  image_shape: {policy_config['image_shape']}")
    print(f"  state_dim: {policy_config['state_dim']}")
    print(f"  action_dim: {policy_config['action_dim']}")
    
    # Initialize cameras
    print("\n[Init] Starting cameras...")
    fixed_cam = CameraCapture(args.fixed_cam_id, args.image_width, args.image_height)
    wrist_cam = None  # Only create if needed
    
    try:
        fixed_cam.start()
        if policy_config['has_wrist']:
            wrist_cam = CameraCapture(args.wrist_cam_id, args.image_width, args.image_height)
            wrist_cam.start()
        else:
            print("[Init] Policy does not use wrist camera, skipping.")
    except RuntimeError as e:
        print(f"[Error] Camera initialization failed: {e}")
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
    
    # Load policy
    print("\n[Init] Loading diffusion policy...")
    try:
        policy, preprocessor, postprocessor, num_inference_steps, n_action_steps = load_diffusion_policy(
            pretrained_dir=args.checkpoint,
            device=args.device,
            num_inference_steps=args.num_inference_steps,
            n_action_steps=args.n_action_steps,
        )
    except Exception as e:
        print(f"[Error] Policy loading failed: {e}")
        import traceback
        traceback.print_exc()
        piper.disconnect()
        fixed_cam.stop()
        wrist_cam.stop()
        return
    
    # Use loaded n_action_steps if not specified
    if args.n_action_steps is None:
        args.n_action_steps = n_action_steps
    
    print("\n" + "=" * 60)
    print("Ready for evaluation!")
    print("Press SPACE to start each episode.")
    print("Press Q to quit.")
    print("=" * 60 + "\n")
    
    # Evaluation results
    results = EvaluationResults()
    
    try:
        episode_id = 0
        
        while episode_id < args.num_episodes:
            # Poll keyboard
            events = keyboard_handler.poll()
            
            if events['quit']:
                print("\n[Main] Quit requested.")
                break
            
            if events['emergency_stop']:
                piper.emergency_stop()
                time.sleep(0.5)
                continue
            
            if events['start']:
                piper.clear_emergency_stop()
                
                # Create video recorder if needed
                video_recorder = None
                if args.record_video:
                    video_path = str(out_dir / f"episode_{episode_id:02d}.mp4")
                    video_recorder = VideoRecorder(
                        video_path, 
                        fps=args.control_freq,
                        width=args.image_width,
                        height=args.image_height // 2,  # Combined view
                    )
                
                # Run episode
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
                    plate_center=plate_center,
                    policy_image_size=args.policy_image_size,
                    video_recorder=video_recorder,
                )
                
                # Save video
                if video_recorder is not None:
                    suffix = "success" if result.success else "fail"
                    video_recorder.output_path = str(out_dir / f"episode_{episode_id:02d}_{suffix}.mp4")
                    video_recorder.save()
                
                # Record result
                results.episodes.append(result)
                episode_id += 1
                
                # Print progress
                print(f"\n[Progress] {episode_id}/{args.num_episodes} episodes")
                print(f"  Current success rate: {results.success_rate:.1%}")
                
                # Return to home and wait
                piper.open_gripper()
                piper.go_to_home()
                print("\nPress SPACE for next episode...")
            
            time.sleep(0.01)
    
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
        if wrist_cam is not None:
            wrist_cam.stop()
        
        # Save results
        results.save(str(out_dir / "eval_results.json"))
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"  Total episodes: {results.num_episodes}")
        print(f"  Success rate: {results.success_rate:.1%}")
        print(f"  Average steps: {results.avg_steps:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
