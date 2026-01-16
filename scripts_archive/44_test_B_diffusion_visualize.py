#!/usr/bin/env python3
"""Evaluate and visualize the LeRobot Diffusion Policy for Task B in Isaac Lab.

This script loads a diffusion policy trained by ``34_train_B_diffusion_goal.py``
and evaluates it on the REVERSE pick-and-place task:
- Cube starts at the GOAL position (plate center at [0.5, 0.0])
- Policy picks it up and places it at a random table position

A camera is injected into the Isaac Lab scene so the policy receives RGB + EE state.
The RGB stream is recorded to an MP4 video.

=============================================================================
TASK B: REVERSE PICK-AND-PLACE
=============================================================================
- Initial state: Cube at plate center (0.5, 0.0)
- Task: Pick cube from plate center, place at random table position
- Success: Cube within success_radius of target place position

This is the OPPOSITE of Task A (forward task: random table -> plate center).

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation (headless mode, 5 episodes)
CUDA_VISIBLE_DEVICES=1 python scripts/44_test_B_diffusion_visualize.py \
    --checkpoint runs/diffusion_B_goal/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_B_goal/videos \
    --num_episodes 5 \
    --headless

# With XYZ visualization
CUDA_VISIBLE_DEVICES=1 python scripts/44_test_B_diffusion_visualize.py \
    --checkpoint runs/diffusion_B_goal/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_B_goal/videos \
    --num_episodes 3 \
    --visualize_xyz \
    --headless --horizon 500 --n_action_steps 16

# With action chunk visualization
CUDA_VISIBLE_DEVICES=1 python scripts/44_test_B_diffusion_visualize.py \
    --checkpoint runs/diffusion_B_goal/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_B_goal/videos \
    --num_episodes 2 \
    --visualize_action_chunk \
    --headless --horizon 500 --n_action_steps 16

# Custom target place position
CUDA_VISIBLE_DEVICES=1 python scripts/44_test_B_diffusion_visualize.py \
    --checkpoint runs/diffusion_B_goal/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_B_goal/videos \
    --target_place_xy 0.3 0.1 \
    --headless

=============================================================================
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera utilities (same as script 41)
# ---------------------------------------------------------------------------
def compute_camera_quat_from_lookat(eye: Tuple[float, float, float], target: Tuple[float, float, float], up: Tuple[float, float, float] = (0, 0, 1)) -> Tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion for a camera that looks at ``target`` from ``eye``."""
    from scipy.spatial.transform import Rotation as R

    eye_v = np.array(eye, dtype=np.float64)
    target_v = np.array(target, dtype=np.float64)
    up_v = np.array(up, dtype=np.float64)

    forward = target_v - eye_v
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up_v)
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)

    rotation_matrix = np.column_stack([right, down, forward])
    rot = R.from_matrix(rotation_matrix)
    q_xyzw = rot.as_quat()  # [x, y, z, w]
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    return (qw, qx, qy, qz)


def create_target_markers(num_envs: int, device: str):
    """Create visualization markers for start and target positions.
    
    For Task B:
    - Red markers: Target place positions (where cube should be placed)
    - Green markers: Start/goal positions (plate center, where cube starts)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    
    marker_radius = 0.05
    marker_height = 0.002
    table_z = 0.0
    marker_z = table_z + marker_height / 2 + 0.001
    
    # Red marker for target place positions (where we want to place the cube)
    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarkers",
        markers={
            "target": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # Red
                ),
            ),
        },
    )
    target_markers = VisualizationMarkers(target_marker_cfg)
    
    # Green marker for start position (plate center, where cube starts)
    start_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/StartMarkers",
        markers={
            "start": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),  # Green
                ),
            ),
        },
    )
    start_markers = VisualizationMarkers(start_marker_cfg)
    
    return target_markers, start_markers, marker_z


def update_target_markers(
    target_markers,
    start_markers,
    target_xy: tuple,
    start_xy: tuple,
    marker_z: float,
    env,
):
    """Update the positions of target and start markers."""
    import torch
    
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    env_origins = env.unwrapped.scene.env_origins

    target_x = float(target_xy[0])
    target_y = float(target_xy[1])
    start_x = float(start_xy[0])
    start_y = float(start_xy[1])
    
    # Target marker positions
    target_positions = torch.zeros((num_envs, 3), device=device)
    target_positions[:, 0] = target_x
    target_positions[:, 1] = target_y
    target_positions[:, 2] = marker_z
    target_positions_w = target_positions + env_origins
    
    # Start marker positions
    start_positions = torch.zeros((num_envs, 3), device=device)
    start_positions[:, 0] = start_x
    start_positions[:, 1] = start_y
    start_positions[:, 2] = marker_z
    start_positions_w = start_positions + env_origins
    
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
    
    target_markers.visualize(target_positions_w, identity_quat)
    start_markers.visualize(start_positions_w, identity_quat)


def add_camera_to_env_cfg(env_cfg, image_width: int, image_height: int) -> None:
    """Inject table-view and wrist cameras into Isaac Lab env cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    camera_eye = (1.6, 0.7, 0.8)
    camera_lookat = (0.4, 0.0, 0.2)
    camera_quat = compute_camera_quat_from_lookat(camera_eye, camera_lookat)

    env_cfg.scene.table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=camera_eye,
            rot=camera_quat,
            convention="ros",
        ),
    )
    
    env_cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )

    env_cfg.scene.env_spacing = 5.0

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "ee_frame"):
        env_cfg.scene.ee_frame.debug_vis = False
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render"):
        env_cfg.sim.render.antialiasing_mode = "FXAA"


def make_env_with_camera(
    task_id: str,
    num_envs: int,
    device: str,
    use_fabric: bool,
    image_width: int,
    image_height: int,
    episode_length_s: float | None = None,
    disable_terminations: bool = False,
):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))

    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s

    if disable_terminations:
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "object_dropping"):
            env_cfg.terminations.object_dropping = None

    add_camera_to_env_cfg(env_cfg, image_width, image_height)

    env = gym.make(task_id, cfg=env_cfg)
    return env


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Task B diffusion policy (reverse pick-and-place) in Isaac Lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/diffusion_B_goal/checkpoints/checkpoints/last/pretrained_model",
        help="Path to LeRobot pretrained_model directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/diffusion_B_goal/videos",
        help="Output directory for recorded videos.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab Gym task id.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="How many episodes to roll out.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for MP4 writer.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="Camera width.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=128,
        help="Camera height.",
    )
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disable Fabric backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of diffusion denoising steps at inference.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Number of action steps to execute before re-inferring.",
    )
    parser.add_argument(
        "--visualize_xyz",
        action="store_true",
        help="Generate XYZ curve visualization videos.",
    )
    parser.add_argument(
        "--visualize_action_chunk",
        action="store_true",
        help="Generate action chunk visualization videos.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Enable overfit testing mode: use saved initial poses from training.",
    )
    parser.add_argument(
        "--target_place_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Custom target place XY position (e.g., --target_place_xy 0.3 0.1). "
             "If not specified, random table position is sampled each episode.",
    )
    parser.add_argument(
        "--start_xy",
        type=float,
        nargs=2,
        default=[0.5, 0.0],
        metavar=("X", "Y"),
        help="Initial cube XY position (default: plate center at 0.5 0.0).",
    )

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ---------------------------------------------------------------------------
# Overfit env init loading
# ---------------------------------------------------------------------------
def load_overfit_env_init(checkpoint_dir: str) -> Dict[str, Any] | None:
    """Load overfit environment initialization parameters from checkpoint directory."""
    import json
    from pathlib import Path
    
    checkpoint_path = Path(checkpoint_dir)
    
    possible_paths = [
        checkpoint_path / "overfit_env_init.json",
        checkpoint_path.parent / "overfit_env_init.json",
        checkpoint_path.parent.parent / "overfit_env_init.json",
        checkpoint_path.parent.parent.parent / "overfit_env_init.json",
        checkpoint_path.parent.parent.parent.parent / "overfit_env_init.json",
    ]
    
    for init_path in possible_paths:
        if init_path.exists():
            print(f"[Overfit Mode] Loading env init params from: {init_path}")
            with open(init_path, "r") as f:
                overfit_env_init = json.load(f)
            print(f"  initial_obj_pose: {overfit_env_init['initial_obj_pose']}")
            print(f"  place_pose: {overfit_env_init.get('place_pose', 'N/A')}")
            return overfit_env_init
    
    print(f"[Overfit Mode] WARNING: overfit_env_init.json not found!")
    return None


# ---------------------------------------------------------------------------
# Policy loading helpers
# ---------------------------------------------------------------------------
def load_policy_config(pretrained_dir: str) -> Dict[str, Any]:
    """Load policy configuration without loading model weights."""
    import json
    from pathlib import Path
    
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
    
    include_obj_pose = (state_dim == 14) if state_dim is not None else False
    
    return {
        "has_wrist": has_wrist,
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "include_obj_pose": include_obj_pose,
        "raw_config": config_dict,
    }


def load_diffusion_policy(
    pretrained_dir: str,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
) -> Tuple[Any, Any, Any]:
    """Load LeRobot diffusion policy from checkpoint directory."""
    import json
    import os
    from pathlib import Path
    
    print(f"[load_policy] dir={pretrained_dir}, device={device}", flush=True)

    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = Path(pretrained_dir)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
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
    if "crop_shape" in config_dict and isinstance(config_dict["crop_shape"], list):
        config_dict["crop_shape"] = tuple(config_dict["crop_shape"])
    if "optimizer_betas" in config_dict and isinstance(config_dict["optimizer_betas"], list):
        config_dict["optimizer_betas"] = tuple(config_dict["optimizer_betas"])
    if "down_dims" in config_dict and isinstance(config_dict["down_dims"], list):
        config_dict["down_dims"] = tuple(config_dict["down_dims"])
    
    # Override num_inference_steps if specified
    if num_inference_steps is not None:
        num_train_timesteps = config_dict.get("num_train_timesteps", 100)
        if num_inference_steps > num_train_timesteps:
            num_inference_steps = num_train_timesteps
        config_dict["num_inference_steps"] = num_inference_steps
    
    # Override n_action_steps if specified
    if n_action_steps is not None:
        horizon = config_dict.get("horizon", 16)
        if n_action_steps > horizon:
            n_action_steps = horizon
        config_dict["n_action_steps"] = n_action_steps
    
    cfg = DiffusionConfig(**config_dict)
    
    t0 = time.time()
    policy = DiffusionPolicy(cfg)
    print(f"[load_policy] DiffusionPolicy created in {time.time()-t0:.2f}s", flush=True)
    
    state_dict = load_file(model_path)
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    
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
    
    print(f"[load_policy] Policy loaded (num_inference_steps={actual_inference_steps}, n_action_steps={actual_n_action_steps})", flush=True)
    return policy, preprocessor, postprocessor, actual_inference_steps, actual_n_action_steps


def sample_table_xy(rng: np.random.Generator, goal_xy: tuple = (0.5, 0.0), min_dist: float = 0.1) -> tuple:
    """Sample a random XY position on the table, avoiding the goal area.
    
    Table bounds (approximate):
    - X: [0.25, 0.75]
    - Y: [-0.3, 0.3]
    
    Args:
        rng: NumPy random generator.
        goal_xy: Goal position to avoid.
        min_dist: Minimum distance from goal.
        
    Returns:
        Tuple (x, y) of sampled position.
    """
    x_min, x_max = 0.25, 0.75
    y_min, y_max = -0.3, 0.3
    
    for _ in range(100):  # Max attempts
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        dist = np.sqrt((x - goal_xy[0])**2 + (y - goal_xy[1])**2)
        if dist >= min_dist:
            return (x, y)
    
    # Fallback: return corner position
    return (x_min, y_max)


def run_episode(
    env,
    policy,
    preprocessor,
    postprocessor,
    horizon: int,
    writer,
    start_xy: tuple = (0.5, 0.0),
    target_xy: tuple = None,
    success_radius: float = 0.05,
    has_wrist: bool = False,
    include_obj_pose: bool = False,
    xyz_visualizer=None,
    overfit_env_init: dict | None = None,
    n_action_steps: int = 8,
    rng: np.random.Generator = None,
    target_markers=None,
    start_markers=None,
    marker_z: float = 0.002,
    action_chunk_visualizer=None,
) -> dict:
    """Run one episode of Task B (reverse pick-and-place).
    
    Args:
        env: Isaac Lab environment.
        policy: Diffusion policy.
        preprocessor: Preprocessor pipeline.
        postprocessor: Postprocessor pipeline.
        horizon: Maximum steps per episode.
        writer: Video writer (or None).
        start_xy: Initial cube XY position (plate center).
        target_xy: Target place XY position. If None, sample randomly.
        success_radius: Success radius in meters.
        has_wrist: Whether the policy expects wrist camera input.
        include_obj_pose: Whether the policy expects object pose in state.
        xyz_visualizer: Optional XYZCurveVisualizer.
        overfit_env_init: Optional dict with initial poses for overfit testing.
        n_action_steps: Number of action steps per inference.
        rng: NumPy random generator for sampling target position.
        target_markers: VisualizationMarkers for target position.
        start_markers: VisualizationMarkers for start position.
        marker_z: Z height for markers.
        action_chunk_visualizer: Optional ActionChunkVisualizer.
        
    Returns:
        Dictionary with episode statistics.
    """
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose

    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None) if has_wrist else None
    device = env.unwrapped.device

    # Reset policy action queue
    policy.reset()

    # Reset environment
    obs_dict, _ = env.reset()
    
    # =========================================================================
    # Task B: Initialize cube at start position (plate center)
    # =========================================================================
    if overfit_env_init is not None:
        # Use saved initial pose from training
        print(f"  [Overfit Mode] Teleporting object to saved initial pose...")
        initial_obj_pose = torch.tensor(
            overfit_env_init["initial_obj_pose"], 
            dtype=torch.float32, 
            device=device
        ).unsqueeze(0)
        teleport_object_to_pose(env, initial_obj_pose, name="object")
        
        # Use saved place_pose as target if available
        if target_xy is None and "place_pose" in overfit_env_init:
            target_xy = (overfit_env_init["place_pose"][0], overfit_env_init["place_pose"][1])
            print(f"  [Overfit Mode] Using saved place_pose as target: {target_xy}")
    else:
        # Teleport cube to start position (plate center)
        print(f"  Teleporting cube to start position: [{start_xy[0]:.3f}, {start_xy[1]:.3f}]...")
        current_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        init_obj_z = 0.055  # Cube height above table
        new_obj_pose = torch.tensor(
            [start_xy[0], start_xy[1], init_obj_z,
             current_obj_pose[3], current_obj_pose[4], current_obj_pose[5], current_obj_pose[6]],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        teleport_object_to_pose(env, new_obj_pose, name="object")
    
    # Let physics settle
    for _ in range(5):
        env.unwrapped.sim.step()
    
    # Sample target place position if not specified
    if target_xy is None:
        if rng is None:
            rng = np.random.default_rng()
        target_xy = sample_table_xy(rng, goal_xy=start_xy, min_dist=0.1)
    
    init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
    
    print(f"  Initial object XY: [{init_obj_pose[0]:.3f}, {init_obj_pose[1]:.3f}]")
    print(f"  Target place XY: [{target_xy[0]:.3f}, {target_xy[1]:.3f}]")
    init_dist = np.linalg.norm(init_obj_pose[:2] - np.array(target_xy))
    print(f"  Initial distance to target: {init_dist:.3f}m")
    
    # Update visual markers
    if target_markers is not None and start_markers is not None:
        update_target_markers(
            target_markers, start_markers,
            target_xy=target_xy,
            start_xy=start_xy,
            marker_z=marker_z,
            env=env,
        )
    
    steps = 0
    success = False
    last_action = None
    final_dist = None

    for t in range(horizon):
        steps = t + 1
        if t % 50 == 0:
            print(f"[Step {t+1}/{horizon}]", flush=True)

        # Get table camera RGB
        table_rgb = table_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        table_rgb_np = table_rgb.cpu().numpy().astype(np.uint8)
        table_rgb_frame = table_rgb_np[0]
        
        # Get wrist camera RGB if available
        if wrist_camera is not None:
            wrist_rgb = wrist_camera.data.output["rgb"]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]
            wrist_rgb_np = wrist_rgb.cpu().numpy().astype(np.uint8)
            wrist_rgb_frame = wrist_rgb_np[0]
            combined_frame = np.concatenate([table_rgb_frame, wrist_rgb_frame], axis=1)
        else:
            combined_frame = table_rgb_frame
        
        # Convert image for policy
        table_rgb_chw = torch.from_numpy(table_rgb_frame).float() / 255.0
        table_rgb_chw = table_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)

        # EE state
        ee_pose = get_ee_pose_w(env)[0:1]
        
        # Build observation.state
        if include_obj_pose:
            obj_pose = get_object_pose_w(env)[0:1]
            state = torch.cat([ee_pose, obj_pose], dim=-1)
        else:
            state = ee_pose

        policy_inputs: Dict[str, torch.Tensor] = {
            "observation.image": table_rgb_chw,
            "observation.state": state,
        }
        
        # Add wrist camera if needed
        if wrist_camera is not None:
            wrist_rgb_chw = torch.from_numpy(wrist_rgb_frame).float() / 255.0
            wrist_rgb_chw = wrist_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)
            policy_inputs["observation.wrist_image"] = wrist_rgb_chw

        # Store raw values for visualization
        ee_pose_raw_np = ee_pose[0].cpu().numpy()[:7]
        
        # Preprocess
        if preprocessor is not None:
            policy_inputs = preprocessor(policy_inputs)
        
        ee_pose_norm_np = policy_inputs['observation.state'][0, :7].cpu().numpy()
        
        # Check if inference step
        is_inference_step = (t % n_action_steps == 0)
        action_chunk_norm = None
        action_chunk_raw = None

        with torch.no_grad():
            action = policy.select_action(policy_inputs)
            raw_action = action.clone()
            
            # Get action chunk for visualization
            if is_inference_step and action_chunk_visualizer is not None:
                try:
                    n_obs_steps_required = policy.config.n_obs_steps if hasattr(policy, 'config') else 2
                    
                    inference_batch = {}
                    
                    if 'observation.state' in policy_inputs:
                        state_tensor = policy_inputs['observation.state']
                        if state_tensor.dim() == 2:
                            state_tensor = state_tensor.unsqueeze(1).repeat(1, n_obs_steps_required, 1)
                        inference_batch['observation.state'] = state_tensor
                    
                    if hasattr(policy, 'config') and hasattr(policy.config, 'image_features'):
                        images_list = []
                        for key in policy.config.image_features:
                            if key in policy_inputs:
                                img = policy_inputs[key]
                                if img.dim() == 4:
                                    img = img.unsqueeze(1).repeat(1, n_obs_steps_required, 1, 1, 1)
                                images_list.append(img)
                        if images_list:
                            inference_batch['observation.images'] = torch.stack(images_list, dim=2)
                    
                    if hasattr(policy, 'diffusion'):
                        full_chunk = policy.diffusion.generate_actions(inference_batch)
                        if full_chunk.dim() == 3:
                            action_chunk_norm = full_chunk[0].cpu().numpy()
                        elif full_chunk.dim() == 2:
                            action_chunk_norm = full_chunk.cpu().numpy()
                        else:
                            action_chunk_norm = full_chunk.cpu().numpy()[np.newaxis, :]
                        
                        if postprocessor is not None:
                            unnorm_actions = []
                            for i in range(action_chunk_norm.shape[0]):
                                single_action = torch.from_numpy(action_chunk_norm[i]).float().to(device)
                                unnorm_action = postprocessor(single_action)
                                unnorm_actions.append(unnorm_action.cpu().numpy())
                            action_chunk_raw = np.array(unnorm_actions)
                        else:
                            action_chunk_raw = action_chunk_norm.copy()
                            
                except Exception as e:
                    if t == 0:
                        print(f"[WARNING] Failed to get action chunk: {e}")
        
        action_norm_np = raw_action[0].cpu().numpy()
        
        # Postprocess action
        if postprocessor is not None:
            action = postprocessor(action)
        
        action_raw_np = action[0].cpu().numpy()

        # Add to XYZ visualizer
        if xyz_visualizer is not None:
            wrist_img = wrist_rgb_frame if wrist_camera is not None else None
            xyz_visualizer.add_frame(
                ee_pose_raw=ee_pose_raw_np[:3],
                ee_pose_norm=ee_pose_norm_np[:3],
                action_raw=action_raw_np[:3],
                action_norm=action_norm_np[:3],
                action_gt=None,
                table_image=table_rgb_frame,
                wrist_image=wrist_img,
                is_inference_step=is_inference_step,
            )
        
        # Add to action chunk visualizer
        if action_chunk_visualizer is not None and is_inference_step and action_chunk_norm is not None:
            wrist_img = wrist_rgb_frame if wrist_camera is not None else None
            action_chunk_visualizer.add_frame(
                ee_pose_raw=ee_pose_raw_np[:3],
                ee_pose_norm=ee_pose_norm_np[:3],
                action_chunk_norm=action_chunk_norm[:, :3],
                action_chunk_raw=action_chunk_raw[:, :3] if action_chunk_raw is not None else None,
                gt_chunk_norm=None,
                gt_chunk_raw=None,
                table_image=table_rgb_frame,
                wrist_image=wrist_img,
            )

        action = action.to(device)
        last_action = action
        
        # Add text overlay
        if writer is not None:
            ee_pose_for_text = ee_pose[0].cpu().numpy()
            obj_pose_for_text = get_object_pose_w(env)[0].cpu().numpy()
            action_for_text = action.cpu().numpy().flatten()
            
            frame_with_text = combined_frame.copy()
            ee_text = f"EE:  [{ee_pose_for_text[0]:.3f}, {ee_pose_for_text[1]:.3f}, {ee_pose_for_text[2]:.3f}]"
            obj_text = f"Obj: [{obj_pose_for_text[0]:.3f}, {obj_pose_for_text[1]:.3f}, {obj_pose_for_text[2]:.3f}]"
            act_text = f"Act: [{action_for_text[0]:.3f}, {action_for_text[1]:.3f}, {action_for_text[2]:.3f}] G:{action_for_text[-1]:.2f}"
            tgt_text = f"Tgt: [{target_xy[0]:.3f}, {target_xy[1]:.3f}]"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            (text_w, text_h), baseline = cv2.getTextSize(ee_text, font, font_scale, thickness)
            cv2.rectangle(frame_with_text, (3, 3), (6 + text_w, 6 + text_h + baseline), bg_color, -1)
            cv2.putText(frame_with_text, ee_text, (4, 4 + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
            
            y_offset1 = 8 + text_h + baseline
            cv2.rectangle(frame_with_text, (3, y_offset1), (6 + text_w, y_offset1 + 3 + text_h + baseline), bg_color, -1)
            cv2.putText(frame_with_text, obj_text, (4, y_offset1 + 1 + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
            
            y_offset2 = y_offset1 + 5 + text_h + baseline
            cv2.rectangle(frame_with_text, (3, y_offset2), (6 + text_w, y_offset2 + 3 + text_h + baseline), bg_color, -1)
            cv2.putText(frame_with_text, act_text, (4, y_offset2 + 1 + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
            
            y_offset3 = y_offset2 + 5 + text_h + baseline
            cv2.rectangle(frame_with_text, (3, y_offset3), (6 + text_w, y_offset3 + 3 + text_h + baseline), bg_color, -1)
            cv2.putText(frame_with_text, tgt_text, (4, y_offset3 + 1 + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
            
            writer.append_data(frame_with_text)

        # Step environment
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and num_envs > 1:
            action = action.repeat(num_envs, 1)

        obs_dict, _, terminated, truncated, _ = env.step(action)

        # Check success: object XY distance to target
        obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        obj_xy = obj_pose[:2]
        dist_to_target = np.linalg.norm(obj_xy - np.array(target_xy))
        final_dist = dist_to_target
        
        if terminated[0] or truncated[0]:
            break

    return {
        "steps": steps,
        "success": success,
        "final_dist": final_dist,
        "target_xy": target_xy,
        "last_action": None if last_action is None else last_action.detach().cpu().numpy(),
    }


def main() -> None:
    args = _parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import imageio

        from rev2fwd_il.utils.seed import set_seed

        set_seed(args.seed)

        device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEBUG] Using device: {device}")

        # Random generator for sampling target positions
        rng = np.random.default_rng(args.seed)

        # =====================================================================
        # Step 1: Load policy config
        # =====================================================================
        print(f"\n{'='*60}")
        print("Loading policy configuration...")
        print(f"{'='*60}")
        policy_info = load_policy_config(args.checkpoint)
        has_wrist = policy_info["has_wrist"]
        include_obj_pose = policy_info["include_obj_pose"]
        
        print(f"  Policy checkpoint: {args.checkpoint}")
        print(f"  Expected image shape: {policy_info['image_shape']} (C, H, W)")
        print(f"  Expected state dim: {policy_info['state_dim']}")
        print(f"  Expected action dim: {policy_info['action_dim']}")
        print(f"  Requires wrist camera: {has_wrist}")
        print(f"  Includes obj_pose in state: {include_obj_pose}")
        
        if policy_info["image_shape"] is not None:
            policy_h, policy_w = policy_info["image_shape"][1], policy_info["image_shape"][2]
            if policy_h != args.image_height or policy_w != args.image_width:
                print(f"\n  ⚠️  WARNING: Image size mismatch!")
                print(f"      Using policy's expected size: {policy_h}x{policy_w}")
                args.image_height = policy_h
                args.image_width = policy_w
        print(f"{'='*60}\n")

        # =====================================================================
        # Step 2: Create environment
        # =====================================================================
        env = make_env_with_camera(
            task_id=args.task,
            num_envs=1,
            device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=120.0,
            disable_terminations=False,
        )

        print("[DEBUG] Env created; checking cameras...")
        table_camera = env.unwrapped.scene.sensors.get("table_cam", None)
        if table_camera is None:
            raise RuntimeError("Camera sensor 'table_cam' not found!")
        
        wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None)
        if has_wrist and wrist_camera is None:
            raise RuntimeError("Policy expects wrist camera but wrist_cam not found!")
        
        # Create visual markers
        print("[DEBUG] Creating target markers...")
        target_markers, start_markers, marker_z = create_target_markers(
            num_envs=1, device=device
        )

        # =====================================================================
        # Step 3: Load policy
        # =====================================================================
        print("Loading diffusion policy weights...")
        policy, preprocessor, postprocessor, num_inference_steps, n_action_steps = load_diffusion_policy(
            args.checkpoint,
            device,
            image_height=args.image_height,
            image_width=args.image_width,
            num_inference_steps=args.num_inference_steps,
            n_action_steps=args.n_action_steps,
        )
        policy.eval()
        print(f"Policy loaded. (num_inference_steps={num_inference_steps}, n_action_steps={n_action_steps})")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        xyz_curves_dir = out_dir / "xyz_curves" if args.visualize_xyz else None
        if xyz_curves_dir is not None:
            xyz_curves_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # Step 4: Load overfit params if needed
        # =====================================================================
        overfit_env_init = None
        if args.overfit:
            overfit_env_init = load_overfit_env_init(args.checkpoint)

        # Start position: plate center (where cube starts in Task B)
        start_xy = tuple(args.start_xy)
        success_radius = 0.05

        print(f"\n{'='*60}")
        print(f"Task B Evaluation Settings:")
        print(f"  Start XY (cube initial): {start_xy}")
        print(f"  Target place XY: {args.target_place_xy if args.target_place_xy else 'Random per episode'}")
        print(f"  Success radius: {success_radius}m")
        print(f"  Horizon: {args.horizon}")
        print(f"  Num episodes: {args.num_episodes}")
        print(f"  Num inference steps: {num_inference_steps}")
        print(f"  N action steps: {n_action_steps}")
        print(f"  Visualize XYZ: {args.visualize_xyz}")
        print(f"  Visualize action chunk: {args.visualize_action_chunk}")
        print(f"  Overfit mode: {args.overfit}")
        print(f"{'='*60}")

        stats = []
        video_paths = []
        xyz_video_paths = []
        action_chunk_video_paths = []
        
        action_chunks_dir = out_dir / "action_chunks" if args.visualize_action_chunk else None
        if action_chunks_dir is not None:
            action_chunks_dir.mkdir(parents=True, exist_ok=True)
        
        for ep in range(args.num_episodes):
            print(f"\nEpisode {ep+1}/{args.num_episodes}")
            
            video_path = out_dir / f"ep{ep}.mp4"
            writer = imageio.get_writer(video_path, fps=args.fps)
            
            xyz_visualizer = None
            if args.visualize_xyz:
                from rev2fwd_il.data import create_eval_xyz_visualizer
                xyz_visualizer = create_eval_xyz_visualizer(
                    output_dir=xyz_curves_dir,
                    episode_id=ep,
                    fps=args.fps,
                )
            
            action_chunk_visualizer = None
            if args.visualize_action_chunk:
                from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer
                action_chunk_visualizer = ActionChunkVisualizer(
                    output_dir=action_chunks_dir,
                    step_id=ep,
                    fps=args.fps,
                )
            
            # Determine target for this episode
            target_xy = tuple(args.target_place_xy) if args.target_place_xy else None
            
            result = run_episode(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                horizon=args.horizon,
                writer=writer,
                start_xy=start_xy,
                target_xy=target_xy,
                success_radius=success_radius,
                has_wrist=has_wrist,
                include_obj_pose=include_obj_pose,
                xyz_visualizer=xyz_visualizer,
                overfit_env_init=overfit_env_init,
                n_action_steps=n_action_steps,
                rng=rng,
                target_markers=target_markers,
                start_markers=start_markers,
                marker_z=marker_z,
                action_chunk_visualizer=action_chunk_visualizer,
            )
            
            writer.close()
            video_paths.append(video_path)
            
            if xyz_visualizer is not None:
                xyz_video_path = xyz_visualizer.generate_video(filename_prefix="eval_xyz_curves")
                xyz_video_paths.append(xyz_video_path)
            
            if action_chunk_visualizer is not None:
                action_chunk_video_path = action_chunk_visualizer.generate_video(filename_prefix="eval_action_chunk")
                action_chunk_video_paths.append(action_chunk_video_path)
            
            stats.append(result)
            status = "SUCCESS" if result['success'] else "RUNNING"
            print(f"  Result: {status} | steps={result['steps']} | final_dist={result['final_dist']:.4f}m")
            print(f"  Target was: [{result['target_xy'][0]:.3f}, {result['target_xy'][1]:.3f}]")
            print(f"  Video saved: {video_path}")

        print(f"\nSaved {len(video_paths)} videos to {out_dir}/")
        if xyz_video_paths:
            print(f"Saved {len(xyz_video_paths)} XYZ curve videos to {xyz_curves_dir}/")
        if action_chunk_video_paths:
            print(f"Saved {len(action_chunk_video_paths)} action chunk videos to {action_chunks_dir}/")

        # Summary
        avg_steps = np.mean([s["steps"] for s in stats])
        avg_dist = np.mean([s["final_dist"] for s in stats])
        min_dist = min(s["final_dist"] for s in stats)
        
        print(f"\n{'='*60}")
        print(f"Task B Evaluation Summary:")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average final distance to target: {avg_dist:.4f}m")
        print(f"  Min final distance: {min_dist:.4f}m")
        print(f"{'='*60}")

        print("Closing environment...", flush=True)
        env.close()

    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()

    finally:
        print("Closing simulation app...", flush=True)
        simulation_app.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
