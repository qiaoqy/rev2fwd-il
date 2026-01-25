#!/usr/bin/env python3
"""Step 5: Evaluate and visualize the trained Diffusion Policy in Isaac Lab.

This script loads a diffusion policy trained by script 4 and evaluates it on
the forward pick-and-place task. Video is recorded from the camera feed.

=============================================================================
OVERVIEW
=============================================================================
- Input: RGB image (128x128) + EE pose (7D)
- Output: Action (8D: target EE pose + gripper)
- Task: Pick cube from random table position and place at goal

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/5_eval_diffusion.py \
    --checkpoint runs/diffusion_A_goal/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_goal/videos \
    --num_episodes 5 --visualize_xyz --headless

# With action chunk visualization
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/5_eval_diffusion.py \
    --checkpoint runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_circle/videos --horizon 500 \
    --num_episodes 10 --n_action_steps 16 --headless

CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/5_eval_diffusion.py \
    --checkpoint runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_circle/videos --horizon 500 \
    --num_episodes 10 --n_action_steps 16 --headless

=============================================================================
VISUALIZATION OPTIONS
=============================================================================
--visualize_xyz:           Per-timestep XYZ curves of EE pose and action
--visualize_action_chunk:  Per-inference visualization with input/output details

=============================================================================
NOTES
=============================================================================
- Video is recorded from camera frames (works in headless mode)
- First inference may be slow due to CUDA JIT compilation
- Isaac Sim may hang during shutdown - use Ctrl+C after video saves if needed
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
# Camera utilities (copied from script 12 to keep the same viewpoint)
# ---------------------------------------------------------------------------
def compute_camera_quat_from_lookat(eye: Tuple[float, float, float], target: Tuple[float, float, float], up: Tuple[float, float, float] = (0, 0, 1)) -> Tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion for a camera that looks at ``target`` from ``eye``.

    ROS optical frame: +Z forward, +X right, +Y down.
    """
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
    """Create visualization markers for start and goal positions.
    
    Creates two sets of flat cylinder markers on the table surface:
    - Red markers: Start positions (where cube initially is)
    - Green markers: Goal positions (fixed at plate center)
    
    These are visual-only markers with no physics interaction.
    
    Args:
        num_envs: Number of parallel environments.
        device: Torch device string.
        
    Returns:
        Tuple of (start_markers, goal_markers, marker_z) VisualizationMarkers objects.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    
    # Marker parameters
    marker_radius = 0.05  # 5cm radius
    marker_height = 0.002  # 2mm height (flat disk)
    table_z = 0.0  # Table surface height
    marker_z = table_z + marker_height / 2 + 0.001  # Slightly above table
    
    # Red marker for start positions
    start_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/StartMarkers",
        markers={
            "start": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # Red
                ),
            ),
        },
    )
    start_markers = VisualizationMarkers(start_marker_cfg)
    
    # Green marker for goal positions
    goal_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),  # Green
                ),
            ),
        },
    )
    goal_markers = VisualizationMarkers(goal_marker_cfg)
    
    return start_markers, goal_markers, marker_z


def update_target_markers(
    start_markers,
    goal_markers,
    start_xy: tuple,
    goal_xy: tuple,
    marker_z: float,
    env,
):
    """Update the positions of start and goal markers.
    
    Args:
        start_markers: VisualizationMarkers for start positions.
        goal_markers: VisualizationMarkers for goal positions.
        start_xy: Tuple (x, y) for the start position (initial object position).
        goal_xy: Tuple (x, y) for the goal position (plate center).
        marker_z: Z height for markers.
        env: Isaac Lab environment (for env_origins).
    """
    import torch
    
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    env_origins = env.unwrapped.scene.env_origins  # (num_envs, 3)
    
    # Convert to Python float to handle numpy.float32 types
    start_x = float(start_xy[0])
    start_y = float(start_xy[1])
    goal_x = float(goal_xy[0])
    goal_y = float(goal_xy[1])
    
    # Build start marker positions (same for all envs in this case)
    start_positions = torch.zeros((num_envs, 3), device=device)
    start_positions[:, 0] = start_x
    start_positions[:, 1] = start_y
    start_positions[:, 2] = marker_z
    # Add env origins for world positions
    start_positions_w = start_positions + env_origins
    
    # Build goal marker positions (same XY for all envs)
    goal_positions = torch.zeros((num_envs, 3), device=device)
    goal_positions[:, 0] = goal_x
    goal_positions[:, 1] = goal_y
    goal_positions[:, 2] = marker_z
    # Add env origins for world positions
    goal_positions_w = goal_positions + env_origins
    
    # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
    
    # Update marker visualizations
    start_markers.visualize(start_positions_w, identity_quat)
    goal_markers.visualize(goal_positions_w, identity_quat)


def add_camera_to_env_cfg(env_cfg, image_width: int, image_height: int) -> None:
    """Inject table-view and wrist cameras into Isaac Lab env cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    # =========================================================================
    # Table Camera - Third-person fixed view
    # =========================================================================
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
    
    # =========================================================================
    # Wrist Camera - Eye-in-hand, attached to robot gripper
    # =========================================================================
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
        description="Test diffusion policy A in Isaac Lab and record video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model",
        help="Path to LeRobot pretrained_model directory (contains config.json, model.safetensors, preprocessors).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/diffusion_A/videos",
        help="Output directory for recorded videos. Each episode saves as ep0.mp4, ep1.mp4, etc.",
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
        help="How many episodes to roll out (records first one).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1000,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for MP4 writer (matches dataset FPS).",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="Camera width (must match training data).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=128,
        help="Camera height (must match training data).",
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
        help="Random seed for torch/numpy.",
    )
    parser.add_argument(
        "--min_init_dist",
        type=float,
        default=0.15,
        help="Minimum initial distance from goal to accept (reject if closer). Default: 0.15m.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of diffusion denoising steps at inference. Must be <= num_train_timesteps (default 100). "
             "Higher = more stable but slower. Default: None (use all training timesteps, i.e., 100). "
             "For faster inference, try 50. Cannot exceed training timesteps.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Number of action steps to execute before re-inferring. "
             "If None, uses the value from training config. "
             "Higher = execute more steps from each inference (smoother but less reactive). "
             "Must be <= horizon. Default: None (use training config, typically 8).",
    )
    parser.add_argument(
        "--visualize_xyz",
        action="store_true",
        help="Generate XYZ curve visualization videos for debugging. "
             "Saves to {out_dir}/xyz_curves/ with camera images side by side.",
    )
    parser.add_argument(
        "--visualize_action_chunk",
        action="store_true",
        help="Generate action chunk visualization videos showing model input and predicted 16-step chunk. "
             "Saves to {out_dir}/action_chunks/ with one frame per inference step.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Enable overfit testing mode: initialize environment with saved initial poses "
             "from training. Reads overfit_env_init.json from checkpoint directory.",
    )
    parser.add_argument(
        "--init_obj_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Custom initial XY position for the object (e.g., --init_obj_xy 0.3 0.1). "
             "If not specified, object position is randomized. Z is automatically set to table height.",
    )

    # Isaac Lab AppLauncher flags (headless, device, etc.)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True  # required for headless camera rendering
    return args


# ---------------------------------------------------------------------------
# Overfit env init loading
# ---------------------------------------------------------------------------
def load_overfit_env_init(checkpoint_dir: str) -> Dict[str, Any] | None:
    """Load overfit environment initialization parameters from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory (e.g., runs/diffusion_A/checkpoints).
        
    Returns:
        Dictionary with initial poses, or None if not found.
        Keys: initial_obj_pose, initial_ee_pose, place_pose, goal_pose
    """
    import json
    from pathlib import Path
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Try multiple possible locations for overfit_env_init.json
    possible_paths = [
        checkpoint_path / "overfit_env_init.json",  # Direct in checkpoint dir
        checkpoint_path.parent / "overfit_env_init.json",  # Parent dir
        checkpoint_path.parent.parent  / "overfit_env_init.json",  # runs/xxx/checkpoints/
        checkpoint_path.parent.parent.parent  / "overfit_env_init.json", 
        checkpoint_path.parent.parent.parent.parent  / "overfit_env_init.json",
    ]
    
    for init_path in possible_paths:
        if init_path.exists():
            print(f"[Overfit Mode] Loading env init params from: {init_path}")
            with open(init_path, "r") as f:
                overfit_env_init = json.load(f)
            print(f"  initial_obj_pose: {overfit_env_init['initial_obj_pose']}")
            print(f"  initial_ee_pose: {overfit_env_init['initial_ee_pose']}")
            return overfit_env_init
    
    print(f"[Overfit Mode] WARNING: overfit_env_init.json not found in checkpoint directory!")
    print(f"  Searched paths: {[str(p) for p in possible_paths]}")
    return None


# ---------------------------------------------------------------------------
# Policy loading helpers
# ---------------------------------------------------------------------------
def load_policy_config(pretrained_dir: str) -> Dict[str, Any]:
    """Load policy configuration without loading model weights.
    
    This is useful for checking policy requirements (e.g., image size, 
    whether wrist camera is needed) before creating the simulation environment.
    
    Args:
        pretrained_dir: Path to the pretrained model directory.
        
    Returns:
        Dictionary with policy configuration including:
        - has_wrist: Whether policy expects wrist camera input
        - image_shape: Expected image shape (C, H, W)
        - state_dim: Expected state dimension
        - action_dim: Expected action dimension
    """
    import json
    from pathlib import Path
    
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
    
    # Check if obj_pose and gripper are included based on state_dim:
    # state_dim=7:  ee_pose(7) only
    # state_dim=8:  ee_pose(7) + gripper(1)
    # state_dim=14: ee_pose(7) + obj_pose(7)
    # state_dim=15: ee_pose(7) + obj_pose(7) + gripper(1)
    include_obj_pose = (state_dim in [14, 15]) if state_dim is not None else False
    include_gripper = (state_dim in [8, 15]) if state_dim is not None else False
    
    return {
        "has_wrist": has_wrist,
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "include_obj_pose": include_obj_pose,
        "include_gripper": include_gripper,
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
    """Load LeRobot diffusion policy from checkpoint directory.
    
    Args:
        pretrained_dir: Path to the pretrained model directory.
        device: Device to load the model on.
        image_height: Image height (must match training data).
        image_width: Image width (must match training data).
        num_inference_steps: Number of diffusion denoising steps at inference.
            If None, uses the value from training config (typically num_train_timesteps).
            Higher values = more stable predictions but slower inference.
        
    Returns:
        Tuple of (policy, preprocessor, postprocessor).
    """
    import json
    import os
    from pathlib import Path
    
    print(f"[load_policy] dir={pretrained_dir}, device={device}", flush=True)
    print(f"[load_policy] files: {os.listdir(pretrained_dir)}", flush=True)

    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    pretrained_path = Path(pretrained_dir)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"
    
    # Load config.json
    print(f"[load_policy] Loading config.json...", flush=True)
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
    
    # Parse normalization_mapping (keys are strings in lerobot 0.4.2)
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
        # Validate: num_inference_steps cannot exceed num_train_timesteps
        num_train_timesteps = config_dict.get("num_train_timesteps", 100)
        if num_inference_steps > num_train_timesteps:
            print(f"[load_policy] WARNING: num_inference_steps ({num_inference_steps}) cannot exceed "
                  f"num_train_timesteps ({num_train_timesteps}). Clamping to {num_train_timesteps}.", flush=True)
            num_inference_steps = num_train_timesteps
        config_dict["num_inference_steps"] = num_inference_steps
        print(f"[load_policy] Setting num_inference_steps to {num_inference_steps}", flush=True)
    
    # Override n_action_steps if specified
    if n_action_steps is not None:
        horizon = config_dict.get("horizon", 16)
        if n_action_steps > horizon:
            print(f"[load_policy] WARNING: n_action_steps ({n_action_steps}) cannot exceed "
                  f"horizon ({horizon}). Clamping to {horizon}.", flush=True)
            n_action_steps = horizon
        config_dict["n_action_steps"] = n_action_steps
        print(f"[load_policy] Setting n_action_steps to {n_action_steps}", flush=True)
    
    # Create DiffusionConfig
    print(f"[load_policy] Creating DiffusionConfig...", flush=True)
    cfg = DiffusionConfig(**config_dict)
    print(f"[load_policy] DiffusionConfig created (num_inference_steps={cfg.num_inference_steps})", flush=True)

    # Create policy model
    print(f"[load_policy] Creating DiffusionPolicy model...", flush=True)
    t0 = time.time()
    policy = DiffusionPolicy(cfg)
    print(f"[load_policy] DiffusionPolicy created in {time.time()-t0:.2f}s", flush=True)
    
    # Load weights
    print(f"[load_policy] Loading model weights...", flush=True)
    t0 = time.time()
    state_dict = load_file(model_path)
    print(f"[load_policy] Weights file loaded in {time.time()-t0:.2f}s, keys={len(state_dict)}", flush=True)
    
    print(f"[load_policy] Calling load_state_dict...", flush=True)
    t0 = time.time()
    policy.load_state_dict(state_dict)
    print(f"[load_policy] load_state_dict done in {time.time()-t0:.2f}s", flush=True)
    
    # Move to device
    print(f"[load_policy] Moving to device {device}...", flush=True)
    t0 = time.time()
    policy = policy.to(device)
    print(f"[load_policy] Moved to device in {time.time()-t0:.2f}s", flush=True)
    
    # Load preprocessor and postprocessor from saved files
    # These handle normalization (input) and unnormalization (output)
    print(f"[load_policy] Loading preprocessor and postprocessor...", flush=True)
    t0 = time.time()
    
    # Override device for preprocessor to match inference device
    preprocessor_overrides = {
        "device_processor": {"device": device},
    }
    postprocessor_overrides = {
        "device_processor": {"device": device},
    }
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    print(f"[load_policy] Processors loaded in {time.time()-t0:.2f}s", flush=True)
    
    # Get actual num_inference_steps used by the policy
    # It may be stored in diffusion.num_inference_steps or we compute from config
    actual_inference_steps = cfg.num_inference_steps
    if actual_inference_steps is None:
        actual_inference_steps = cfg.num_train_timesteps
    
    # =========================================================================
    # DEBUG: Print normalization settings for inference
    # =========================================================================
    print("\n" + "=" * 60)
    print("[DEBUG] INFERENCE Normalization Settings")
    print("=" * 60)
    print(f"  policy_cfg.normalization_mapping:")
    for feat_type, norm_mode in cfg.normalization_mapping.items():
        print(f"    {feat_type}: {norm_mode}")
    print("\n  Preprocessor steps:")
    for i, step in enumerate(preprocessor.steps):
        step_name = getattr(step, '__class__', type(step)).__name__
        print(f"    [{i}] {step_name}")
        # Print normalizer details if it's a normalizer step
        if hasattr(step, 'norm_map'):
            print(f"        norm_map: {step.norm_map}")
        if hasattr(step, 'features'):
            print(f"        features: {list(step.features.keys())}")
    print("\n  Postprocessor steps:")
    for i, step in enumerate(postprocessor.steps):
        step_name = getattr(step, '__class__', type(step)).__name__
        print(f"    [{i}] {step_name}")
        # Print unnormalizer details if it's an unnormalizer step
        if hasattr(step, 'norm_map'):
            print(f"        norm_map: {step.norm_map}")
        if hasattr(step, 'features'):
            print(f"        features: {list(step.features.keys())}")
    print("=" * 60 + "\n")
    
    # Get actual n_action_steps used by the policy
    actual_n_action_steps = cfg.n_action_steps
    
    print(f"[load_policy] Policy loading complete! (num_inference_steps={actual_inference_steps}, n_action_steps={actual_n_action_steps})", flush=True)
    return policy, preprocessor, postprocessor, actual_inference_steps, actual_n_action_steps


def run_episode(
    env,
    policy,
    preprocessor,
    postprocessor,
    horizon: int,
    writer,
    goal_xy: tuple = (0.5, 0.0),
    success_radius: float = 0.05,
    min_init_dist: float = 0.08,
    max_reset_attempts: int = 20,
    has_wrist: bool = False,
    include_obj_pose: bool = False,
    include_gripper: bool = False,
    xyz_visualizer=None,
    overfit_env_init: dict | None = None,
    n_action_steps: int = 8,
    init_obj_xy: tuple | None = None,
    start_markers=None,
    goal_markers=None,
    marker_z: float = 0.002,
    action_chunk_visualizer=None,
) -> dict:
    """Run one episode, write frames to writer if provided, return summary.
    
    Args:
        env: Isaac Lab environment.
        policy: Diffusion policy.
        preprocessor: Preprocessor pipeline for input normalization.
        postprocessor: Postprocessor pipeline for action unnormalization.
        horizon: Maximum steps per episode.
        writer: Video writer (or None).
        goal_xy: Goal XY position (plate center).
        success_radius: Success radius in meters.
        min_init_dist: Minimum initial distance from goal (reject if closer).
        max_reset_attempts: Max attempts to find valid initial position.
        has_wrist: Whether the policy expects wrist camera input.
        include_obj_pose: Whether the policy expects object pose in state.
        include_gripper: Whether the policy expects gripper state in state.
        xyz_visualizer: Optional XYZCurveVisualizer for debugging.
        overfit_env_init: Optional dict with initial poses for overfit testing.
            If provided, teleports object to saved initial pose after reset.
        n_action_steps: Number of action steps per inference. Used to mark
            action chunk boundaries in XYZ visualization.
        init_obj_xy: Optional tuple (x, y) for custom initial object position.
            If provided, teleports object to this XY position after reset.
            Takes precedence over random initialization but not over overfit_env_init.
        start_markers: Optional VisualizationMarkers for start position (red circle).
        goal_markers: Optional VisualizationMarkers for goal position (green circle).
        marker_z: Z height for markers.
        action_chunk_visualizer: Optional ActionChunkVisualizer for visualizing
            model input and predicted action chunks at each inference step.
        
    Returns:
        Dictionary with episode statistics.
    """

    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose

    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None) if has_wrist else None
    device = env.unwrapped.device

    # Reset policy action queue for new episode
    # This is CRITICAL for proper action chunking behavior!
    # Without this, the action queue may contain stale actions from previous episodes.
    policy.reset()

    # Reset environment
    obs_dict, _ = env.reset()
    
    # If overfit mode, teleport object to saved initial pose
    if overfit_env_init is not None:
        print(f"  [Overfit Mode] Teleporting object to saved initial pose...")
        initial_obj_pose = torch.tensor(
            overfit_env_init["initial_obj_pose"], 
            dtype=torch.float32, 
            device=device
        ).unsqueeze(0)  # (1, 7)
        teleport_object_to_pose(env, initial_obj_pose, name="object")
        
        # Step simulation a few times to let physics settle
        for _ in range(5):
            env.unwrapped.sim.step()
        
        init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        init_dist = np.linalg.norm(init_obj_pose[:2] - np.array(goal_xy))
        print(f"  [Overfit Mode] Object teleported to: [{init_obj_pose[0]:.3f}, {init_obj_pose[1]:.3f}, {init_obj_pose[2]:.3f}]")
    elif init_obj_xy is not None:
        # Custom initial XY position specified by user
        print(f"  [Custom Init] Teleporting object to user-specified XY: [{init_obj_xy[0]:.3f}, {init_obj_xy[1]:.3f}]...")
        # Get current object pose to preserve orientation
        current_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        # Build new pose: user XY, fixed Z slightly above table, keep quaternion from current pose
        init_obj_z = 0.0022  # Slightly above table surface
        new_obj_pose = torch.tensor(
            [init_obj_xy[0], init_obj_xy[1], init_obj_z,  # XYZ
             current_obj_pose[3], current_obj_pose[4], current_obj_pose[5], current_obj_pose[6]],  # quaternion
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)  # (1, 7)
        teleport_object_to_pose(env, new_obj_pose, name="object")
        
        # Step simulation a few times to let physics settle
        for _ in range(5):
            env.unwrapped.sim.step()
        
        init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        init_dist = np.linalg.norm(init_obj_pose[:2] - np.array(goal_xy))
        print(f"  [Custom Init] Object teleported to: [{init_obj_pose[0]:.3f}, {init_obj_pose[1]:.3f}, {init_obj_pose[2]:.3f}]")
    else:
        # Normal mode: Reset until object is far enough from goal
        goal_xy_arr = np.array(goal_xy)
        for attempt in range(max_reset_attempts):
            if attempt > 0:
                obs_dict, _ = env.reset()
            init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
            init_obj_xy = init_obj_pose[:2]
            init_dist = np.linalg.norm(init_obj_xy - goal_xy_arr)
            
            if init_dist >= min_init_dist:
                break
            
            if attempt < max_reset_attempts - 1:
                print(f"  Reset attempt {attempt+1}: object too close to goal "
                      f"(dist={init_dist:.3f}m < {min_init_dist}m), retrying...")
    
    print(f"  Initial object XY: [{init_obj_pose[0]:.3f}, {init_obj_pose[1]:.3f}]")
    print(f"  Goal XY: [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}]")
    print(f"  Initial distance to goal: {init_dist:.3f}m")
    
    # Update visual markers for start and goal positions
    if start_markers is not None and goal_markers is not None:
        update_target_markers(
            start_markers, goal_markers,
            start_xy=(init_obj_pose[0], init_obj_pose[1]),
            goal_xy=goal_xy,
            marker_z=marker_z,
            env=env,
        )
    
    steps = 0
    success = False
    success_step = None  # Step at which success was first achieved
    last_action = None
    final_dist = None
    
    # Track gripper state (initialized to open, will be updated from actions)
    # Gripper: +1 = open, -1 = close
    current_gripper_state = 1.0


    for t in range(horizon):
        steps = t + 1
        if t % 50 == 0:
            print(f"[Step {t+1}/{horizon}]", flush=True)

        # Acquire table camera RGB (num_envs, H, W, 3) -> float32 BCHW
        table_rgb = table_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        table_rgb_np = table_rgb.cpu().numpy().astype(np.uint8)
        table_rgb_frame = table_rgb_np[0]
        
        # Acquire wrist camera RGB if available
        if wrist_camera is not None:
            wrist_rgb = wrist_camera.data.output["rgb"]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]
            wrist_rgb_np = wrist_rgb.cpu().numpy().astype(np.uint8)
            wrist_rgb_frame = wrist_rgb_np[0]
            # Concatenate side-by-side for video
            combined_frame = np.concatenate([table_rgb_frame, wrist_rgb_frame], axis=1)
        else:
            combined_frame = table_rgb_frame
        
        # Convert table image to float32 [0, 1] and BCHW format for policy
        table_rgb_chw = torch.from_numpy(table_rgb_frame).float() / 255.0  # uint8 -> float [0,1]
        table_rgb_chw = table_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
        
        # DEBUG: Print image stats before preprocessing (first step only)
        if t == 0:
            print(f"[DEBUG] Step 0: Image BEFORE preprocessing:")
            print(f"  Range: [{table_rgb_chw.min():.4f}, {table_rgb_chw.max():.4f}]")
            print(f"  Mean per channel: R={table_rgb_chw[0,0].mean():.4f}, G={table_rgb_chw[0,1].mean():.4f}, B={table_rgb_chw[0,2].mean():.4f}")

        # EE state
        ee_pose = get_ee_pose_w(env)[0:1]
        
        # Build observation.state based on include_obj_pose and include_gripper:
        # state_dim=7:  ee_pose(7) only
        # state_dim=8:  ee_pose(7) + gripper(1)
        # state_dim=14: ee_pose(7) + obj_pose(7)
        # state_dim=15: ee_pose(7) + obj_pose(7) + gripper(1)
        state_parts = [ee_pose]
        if include_obj_pose:
            obj_pose = get_object_pose_w(env)[0:1]  # (1, 7)
            state_parts.append(obj_pose)
        if include_gripper:
            gripper_tensor = torch.tensor([[current_gripper_state]], dtype=torch.float32, device=device)
            state_parts.append(gripper_tensor)
        state = torch.cat(state_parts, dim=-1)

        policy_inputs: Dict[str, torch.Tensor] = {
            "observation.image": table_rgb_chw,
            "observation.state": state,
        }
        
        # Add wrist camera input if policy expects it
        if wrist_camera is not None:
            wrist_rgb_chw = torch.from_numpy(wrist_rgb_frame).float() / 255.0
            wrist_rgb_chw = wrist_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)
            policy_inputs["observation.wrist_image"] = wrist_rgb_chw

        # Preprocess inputs (normalizes state and images with ImageNet mean/std)
        # DEBUG: Print before/after normalization on first step
        if t == 0 and preprocessor is not None:
            print("\n[DEBUG] Step 0: Before preprocessing (raw inputs):")
            print(f"  observation.state: {policy_inputs['observation.state'][0, :7].cpu().numpy()}")
            print(f"  observation.image range: [{policy_inputs['observation.image'].min():.4f}, {policy_inputs['observation.image'].max():.4f}]")
            print(f"  observation.image mean per channel: R={policy_inputs['observation.image'][0,0].mean():.4f}, G={policy_inputs['observation.image'][0,1].mean():.4f}, B={policy_inputs['observation.image'][0,2].mean():.4f}")
        
        # Store raw EE pose before normalization for XYZ visualization
        ee_pose_raw_np = ee_pose[0].cpu().numpy()[:7]
        
        if preprocessor is not None:
            policy_inputs = preprocessor(policy_inputs)
        
        # Store normalized EE pose for XYZ visualization
        ee_pose_norm_np = policy_inputs['observation.state'][0, :7].cpu().numpy()
        
        if t == 0 and preprocessor is not None:
            print("[DEBUG] Step 0: After preprocessing (ImageNet normalized):")
            print(f"  observation.state: {policy_inputs['observation.state'][0, :7].cpu().numpy()}")
            print(f"  observation.image range: [{policy_inputs['observation.image'].min():.4f}, {policy_inputs['observation.image'].max():.4f}]")
            print(f"  observation.image mean per channel: R={policy_inputs['observation.image'][0,0].mean():.4f}, G={policy_inputs['observation.image'][0,1].mean():.4f}, B={policy_inputs['observation.image'][0,2].mean():.4f}")
            print(f"  Expected after ImageNet norm: mean~0, std~1 per channel")

        with torch.no_grad():
            action = policy.select_action(policy_inputs)
            raw_actioin = action.clone()
            
            # Get full action chunk for visualization at inference steps
            # This is when t % n_action_steps == 0, meaning policy does new inference
            is_inference_step = (t % n_action_steps == 0)
            action_chunk_norm = None
            action_chunk_raw = None
            
            if is_inference_step and action_chunk_visualizer is not None:
                # Get full action chunk by calling diffusion.generate_actions directly
                # This bypasses the action queue and gives us all predicted actions
                try:
                    # For diffusion policy, we need to prepare the batch correctly
                    # The diffusion module expects:
                    # - observation.state: (B, n_obs_steps, state_dim)
                    # - observation.images: (B, n_obs_steps, num_cams, C, H, W)
                    
                    # Get the required n_obs_steps from policy config
                    n_obs_steps_required = policy.config.n_obs_steps if hasattr(policy, 'config') else 2
                    
                    if t == 0:
                        print(f"[DEBUG] Action chunk viz: n_obs_steps_required={n_obs_steps_required}")
                    
                    inference_batch = {}
                    
                    # Copy state - replicate to fill n_obs_steps dimension
                    if 'observation.state' in policy_inputs:
                        state = policy_inputs['observation.state']  # (B, state_dim)
                        # (B, state_dim) -> (B, n_obs_steps, state_dim)
                        if state.dim() == 2:
                            state = state.unsqueeze(1).repeat(1, n_obs_steps_required, 1)
                        inference_batch['observation.state'] = state
                    
                    # Stack image features into observation.images
                    if hasattr(policy, 'config') and hasattr(policy.config, 'image_features'):
                        images_list = []
                        for key in policy.config.image_features:
                            if key in policy_inputs:
                                img = policy_inputs[key]  # (B, C, H, W)
                                # (B, C, H, W) -> (B, n_obs_steps, C, H, W)
                                if img.dim() == 4:
                                    img = img.unsqueeze(1).repeat(1, n_obs_steps_required, 1, 1, 1)
                                images_list.append(img)
                        if images_list:
                            # Stack along camera dimension (dim=2)
                            # Result: (B, n_obs_steps, num_cams, C, H, W)
                            inference_batch['observation.images'] = torch.stack(images_list, dim=2)
                            
                            if t == 0:
                                print(f"[DEBUG] Action chunk viz: observation.images shape={inference_batch['observation.images'].shape}")
                                print(f"[DEBUG] Action chunk viz: observation.state shape={inference_batch['observation.state'].shape}")
                    
                    # Call diffusion.generate_actions to get full chunk
                    if hasattr(policy, 'diffusion'):
                        full_chunk = policy.diffusion.generate_actions(inference_batch)
                        # full_chunk shape: (B, horizon, action_dim)
                        if full_chunk.dim() == 3:
                            action_chunk_norm = full_chunk[0].cpu().numpy()  # (horizon, action_dim)
                        elif full_chunk.dim() == 2:
                            action_chunk_norm = full_chunk.cpu().numpy()
                        else:
                            action_chunk_norm = full_chunk.cpu().numpy()[np.newaxis, :]
                        
                        if t == 0:
                            print(f"[DEBUG] Action chunk viz: got action_chunk_norm shape={action_chunk_norm.shape}")
                        
                        # Unnormalize each action in the chunk
                        if postprocessor is not None:
                            unnorm_actions = []
                            for i in range(action_chunk_norm.shape[0]):
                                single_action = torch.from_numpy(action_chunk_norm[i]).float().to(device)
                                unnorm_action = postprocessor(single_action)
                                unnorm_actions.append(unnorm_action.cpu().numpy())
                            action_chunk_raw = np.array(unnorm_actions)  # (horizon, action_dim)
                        else:
                            action_chunk_raw = action_chunk_norm.copy()
                    else:
                        if t == 0:
                            print(f"[WARNING] Policy does not have 'diffusion' attribute, cannot get action chunk")
                            
                except Exception as e:
                    import traceback
                    print(f"[WARNING] Failed to get action chunk for visualization: {e}")
                    if t == 0:
                        traceback.print_exc()
        
        # Store normalized action for XYZ visualization
        action_norm_np = raw_actioin[0].cpu().numpy()
        
        # DEBUG: Print raw action (in normalized space) on first step
        if t == 0:
            print(f"[DEBUG] Step 0: Raw action (normalized, from policy): {raw_actioin[0].cpu().numpy()}")

        # Postprocess action (unnormalizes from [-1, 1] to original range)
        # postprocessor expects a Tensor (PolicyAction) and returns a Tensor
        if postprocessor is not None:
            action = postprocessor(action)
        
        # Store unnormalized action for XYZ visualization
        action_raw_np = action[0].cpu().numpy()
        
        # DEBUG: Print after unnormalization on first step
        if t == 0:
            print(f"[DEBUG] Step 0: Unnormalized action: {action[0].cpu().numpy()}\n")

        # Add frame to XYZ visualizer if provided
        if xyz_visualizer is not None:
            # Get wrist image if available
            wrist_img = wrist_rgb_frame if wrist_camera is not None else None
            # Determine if this is an inference step (start of new action chunk)
            # New inference happens at t=0, n_action_steps, 2*n_action_steps, etc.
            xyz_visualizer.add_frame(
                ee_pose_raw=ee_pose_raw_np[:3],  # XYZ only
                ee_pose_norm=ee_pose_norm_np[:3],  # XYZ only
                action_raw=action_raw_np[:3],  # XYZ only
                action_norm=action_norm_np[:3],  # XYZ only
                action_gt=None,  # No ground truth during evaluation
                table_image=table_rgb_frame,  # Camera image for visualization
                wrist_image=wrist_img,  # Wrist camera image if available
                is_inference_step=is_inference_step,  # Mark action chunk boundaries
            )
        
        # Add frame to action chunk visualizer at inference steps
        if action_chunk_visualizer is not None and is_inference_step and action_chunk_norm is not None:
            wrist_img = wrist_rgb_frame if wrist_camera is not None else None
            action_chunk_visualizer.add_frame(
                ee_pose_raw=ee_pose_raw_np[:3],  # XYZ only
                ee_pose_norm=ee_pose_norm_np[:3],  # XYZ only
                action_chunk_norm=action_chunk_norm[:, :3],  # XYZ only (horizon, 3)
                action_chunk_raw=action_chunk_raw[:, :3] if action_chunk_raw is not None else None,
                gt_chunk_norm=None,  # No ground truth during evaluation
                gt_chunk_raw=None,
                table_image=table_rgb_frame,
                wrist_image=wrist_img,
            )

        action = action.to(device)
        last_action = action
        
        # Get poses and action for text overlay
        ee_pose_for_text = ee_pose[0].cpu().numpy()
        obj_pose_for_text = get_object_pose_w(env)[0].cpu().numpy()
        action_for_text = action.cpu().numpy().flatten()
        
        # Add text overlay with EE, object, and action XYZ coordinates + gripper
        if writer is not None:
            frame_with_text = combined_frame.copy()
            ee_text = f"EE:  [{ee_pose_for_text[0]:.3f}, {ee_pose_for_text[1]:.3f}, {ee_pose_for_text[2]:.3f}]"
            obj_text = f"Obj: [{obj_pose_for_text[0]:.3f}, {obj_pose_for_text[1]:.3f}, {obj_pose_for_text[2]:.3f}]"
            act_text = f"Act: [{action_for_text[0]:.3f}, {action_for_text[1]:.3f}, {action_for_text[2]:.3f}] G:{action_for_text[-1]:.2f}"
            raw_actioin_for_text = f"RawAct: [{raw_actioin[0,0]:.3f}, {raw_actioin[0,1]:.3f}, {raw_actioin[0,2]:.3f}] G:{raw_actioin[0,-1]:.2f}"
            # Text parameters (smaller font)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background
            
            # Draw EE text with background
            (text_w, text_h), baseline = cv2.getTextSize(ee_text, font, font_scale, thickness)
            cv2.rectangle(frame_with_text, (3, 3), (6 + text_w, 6 + text_h + baseline), bg_color, -1)
            cv2.putText(frame_with_text, ee_text, (4, 4 + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
            
            # Draw Obj text with background (below EE text)
            y_offset1 = 8 + text_h + baseline
            (text_w2, text_h2), baseline2 = cv2.getTextSize(obj_text, font, font_scale, thickness)
            cv2.rectangle(frame_with_text, (3, y_offset1), (6 + text_w2, y_offset1 + 3 + text_h2 + baseline2), bg_color, -1)
            cv2.putText(frame_with_text, obj_text, (4, y_offset1 + 1 + text_h2), font, font_scale, color, thickness, cv2.LINE_AA)
            
            # Draw Act text with background (below Obj text)
            y_offset2 = y_offset1 + 5 + text_h2 + baseline2
            (text_w3, text_h3), baseline3 = cv2.getTextSize(act_text, font, font_scale, thickness)
            cv2.rectangle(frame_with_text, (3, y_offset2), (6 + text_w3, y_offset2 + 3 + text_h3 + baseline3), bg_color, -1)
            cv2.putText(frame_with_text, act_text, (4, y_offset2 + 1 + text_h3), font, font_scale, color, thickness, cv2.LINE_AA)

            # Draw RawAct text with background (below Act text)
            y_offset3 = y_offset2 + 5 + text_h3 + baseline3
            (text_w4, text_h4), baseline4 = cv2.getTextSize(raw_actioin_for_text, font, font_scale, thickness)
            cv2.rectangle(frame_with_text, (3, y_offset3), (6 + text_w4, y_offset3 + 3 + text_h4 + baseline4), bg_color, -1)
            cv2.putText(frame_with_text, raw_actioin_for_text, (4, y_offset3 + 1 + text_h4), font, font_scale, color, thickness, cv2.LINE_AA)
            
            writer.append_data(frame_with_text)

        # Tile action for num_envs (vectorized env)
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and num_envs > 1:
            action = action.repeat(num_envs, 1)

        obs_dict, _, terminated, truncated, _ = env.step(action)
        
        # Update gripper state from the action (last element)
        # The action is (8,): [pos(3), quat(4), gripper(1)]
        # Gripper: +1 = open, -1 = close
        if include_gripper:
            current_gripper_state = action_for_text[-1]

        # Check success: object XY distance to goal
        obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        obj_xy = obj_pose[:2]
        dist_to_goal = np.linalg.norm(obj_xy - np.array(goal_xy))
        final_dist = dist_to_goal
        
        # Check if success achieved (first time only)
        if not success and dist_to_goal < success_radius:
            success = True
            success_step = t + 1
            print(f"  ✓ SUCCESS at step {t+1}! Distance: {dist_to_goal:.4f}m (continuing to complete horizon for debug)")
        
        # Continue execution even after success to complete full horizon for debugging
        # Only break on early termination from environment
        if terminated[0] or truncated[0]:
            break

    return {
        "steps": steps,
        "success": success,
        "success_step": success_step,  # Step at which success was first achieved (None if failed)
        "final_dist": final_dist,
        "last_action": None if last_action is None else last_action.detach().cpu().numpy(),
    }


def main() -> None:
    args = _parse_args()

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Launch Isaac Sim first
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import imageio

        from rev2fwd_il.utils.seed import set_seed

        set_seed(args.seed)

        # AppLauncher injects args.device; use it for both sim and policy
        device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[DEBUG] Using device: {device}")

        # =====================================================================
        # Step 1: Load policy config FIRST to check requirements
        # =====================================================================
        print(f"\n{'='*60}")
        print("Loading policy configuration...")
        print(f"{'='*60}")
        policy_info = load_policy_config(args.checkpoint)
        has_wrist = policy_info["has_wrist"]
        include_obj_pose = policy_info["include_obj_pose"]
        include_gripper = policy_info["include_gripper"]
        
        print(f"  Policy checkpoint: {args.checkpoint}")
        print(f"  Expected image shape: {policy_info['image_shape']} (C, H, W)")
        print(f"  Expected state dim: {policy_info['state_dim']}")
        print(f"  Expected action dim: {policy_info['action_dim']}")
        print(f"  Requires wrist camera: {has_wrist}")
        print(f"  Includes obj_pose in state: {include_obj_pose}")
        print(f"  Includes gripper in state: {policy_info['include_gripper']}")
        
        # Validate image dimensions match
        if policy_info["image_shape"] is not None:
            policy_h, policy_w = policy_info["image_shape"][1], policy_info["image_shape"][2]
            if policy_h != args.image_height or policy_w != args.image_width:
                print(f"\n  ⚠️  WARNING: Image size mismatch!")
                print(f"      Policy expects: {policy_h}x{policy_w}")
                print(f"      Args specify:   {args.image_height}x{args.image_width}")
                print(f"      Using policy's expected size: {policy_h}x{policy_w}")
                args.image_height = policy_h
                args.image_width = policy_w
        print(f"{'='*60}\n")

        # =====================================================================
        # Step 2: Create environment with appropriate cameras
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
            raise RuntimeError("Camera sensor 'table_cam' not found; did add_camera_to_env_cfg run?")
        print(f"[DEBUG] Table camera cfg: {table_camera.cfg.width}x{table_camera.cfg.height}, data_types={table_camera.cfg.data_types}")
        
        wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None)
        if wrist_camera is not None:
            print(f"[DEBUG] Wrist camera cfg: {wrist_camera.cfg.width}x{wrist_camera.cfg.height}, data_types={wrist_camera.cfg.data_types}")
        else:
            print("[DEBUG] Wrist camera not found in env")
        
        # Validate: if policy needs wrist camera, env must have it
        if has_wrist and wrist_camera is None:
            raise RuntimeError(
                "Policy expects wrist camera input (observation.wrist_image) "
                "but wrist_cam sensor not found in environment!"
            )
        
        # =====================================================================
        # Create visual markers for start and goal positions
        # =====================================================================
        print("[DEBUG] Creating target markers...")
        start_markers, goal_markers, marker_z = create_target_markers(
            num_envs=1, device=device
        )
        print("[DEBUG] Target markers created")

        # =====================================================================
        # Step 3: Load full policy model
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
        
        # Create XYZ curves output directory if visualization is enabled
        xyz_curves_dir = out_dir / "xyz_curves" if args.visualize_xyz else None
        if xyz_curves_dir is not None:
            xyz_curves_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # Step 4: Load overfit env init params if in overfit mode
        # =====================================================================
        overfit_env_init = None
        if args.overfit:
            overfit_env_init = load_overfit_env_init(args.checkpoint)
            if overfit_env_init is None:
                print("\n⚠️  WARNING: --overfit flag set but overfit_env_init.json not found!")
                print("    Will use random initialization instead.")

        # Goal position: plate center
        goal_xy = (0.5, 0.0)
        success_radius = 0.05  # 5cm - success if object within this distance
        min_init_dist = args.min_init_dist   # 15cm - reject initial positions closer than this

        print(f"\n{'='*60}")
        print(f"Evaluation Settings:")
        print(f"  Goal XY: {goal_xy}")
        print(f"  Success radius: {success_radius}m")
        print(f"  Min initial distance: {min_init_dist}m")
        print(f"  Horizon: {args.horizon}")
        print(f"  Num episodes: {args.num_episodes}")
        print(f"  Num inference steps: {num_inference_steps}")
        print(f"  N action steps: {n_action_steps} (execute this many steps before re-inferring)")
        print(f"  Visualize XYZ: {args.visualize_xyz}")
        print(f"  Visualize action chunk: {args.visualize_action_chunk}")
        print(f"  Overfit mode: {args.overfit}")
        if overfit_env_init is not None:
            print(f"  Overfit initial obj pose: {overfit_env_init['initial_obj_pose'][:3]}")
        if args.init_obj_xy is not None:
            print(f"  Custom init obj XY: [{args.init_obj_xy[0]:.3f}, {args.init_obj_xy[1]:.3f}]")
        print(f"{'='*60}")

        stats = []
        video_paths = []
        xyz_video_paths = []
        action_chunk_video_paths = []
        
        # Create action chunks output directory if visualization is enabled
        action_chunks_dir = out_dir / "action_chunks" if args.visualize_action_chunk else None
        if action_chunks_dir is not None:
            action_chunks_dir.mkdir(parents=True, exist_ok=True)
        
        for ep in range(args.num_episodes):
            print(f"\nEpisode {ep+1}/{args.num_episodes}")
            
            # Create a separate video writer for each episode
            video_path = out_dir / f"ep{ep}.mp4"
            writer = imageio.get_writer(video_path, fps=args.fps)
            
            # Create XYZ visualizer for this episode if enabled
            xyz_visualizer = None
            if args.visualize_xyz:
                from rev2fwd_il.data import create_eval_xyz_visualizer
                xyz_visualizer = create_eval_xyz_visualizer(
                    output_dir=xyz_curves_dir,
                    episode_id=ep,
                    fps=args.fps,
                )
            
            # Create action chunk visualizer for this episode if enabled
            action_chunk_visualizer = None
            if args.visualize_action_chunk:
                from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer
                action_chunk_visualizer = ActionChunkVisualizer(
                    output_dir=action_chunks_dir,
                    step_id=ep,  # Use episode id as step_id
                    fps=args.fps,
                )
            
            result = run_episode(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                horizon=args.horizon,
                writer=writer,
                goal_xy=goal_xy,
                success_radius=success_radius,
                min_init_dist=min_init_dist,
                has_wrist=has_wrist,
                include_obj_pose=include_obj_pose,
                include_gripper=include_gripper,
                xyz_visualizer=xyz_visualizer,
                overfit_env_init=overfit_env_init,
                n_action_steps=n_action_steps,
                init_obj_xy=tuple(args.init_obj_xy) if args.init_obj_xy is not None else None,
                start_markers=start_markers,
                goal_markers=goal_markers,
                marker_z=marker_z,
                action_chunk_visualizer=action_chunk_visualizer,
            )
            
            writer.close()
            
            # Rename video file based on success/failure
            if result['success']:
                new_video_path = out_dir / f"ep{ep}_success.mp4"
            else:
                new_video_path = out_dir / f"ep{ep}_failed.mp4"
            video_path.rename(new_video_path)
            video_path = new_video_path
            video_paths.append(video_path)
            
            # Generate XYZ curves video if enabled
            if xyz_visualizer is not None:
                suffix = "_success" if result['success'] else "_failed"
                xyz_video_path = xyz_visualizer.generate_video(filename_prefix=f"eval_xyz_curves{suffix}")
                xyz_video_paths.append(xyz_video_path)
            
            # Generate action chunk video if enabled
            if action_chunk_visualizer is not None:
                suffix = "_success" if result['success'] else "_failed"
                action_chunk_video_path = action_chunk_visualizer.generate_video(filename_prefix=f"eval_action_chunk{suffix}")
                action_chunk_video_paths.append(action_chunk_video_path)
            
            stats.append(result)
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            success_info = f" (achieved at step {result['success_step']})" if result['success_step'] else ""
            print(f"  Result: {status}{success_info} | total_steps={result['steps']} | final_dist={result['final_dist']:.4f}m")
            print(f"  Video saved: {video_path}")
            if xyz_visualizer is not None:
                print(f"  XYZ curves saved: {xyz_video_path}")
            if action_chunk_visualizer is not None:
                print(f"  Action chunk video saved: {action_chunk_video_path}")

        print(f"\nSaved {len(video_paths)} videos to {out_dir}/")
        if xyz_video_paths:
            print(f"Saved {len(xyz_video_paths)} XYZ curve videos to {xyz_curves_dir}/")
        if action_chunk_video_paths:
            print(f"Saved {len(action_chunk_video_paths)} action chunk videos to {action_chunks_dir}/")

        # Print summary statistics
        num_success = sum(1 for s in stats if s["success"])
        avg_steps = np.mean([s["steps"] for s in stats])
        avg_dist = np.mean([s["final_dist"] for s in stats])
        min_dist = min(s["final_dist"] for s in stats)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"  Success rate: {num_success}/{len(stats)} ({100*num_success/len(stats):.1f}%)")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average final distance: {avg_dist:.4f}m")
        print(f"  Min final distance: {min_dist:.4f}m")
        print(f"{'='*60}")

        # Close env before simulation_app to avoid hang
        print("Closing environment...", flush=True)
        env.close()
        print("Environment closed.", flush=True)

    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}\n")

    finally:
        print("Closing simulation app...", flush=True)
        simulation_app.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
