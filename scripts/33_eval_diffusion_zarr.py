#!/usr/bin/env python3
"""
Evaluate Diffusion Policy trained by train_dp_zarr.py in Isaac Lab.

This script loads a diffusion policy checkpoint trained by train_dp_zarr.py
and evaluates it on the pick-and-place task in Isaac Lab. It supports 
headless rendering and saves evaluation videos.

=============================================================================
KEY FEATURES
=============================================================================
- Compatible with models trained by train_dp_zarr.py
- Uses repo_id="local/diffusion_from_zarr" for dataset stats
- Supports headless rendering (--headless) for server evaluation
- Automatically saves videos of each episode
- Outputs evaluation metrics to JSON

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation (headless with video recording)
CUDA_VISIBLE_DEVICES=1 python scripts/33_eval_diffusion_zarr.py \
    --checkpoint runs/runs_500/diffusion_from_zarr/checkpoints/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/runs_500/diffusion_from_zarr/lerobot_dataset \
    --out_dir runs/runs_500/eval \
    --num_episodes 10 \
    --visualize_action_chunk \
    --headless

# Evaluate with GUI (for debugging)
CUDA_VISIBLE_DEVICES=1 python scripts/33_eval_diffusion_zarr.py \
    --checkpoint runs/runs_500/diffusion_from_zarr/checkpoints/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/runs_500/diffusion_from_zarr/lerobot_dataset \
    --out_dir runs/runs_500/eval \
    --num_episodes 3

# Custom parameters
CUDA_VISIBLE_DEVICES=0 python scripts/33_eval_diffusion_zarr.py \
    --checkpoint runs/runs_500/diffusion_from_zarr/checkpoints/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/runs_500/diffusion_from_zarr/lerobot_dataset \
    --out_dir runs/runs_500/eval \
    --num_episodes 20 \
    --max_steps 500 \
    --n_action_steps 8 \
    --success_radius 0.03 \
    --headless

# Different checkpoint directory
CUDA_VISIBLE_DEVICES=1 python scripts/33_eval_diffusion_zarr.py \
    --checkpoint runs/diffusion_from_zarr/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/diffusion_from_zarr/lerobot_dataset \
    --out_dir eval_results \
    --headless

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# Suppress numpy warnings
def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x
    return npwarn_decorator


np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)


# ---------------------------------------------------------------------------
# Argument parsing (before AppLauncher)
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Diffusion Policy from train_dp_zarr.py in Isaac Lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # =========================================================================
    # Model and Dataset Paths
    # =========================================================================
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/runs_500/diffusion_from_zarr/checkpoints/checkpoints/last/pretrained_model",
        help="Path to pretrained model directory (contains config.json, model.safetensors).",
    )
    parser.add_argument(
        "--lerobot_dataset",
        type=str,
        default=None,
        help="Path to LeRobot dataset directory (for normalization stats). "
             "Optional if preprocessors are saved in checkpoint directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_results",
        help="Output directory for videos and results.",
    )

    # =========================================================================
    # Environment Settings
    # =========================================================================
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab task ID.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments.",
    )

    # =========================================================================
    # Evaluation Settings
    # =========================================================================
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Action steps to execute per prediction. If None, uses training config.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps. If None, uses training config.",
    )

    # =========================================================================
    # Success Criteria
    # =========================================================================
    parser.add_argument(
        "--success_radius",
        type=float,
        default=0.03,
        help="Success threshold distance in meters.",
    )
    parser.add_argument(
        "--goal_x",
        type=float,
        default=0.6,
        help="Goal X position in world coordinates.",
    )
    parser.add_argument(
        "--goal_y",
        type=float,
        default=0.0,
        help="Goal Y position in world coordinates.",
    )
    parser.add_argument(
        "--goal_z",
        type=float,
        default=0.2,
        help="Goal Z position in world coordinates.",
    )

    # =========================================================================
    # Camera Settings
    # =========================================================================
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Camera image width (should match training).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Camera image height (should match training).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video FPS.",
    )

    # =========================================================================
    # Video Recording
    # =========================================================================
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Save video of each episode (default: True).",
    )
    parser.add_argument(
        "--no_save_video",
        action="store_true",
        help="Disable video saving.",
    )

    # =========================================================================
    # Action Chunk Visualization
    # =========================================================================
    parser.add_argument(
        "--visualize_action_chunk",
        action="store_true",
        help="Generate action chunk visualization video for each episode.",
    )
    parser.add_argument(
        "--action_chunk_fps",
        type=int,
        default=5,
        help="FPS for action chunk visualization video. Default: 5.",
    )

    # =========================================================================
    # Reproducibility
    # =========================================================================
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    # =========================================================================
    # Isaac Lab AppLauncher Flags
    # =========================================================================
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    # Post-process arguments
    args.enable_cameras = True
    if args.no_save_video:
        args.save_video = False

    return args


# Parse args before launching app
args = _parse_args()

# Launch Isaac Sim
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Imports after app launch
# ---------------------------------------------------------------------------
import torch
import gymnasium as gym

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("[WARNING] imageio not installed. Video saving disabled.")

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg

# Action chunk visualization
from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------
def compute_camera_quat_from_lookat(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0, 0, 1),
) -> Tuple[float, float, float, float]:
    """
    Compute camera quaternion (w, x, y, z) for a camera looking at target from eye.
    
    Args:
        eye: Camera position (x, y, z).
        target: Look-at target position (x, y, z).
        up: Up vector, default is Z-up.
    
    Returns:
        Quaternion (w, x, y, z).
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
    q_xyzw = rot.as_quat()
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    return (qw, qx, qy, qz)


def add_camera_to_env_cfg(env_cfg, image_width: int, image_height: int) -> None:
    """
    Add table camera to environment configuration.
    
    This matches the camera setup used in train_dp_zarr.py data collection.
    
    Args:
        env_cfg: Environment configuration object.
        image_width: Camera image width.
        image_height: Camera image height.
    """
    # Table Camera - Third-person fixed view
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

    # Increase env spacing to avoid camera overlap
    env_cfg.scene.env_spacing = 5.0

    # Disable debug visualization
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "ee_frame"):
        env_cfg.scene.ee_frame.debug_vis = False


# ---------------------------------------------------------------------------
# Marker utilities
# ---------------------------------------------------------------------------
def create_target_markers(num_envs: int, device: str):
    """
    Create visualization markers for start and goal positions.
    
    Args:
        num_envs: Number of environments.
        device: Torch device.
    
    Returns:
        Tuple of (start_markers, goal_markers, marker_z).
    """
    marker_radius = 0.05
    marker_height = 0.002
    table_z = 0.0
    marker_z = table_z + marker_height / 2 + 0.001

    start_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/StartMarkers",
        markers={
            "start": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            ),
        },
    )
    start_markers = VisualizationMarkers(start_marker_cfg)

    goal_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
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
    """Update marker positions in the scene."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    env_origins = env.unwrapped.scene.env_origins

    start_x, start_y = float(start_xy[0]), float(start_xy[1])
    goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])

    start_positions = torch.zeros((num_envs, 3), device=device)
    start_positions[:, 0] = start_x
    start_positions[:, 1] = start_y
    start_positions[:, 2] = marker_z
    start_positions_w = start_positions + env_origins

    goal_positions = torch.zeros((num_envs, 3), device=device)
    goal_positions[:, 0] = goal_x
    goal_positions[:, 1] = goal_y
    goal_positions[:, 2] = marker_z
    goal_positions_w = goal_positions + env_origins

    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)

    start_markers.visualize(start_positions_w, identity_quat)
    goal_markers.visualize(goal_positions_w, identity_quat)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_policy_config(pretrained_dir: str) -> Dict[str, Any]:
    """
    Load and parse policy configuration from config.json.
    
    Args:
        pretrained_dir: Path to pretrained model directory.
    
    Returns:
        Dictionary with parsed config info.
    """
    config_path = Path(pretrained_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    input_features = config_dict.get("input_features", {})
    output_features = config_dict.get("output_features", {})

    # Check for image shape
    image_shape = None
    if "observation.image" in input_features:
        image_shape = tuple(input_features["observation.image"]["shape"])

    # State dimension
    state_dim = None
    if "observation.state" in input_features:
        state_dim = input_features["observation.state"]["shape"][0]

    # Action dimension
    action_dim = None
    if "action" in output_features:
        action_dim = output_features["action"]["shape"][0]

    return {
        "image_shape": image_shape,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_obs_steps": config_dict.get("n_obs_steps", 2),
        "horizon": config_dict.get("horizon", 16),
        "n_action_steps": config_dict.get("n_action_steps", 8),
        "raw_config": config_dict,
    }


def load_policy_with_processors(
    checkpoint_path: str,
    dataset_dir: str | None,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
):
    """
    Load pretrained DiffusionPolicy with preprocessors.
    
    This function loads the policy weights and creates preprocessors.
    If dataset_dir is provided, it uses the normalization stats from the dataset.
    Otherwise, it loads preprocessors directly from the checkpoint directory.
    
    Args:
        checkpoint_path: Path to pretrained model directory.
        dataset_dir: Path to LeRobot dataset directory (optional).
        device: Torch device for inference.
        num_inference_steps: Override number of diffusion denoising steps.
        n_action_steps: Override number of action steps to execute.
    
    Returns:
        Tuple of (policy, preprocessor, postprocessor, config).
    """
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"[INFO] Loading policy from: {checkpoint_path}")

    pretrained_path = Path(checkpoint_path)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"

    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Remove type key (not a DiffusionConfig field)
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
            norm_mapping[FeatureType[key] if isinstance(key, str) else key] = (
                NormalizationMode[val] if isinstance(val, str) else val
            )
        config_dict["normalization_mapping"] = norm_mapping

    # Convert lists to tuples
    if "crop_shape" in config_dict and isinstance(config_dict["crop_shape"], list):
        config_dict["crop_shape"] = tuple(config_dict["crop_shape"])
    if "optimizer_betas" in config_dict and isinstance(config_dict["optimizer_betas"], list):
        config_dict["optimizer_betas"] = tuple(config_dict["optimizer_betas"])
    if "down_dims" in config_dict and isinstance(config_dict["down_dims"], list):
        config_dict["down_dims"] = tuple(config_dict["down_dims"])

    # Override inference parameters if specified
    if num_inference_steps is not None:
        print(f"[INFO] Overriding num_inference_steps: {config_dict.get('num_inference_steps')} -> {num_inference_steps}")
        config_dict["num_inference_steps"] = num_inference_steps

    if n_action_steps is not None:
        print(f"[INFO] Overriding n_action_steps: {config_dict.get('n_action_steps')} -> {n_action_steps}")
        config_dict["n_action_steps"] = n_action_steps

    # Create config and policy
    cfg = DiffusionConfig(**config_dict)
    policy = DiffusionPolicy(cfg)

    # Load weights
    state_dict = load_file(model_path)
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    print(f"[INFO] Policy loaded: n_obs_steps={cfg.n_obs_steps}, horizon={cfg.horizon}, "
          f"n_action_steps={cfg.n_action_steps}, num_inference_steps={cfg.num_inference_steps}")

    # Create preprocessors
    # Option 1: Load from dataset stats if dataset_dir is provided
    # Option 2: Load saved preprocessors from checkpoint directory
    dataset_stats = None
    if dataset_dir is not None:
        dataset_path = Path(dataset_dir)
        info_path = dataset_path / "meta" / "info.json"
        if info_path.exists():
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            try:
                dataset_metadata = LeRobotDatasetMetadata(
                    repo_id="local/diffusion_from_zarr",
                    root=str(dataset_path),
                    local_files_only=True,  # Don't try to fetch from HuggingFace Hub
                )
                dataset_stats = dataset_metadata.stats
                print(f"[INFO] Loaded dataset stats from: {dataset_path}")
                print(f"[INFO] Dataset stats keys: {list(dataset_stats.keys())}")
            except Exception as e:
                print(f"[WARNING] Failed to load dataset stats: {e}")
                print("[INFO] Will load preprocessors from checkpoint directory instead.")
        else:
            print(f"[WARNING] Dataset info not found at: {info_path}")
            print("[INFO] Will load preprocessors from checkpoint directory instead.")

    # Check if preprocessors are saved in checkpoint directory
    preprocessor_json = pretrained_path / "policy_preprocessor.json"
    postprocessor_json = pretrained_path / "policy_postprocessor.json"
    
    if preprocessor_json.exists() and postprocessor_json.exists():
        print(f"[INFO] Found saved preprocessors in checkpoint directory")
        # Load preprocessors from checkpoint (will use saved normalizer stats)
        preprocessor_overrides = {"device_processor": {"device": device}}
        postprocessor_overrides = {"device_processor": {"device": device}}
        
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(pretrained_path),
            dataset_stats=dataset_stats,  # Can be None if not provided
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )
        print(f"[INFO] Preprocessors loaded from checkpoint directory")
    elif dataset_stats is not None:
        # Fallback: create preprocessors from dataset stats only
        preprocessor_overrides = {"device_processor": {"device": device}}
        postprocessor_overrides = {"device_processor": {"device": device}}
        
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(pretrained_path),
            dataset_stats=dataset_stats,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )
        print(f"[INFO] Preprocessors created from dataset stats")
    else:
        raise RuntimeError(
            "Cannot create preprocessors: no saved preprocessors found in checkpoint "
            "and no valid dataset directory provided.\n"
            "Please specify --lerobot_dataset with a valid LeRobot dataset path."
        )

    return policy, preprocessor, postprocessor, cfg


# ---------------------------------------------------------------------------
# Full horizon action chunk extraction
# ---------------------------------------------------------------------------
def get_full_horizon_action_chunk(
    policy,
    obs: dict,
    postprocessor,
) -> tuple:
    """
    Get the full horizon action chunk from diffusion policy.
    
    The standard select_action() only returns n_action_steps actions.
    This function extracts the full horizon-length prediction for visualization.
    
    Args:
        policy: DiffusionPolicy instance.
        obs: Preprocessed observation dict (after preprocessor, before select_action modifies it).
        postprocessor: Action denormalization postprocessor.
    
    Returns:
        Tuple of (action_chunk_norm, action_chunk_raw), both shape (horizon, action_dim).
    """
    config = policy.config
    n_obs_steps = config.n_obs_steps
    
    # Build batch from policy queues (populated by previous select_action call)
    # Queue format: deque of n_obs_steps tensors, each (B, ...)
    batch = {}
    for k in policy._queues:
        queue_list = list(policy._queues[k])
        if len(queue_list) == 0:
            continue
        # Stack along time dimension: (B, n_obs_steps, ...)
        batch[k] = torch.stack(queue_list, dim=1)
    
    # Prepare image format for diffusion model
    # The diffusion model expects 'observation.images' with shape (B, n_obs_steps, num_cams, C, H, W)
    if hasattr(config, 'image_features') and config.image_features:
        images_list = []
        for key in config.image_features:
            if key in batch:
                # batch[key] shape: (B, n_obs_steps, C, H, W)
                images_list.append(batch[key])
        if images_list:
            # Stack along new camera dimension: (B, n_obs_steps, num_cams, C, H, W)
            batch['observation.images'] = torch.stack(images_list, dim=2)
    
    # Get the global conditioning
    global_cond = policy.diffusion._prepare_global_conditioning(batch)
    
    # Sample full horizon actions (not truncated to n_action_steps)
    batch_size = global_cond.shape[0]
    full_actions = policy.diffusion.conditional_sample(batch_size, global_cond=global_cond)
    # full_actions shape: (B, horizon, action_dim)
    
    # Extract normalized actions
    action_chunk_norm = full_actions[0].detach().cpu().numpy()  # (horizon, action_dim)
    
    # Denormalize using postprocessor
    action_chunk_raw = postprocessor(full_actions)[0].detach().cpu().numpy()  # (horizon, action_dim)
    
    return action_chunk_norm, action_chunk_raw


# ---------------------------------------------------------------------------
# Observation preparation
# ---------------------------------------------------------------------------
def prepare_observation(
    rgb: np.ndarray,
    ee_pose: np.ndarray,
    device: str,
) -> dict:
    """
    Prepare observation dict for policy input.
    
    Args:
        rgb: (H, W, 3) uint8 image from camera.
        ee_pose: (7,) float32 end-effector pose [x, y, z, qw, qx, qy, qz].
        device: Torch device.
    
    Returns:
        Dictionary with observation tensors.
    """
    # Image: (H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1]
    img = rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # State: (7,) -> (1, 7)
    state = ee_pose.astype(np.float32)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    return {
        "observation.image": img_tensor,
        "observation.state": state_tensor,
    }


# ---------------------------------------------------------------------------
# Action processing
# ---------------------------------------------------------------------------
def process_action_chunk(action_chunk, device: str) -> np.ndarray:
    """
    Process action chunk from policy output.
    
    Args:
        action_chunk: Policy output, could be torch.Tensor or np.ndarray.
        device: Torch device (unused, kept for compatibility).
    
    Returns:
        (horizon, action_dim) np.ndarray.
    """
    if isinstance(action_chunk, torch.Tensor):
        action_chunk = action_chunk.cpu().numpy()

    # Remove batch dimension if present
    if action_chunk.ndim == 3:
        action_chunk = action_chunk[0]  # (1, horizon, action_dim) -> (horizon, action_dim)
    elif action_chunk.ndim == 1:
        action_chunk = action_chunk[np.newaxis, :]  # (action_dim,) -> (1, action_dim)

    return action_chunk


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------
def save_video(frames: list, output_path: Path, fps: int = 20):
    """
    Save frames as MP4 video.
    
    Args:
        frames: List of (H, W, 3) uint8 images.
        output_path: Output file path.
        fps: Frames per second.
    """
    if not frames or not IMAGEIO_AVAILABLE:
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"[INFO] Video saved: {output_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save video: {e}")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_episode_metrics(
    obj_final_xy: np.ndarray,
    goal_xy: np.ndarray,
    success_radius: float,
    episode_length: int,
) -> dict:
    """
    Compute metrics for a single episode.
    
    Args:
        obj_final_xy: Final object XY position.
        goal_xy: Goal XY position.
        success_radius: Success threshold distance.
        episode_length: Number of steps in episode.
    
    Returns:
        Dictionary with episode metrics.
    """
    dist = float(np.linalg.norm(obj_final_xy - goal_xy))
    success = dist < success_radius

    return {
        "success": success,
        "final_distance": dist,
        "episode_length": episode_length,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main():
    """Main evaluation function."""
    # Setup random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy config to check requirements
    policy_config = load_policy_config(args.checkpoint)

    print("\n" + "=" * 60)
    print("Policy Configuration")
    print("=" * 60)
    print(f"  Image shape: {policy_config['image_shape']}")
    print(f"  State dim: {policy_config['state_dim']}")
    print(f"  Action dim: {policy_config['action_dim']}")
    print(f"  n_obs_steps: {policy_config['n_obs_steps']}")
    print(f"  horizon: {policy_config['horizon']}")
    print(f"  n_action_steps: {policy_config['n_action_steps']}")
    print("=" * 60)

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
    )

    # Add camera to environment
    print("[INFO] Adding camera to environment config...")
    add_camera_to_env_cfg(env_cfg, args.image_width, args.image_height)

    # Create environment
    print("[INFO] Creating environment...")
    env = gym.make(args.task, cfg=env_cfg)
    obs_dict, _ = env.reset()

    device = env.unwrapped.device

    # Get camera reference
    camera = env.unwrapped.scene.sensors["table_cam"]

    # Load policy with preprocessors
    policy, preprocessor, postprocessor, policy_cfg = load_policy_with_processors(
        args.checkpoint,
        args.lerobot_dataset,
        device,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
    )

    n_action_steps = policy_cfg.n_action_steps

    # Goal position (configurable via command line)
    goal_world = np.array([args.goal_x, args.goal_y, args.goal_z], dtype=np.float32)
    goal_xy = (args.goal_x, args.goal_y)

    # Initialize action tensor
    actions = torch.zeros(env.unwrapped.action_space.shape, device=device)
    actions[:, 3] = 1.0  # quaternion w
    actions[:, 7] = 1.0  # gripper open

    # Evaluation metrics
    all_metrics = []
    success_count = 0

    print("\n" + "=" * 60)
    print("Starting Diffusion Policy Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.lerobot_dataset or '(using saved preprocessors)'}")
    print(f"  Output: {out_dir}")
    print(f"  Num episodes: {args.num_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  Success radius: {args.success_radius}m")
    print(f"  Goal position: [{args.goal_x}, {args.goal_y}, {args.goal_z}]")
    print(f"  Headless: {args.headless}")
    print(f"  Save video: {args.save_video}")
    print(f"  Visualize action chunk: {args.visualize_action_chunk}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for ep_idx in range(args.num_episodes):
        print(f"\n[Episode {ep_idx + 1}/{args.num_episodes}]")

        # Reset environment and policy
        policy.reset()
        obs_dict, _ = env.reset()

        # Get initial object position
        object_data: RigidObjectData = env.unwrapped.scene["object"].data
        obj_pos = object_data.root_pos_w[0].cpu().numpy() - env.unwrapped.scene.env_origins[0].cpu().numpy()
        init_obj_xy = (obj_pos[0], obj_pos[1])

        init_dist = np.linalg.norm(np.array(init_obj_xy) - np.array(goal_xy))
        print(f"  Initial object XY: [{init_obj_xy[0]:.3f}, {init_obj_xy[1]:.3f}]")
        print(f"  Goal XY: [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}]")
        print(f"  Initial distance: {init_dist:.3f}m")

        # Video frames
        video_frames = [] if args.save_video else None

        # Action chunk visualizer for this episode
        action_chunk_viz = None
        if args.visualize_action_chunk:
            action_chunk_viz = ActionChunkVisualizer(
                output_dir=out_dir / "action_chunk_viz",
                step_id=ep_idx,
                fps=args.action_chunk_fps,
            )

        # Action buffer
        action_buffer = None
        action_buffer_idx = 0

        # Reset actions
        actions.zero_()
        actions[:, 3] = 1.0
        actions[:, 7] = 1.0

        success = False
        final_dist = None

        for step in range(args.max_steps):
            # =========================================================
            # 1) Get current observation
            # =========================================================
            # RGB image
            rgb = camera.data.output["rgb"]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            rgb_np = rgb[0].detach().cpu().numpy().astype(np.uint8)

            # End-effector pose
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
            ee_pose_np = ee_pose[0].detach().cpu().numpy().astype(np.float32)

            # Object pose (for metrics and optionally for state)
            obj_pos = object_data.root_pos_w[0].cpu().numpy() - env.unwrapped.scene.env_origins[0].cpu().numpy()
            obj_quat = object_data.root_quat_w[0].cpu().numpy()
            obj_pose_np = np.concatenate([obj_pos, obj_quat]).astype(np.float32)  # (7,)
            obj_xy = obj_pos[:2]

            # Save video frame
            if video_frames is not None:
                video_frames.append(rgb_np.copy())

            # Check success
            dist = np.linalg.norm(obj_xy - np.array(goal_xy))
            if dist < args.success_radius:
                success = True
                final_dist = dist
                print(f"  SUCCESS at step {step}! Distance: {dist:.4f}m")
                break

            # =========================================================
            # 2) Get action from policy (or use buffered action)
            # =========================================================
            if action_buffer is None or action_buffer_idx >= len(action_buffer):
                # Prepare observation
                obs_raw = prepare_observation(rgb_np, ee_pose_np, device)

                # Apply preprocessing (normalization)
                obs = preprocessor(obs_raw)

                # Get action from policy
                with torch.no_grad():
                    # Get raw normalized action chunk for visualization
                    action_chunk_norm = policy.select_action(obs)
                    # Apply postprocessing (denormalization)
                    action_chunk = postprocessor(action_chunk_norm)

                action_buffer = process_action_chunk(action_chunk, device)
                action_buffer_idx = 0

                # Debug print (first step only)
                if step == 0:
                    print(f"  [DEBUG] Action chunk shape: {action_buffer.shape}")
                    print(f"  [DEBUG] First action: {action_buffer[0]}")

                # =========================================================
                # Collect action chunk visualization data (full horizon)
                # =========================================================
                if action_chunk_viz is not None:
                    # Get FULL horizon action chunk for visualization
                    # (select_action only returns n_action_steps, we want full horizon)
                    full_chunk_norm, full_chunk_raw = get_full_horizon_action_chunk(
                        policy, obs, postprocessor
                    )
                    # full_chunk_norm/raw shape: (horizon, action_dim)

                    # Get normalized state (ee_pose) from preprocessed observation
                    if "observation.state" in obs:
                        state_norm = obs["observation.state"]
                        if state_norm.dim() == 3:  # (1, n_obs_steps, state_dim)
                            ee_norm_np = state_norm[0, -1, :3].cpu().numpy()
                        else:  # (1, state_dim)
                            ee_norm_np = state_norm[0, :3].cpu().numpy()
                    else:
                        ee_norm_np = ee_pose_np[:3]

                    action_chunk_viz.add_frame(
                        ee_pose_raw=ee_pose_np[:3],
                        ee_pose_norm=ee_norm_np,
                        action_chunk_norm=full_chunk_norm[:, :3],  # XYZ only, full horizon
                        action_chunk_raw=full_chunk_raw[:, :3],     # XYZ only, full horizon
                        gt_chunk_norm=None,  # No GT during evaluation
                        gt_chunk_raw=None,
                        table_image=rgb_np,
                        wrist_image=None,
                    )

            # Get current action from buffer
            pred_action = action_buffer[action_buffer_idx]
            action_buffer_idx += 1

            # =========================================================
            # 3) Execute action in environment
            # =========================================================
            actions[0] = torch.from_numpy(pred_action).to(device)
            obs_dict, reward, terminated, truncated, info = env.step(actions)

            # Check for early termination
            dones = terminated | truncated
            if dones[0].item():
                break

        # Episode done
        if final_dist is None:
            final_dist = dist

        metrics = compute_episode_metrics(
            obj_final_xy=obj_xy,
            goal_xy=np.array(goal_xy),
            success_radius=args.success_radius,
            episode_length=step + 1,
        )
        all_metrics.append(metrics)

        if metrics["success"]:
            success_count += 1

        print(f"  Steps: {step + 1}, Success: {metrics['success']}, Final dist: {final_dist:.4f}m")

        # Save video
        if video_frames is not None and len(video_frames) > 0:
            video_path = out_dir / f"episode_{ep_idx:04d}.mp4"
            save_video(video_frames, video_path, args.fps)

        # Generate action chunk visualization video for this episode
        if action_chunk_viz is not None and len(action_chunk_viz.frames) > 0:
            action_chunk_viz.generate_video(filename_prefix=f"ep{ep_idx:04d}_action_chunk")

    # =========================================================
    # Compute and save final results
    # =========================================================
    elapsed = time.time() - start_time

    success_rate = success_count / len(all_metrics) if all_metrics else 0.0
    avg_distance = np.mean([m["final_distance"] for m in all_metrics]) if all_metrics else 0.0
    avg_length = np.mean([m["episode_length"] for m in all_metrics]) if all_metrics else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes: {len(all_metrics)}")
    print(f"  Success Rate: {success_rate * 100:.1f}% ({success_count}/{len(all_metrics)})")
    print(f"  Avg Final Distance: {avg_distance:.4f}m")
    print(f"  Avg Episode Length: {avg_length:.1f}")
    print(f"  Total Time: {elapsed:.1f}s")
    print("=" * 60)

    # Save results to JSON
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "dataset": args.lerobot_dataset or "(saved preprocessors)",
            "num_episodes": len(all_metrics),
            "max_steps": args.max_steps,
            "n_action_steps": n_action_steps,
            "success_radius": args.success_radius,
            "goal_position": [args.goal_x, args.goal_y, args.goal_z],
            "image_size": [args.image_width, args.image_height],
        },
        "summary": {
            "success_rate": success_rate,
            "avg_final_distance": avg_distance,
            "avg_episode_length": avg_length,
            "total_successes": success_count,
        },
        "episodes": all_metrics,
        "evaluation_time_s": elapsed,
    }

    results_file = out_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {results_file}")

    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
