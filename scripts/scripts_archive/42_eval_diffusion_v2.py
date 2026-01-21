#!/usr/bin/env python3
"""
Evaluate Diffusion Policy trained by script 32 (train_lerobot_dataset.py).

This script loads a diffusion policy and evaluates it on the pick-and-place task
in Isaac Lab. It supports dual cameras (table + wrist) and saves videos in headless mode.

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation with video recording
CUDA_VISIBLE_DEVICES=0 python scripts/42_eval_diffusion_v2.py \
    --checkpoint runs/diffusion_A_mark_v2/checkpoints/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out_dir runs/diffusion_A_mark_v2/eval \
    --num_episodes 5 \
    --headless

# With XYZ visualization
CUDA_VISIBLE_DEVICES=0 python scripts/42_eval_diffusion_v2.py \
    --checkpoint runs/diffusion_A_mark_v2/checkpoints/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out_dir runs/diffusion_A_mark_v2/eval \
    --num_episodes 3 \
    --visualize_xyz \
    --headless

# Custom parameters
CUDA_VISIBLE_DEVICES=0 python scripts/42_eval_diffusion_v2.py \
    --checkpoint runs/diffusion_A_mark_v2/checkpoints/checkpoints/last/pretrained_model \
    --lerobot_dataset runs/diffusion_A_mark/lerobot_dataset \
    --out_dir runs/diffusion_A_mark_v2/eval \
    --num_episodes 10 \
    --horizon 500 \
    --n_action_steps 8 \
    --headless

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x
    return npwarn_decorator


np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)


# ---------------------------------------------------------------------------
# Argument parsing (before AppLauncher)
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Diffusion Policy from script 32 in Isaac Lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and dataset
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained model directory (contains config.json, model.safetensors).",
    )
    parser.add_argument(
        "--lerobot_dataset",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory (for normalization stats).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_results",
        help="Output directory for videos and results.",
    )

    # Evaluation settings
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab task ID.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Action steps per inference. If None, uses training config.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps. If None, uses training config.",
    )

    # Camera settings
    parser.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="Camera image width.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=128,
        help="Camera image height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video FPS.",
    )

    # Success criteria
    parser.add_argument(
        "--success_radius",
        type=float,
        default=0.05,
        help="Success threshold distance in meters.",
    )
    parser.add_argument(
        "--min_init_dist",
        type=float,
        default=0.15,
        help="Minimum initial distance from goal.",
    )

    # Visualization
    parser.add_argument(
        "--visualize_xyz",
        action="store_true",
        help="Generate XYZ curve visualization.",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    # Custom initial position
    parser.add_argument(
        "--init_obj_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Custom initial XY position for object.",
    )

    # Isaac Lab AppLauncher flags
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
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
import imageio

import isaaclab.sim as sim_utils
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import CameraCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ---------------------------------------------------------------------------
# Camera utilities (from script 41)
# ---------------------------------------------------------------------------
def compute_camera_quat_from_lookat(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0, 0, 1),
) -> Tuple[float, float, float, float]:
    """Return (w, x, y, z) quaternion for a camera looking at target from eye."""
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
    """Add table and wrist cameras to environment config."""
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

    # Wrist Camera - Eye-in-hand
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


# ---------------------------------------------------------------------------
# Marker utilities (from script 41)
# ---------------------------------------------------------------------------
def create_target_markers(num_envs: int, device: str):
    """Create visualization markers for start and goal positions."""
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
    """Update marker positions."""
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
    """Load policy configuration from config.json."""
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


def load_policy_with_processors(
    checkpoint_path: str,
    dataset_dir: str,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
):
    """Load pretrained DiffusionPolicy with preprocessors."""
    from safetensors.torch import load_file
    from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    print(f"[INFO] Loading policy from: {checkpoint_path}")

    pretrained_path = Path(checkpoint_path)
    config_path = pretrained_path / "config.json"
    model_path = pretrained_path / "model.safetensors"

    # Load config
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
            norm_mapping[FeatureType[key] if isinstance(key, str) else key] = NormalizationMode[val] if isinstance(val, str) else val
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

    # Load dataset metadata for normalization stats
    dataset_path = Path(dataset_dir)
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="local/rev2fwd_diffusion",
        root=str(dataset_path),
    )

    print(f"[INFO] Dataset stats keys: {list(dataset_metadata.stats.keys())}")

    # Create preprocessors
    preprocessor_overrides = {"device_processor": {"device": device}}
    postprocessor_overrides = {"device_processor": {"device": device}}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(pretrained_path),
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    print(f"[INFO] Preprocessors created with dataset stats from: {dataset_path}")

    return policy, preprocessor, postprocessor, cfg


# ---------------------------------------------------------------------------
# Observation preparation
# ---------------------------------------------------------------------------
def prepare_observation(
    table_rgb: np.ndarray,
    wrist_rgb: np.ndarray | None,
    ee_pose: np.ndarray,
    obj_pose: np.ndarray | None,
    device: str,
    include_obj_pose: bool = False,
) -> dict:
    """Prepare observation dict for policy."""
    # Table image: (H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1]
    img = table_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # State
    if include_obj_pose and obj_pose is not None:
        state = np.concatenate([ee_pose, obj_pose], axis=-1).astype(np.float32)
    else:
        state = ee_pose.astype(np.float32)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    obs = {
        "observation.image": img_tensor,
        "observation.state": state_tensor,
    }

    # Wrist image if available
    if wrist_rgb is not None:
        wrist_img = wrist_rgb.astype(np.float32) / 255.0
        wrist_img = np.transpose(wrist_img, (2, 0, 1))
        wrist_tensor = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
        obs["observation.wrist_image"] = wrist_tensor

    return obs


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------
def save_video(frames: list, output_path: Path, fps: int = 20):
    """Save frames as MP4 video."""
    if not frames:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"[INFO] Video saved: {output_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save video: {e}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main():
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load policy config to determine requirements
    policy_config = load_policy_config(args.checkpoint)
    has_wrist = policy_config["has_wrist"]
    include_obj_pose = policy_config["include_obj_pose"]
    state_dim = policy_config["state_dim"]

    print("\n" + "=" * 60)
    print("Policy Configuration")
    print("=" * 60)
    print(f"  Has wrist camera: {has_wrist}")
    print(f"  Include obj_pose: {include_obj_pose}")
    print(f"  State dim: {state_dim}")
    print("=" * 60)

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=True,
    )

    # Add cameras
    add_camera_to_env_cfg(env_cfg, args.image_width, args.image_height)

    # Create environment
    print("[INFO] Creating environment...")
    env = gym.make(args.task, cfg=env_cfg)
    obs_dict, _ = env.reset()

    device = env.unwrapped.device

    # Get camera references
    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None) if has_wrist else None

    # Create markers
    start_markers, goal_markers, marker_z = create_target_markers(1, device)

    # Load policy with preprocessors
    policy, preprocessor, postprocessor, policy_cfg = load_policy_with_processors(
        args.checkpoint,
        args.lerobot_dataset,
        device,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
    )

    n_action_steps = policy_cfg.n_action_steps

    # Goal position (plate center)
    goal_xy = (0.5, 0.0)

    # Evaluation metrics
    all_metrics = []
    success_count = 0

    print("\n" + "=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.lerobot_dataset}")
    print(f"  Num episodes: {args.num_episodes}")
    print(f"  Horizon: {args.horizon}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  Success radius: {args.success_radius}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for ep_idx in range(args.num_episodes):
        print(f"\n[Episode {ep_idx + 1}/{args.num_episodes}]")

        # Reset
        policy.reset()
        obs_dict, _ = env.reset()

        # Get initial object position
        object_data: RigidObjectData = env.unwrapped.scene["object"].data
        obj_pos = object_data.root_pos_w[0].cpu().numpy() - env.unwrapped.scene.env_origins[0].cpu().numpy()
        obj_quat = object_data.root_quat_w[0].cpu().numpy()
        init_obj_xy = (obj_pos[0], obj_pos[1])

        # Custom initial position
        if args.init_obj_xy is not None:
            from rev2fwd_il.sim.scene_api import teleport_object_to_pose
            target_pose = np.array([
                args.init_obj_xy[0], args.init_obj_xy[1], obj_pos[2],
                1.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            teleport_object_to_pose(env, target_pose)
            for _ in range(5):
                env.unwrapped.scene.write_data_to_sim()
                env.unwrapped.sim.step()
                env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
            obj_pos = object_data.root_pos_w[0].cpu().numpy() - env.unwrapped.scene.env_origins[0].cpu().numpy()
            init_obj_xy = (obj_pos[0], obj_pos[1])

        init_dist = np.linalg.norm(np.array(init_obj_xy) - np.array(goal_xy))
        print(f"  Initial object XY: [{init_obj_xy[0]:.3f}, {init_obj_xy[1]:.3f}]")
        print(f"  Goal XY: [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}]")
        print(f"  Initial distance: {init_dist:.3f}m")

        # Update markers
        update_target_markers(start_markers, goal_markers, init_obj_xy, goal_xy, marker_z, env)

        # Video frames
        video_frames = []

        # Action buffer
        action_buffer = None
        action_buffer_idx = 0

        # Initialize actions tensor
        actions = torch.zeros(env.unwrapped.action_space.shape, device=device)
        actions[:, 3] = 1.0  # quaternion w
        actions[:, 7] = 1.0  # gripper open

        success = False
        final_dist = None

        for step in range(args.horizon):
            with torch.no_grad():
                # Get observations
                # Table camera
                rgb = table_camera.data.output["rgb"]
                if rgb.shape[-1] > 3:
                    rgb = rgb[..., :3]
                table_rgb = rgb[0].cpu().numpy().astype(np.uint8)

                # Wrist camera
                wrist_rgb = None
                if wrist_camera is not None:
                    wrist_rgb_raw = wrist_camera.data.output["rgb"]
                    if wrist_rgb_raw.shape[-1] > 3:
                        wrist_rgb_raw = wrist_rgb_raw[..., :3]
                    wrist_rgb = wrist_rgb_raw[0].cpu().numpy().astype(np.uint8)

                # EE pose
                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
                tcp_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
                ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
                ee_pose_np = ee_pose[0].cpu().numpy().astype(np.float32)

                # Object pose
                obj_pos = object_data.root_pos_w[0].cpu().numpy() - env.unwrapped.scene.env_origins[0].cpu().numpy()
                obj_quat = object_data.root_quat_w[0].cpu().numpy()
                obj_pose_np = np.concatenate([obj_pos, obj_quat]).astype(np.float32)

                # Save video frame
                video_frames.append(table_rgb.copy())

                # Check success
                obj_xy = obj_pos[:2]
                dist = np.linalg.norm(obj_xy - np.array(goal_xy))
                if dist < args.success_radius:
                    success = True
                    final_dist = dist
                    print(f"  SUCCESS at step {step}! Distance: {dist:.4f}m")
                    break

                # Get action from policy
                if action_buffer is None or action_buffer_idx >= len(action_buffer):
                    # Prepare observation
                    obs_raw = prepare_observation(
                        table_rgb, wrist_rgb, ee_pose_np, obj_pose_np,
                        device, include_obj_pose
                    )

                    # Preprocess
                    obs = preprocessor(obs_raw)

                    # Get action
                    with torch.no_grad():
                        action_chunk = policy.select_action(obs)
                        action_chunk = postprocessor(action_chunk)

                    # Process action chunk
                    if isinstance(action_chunk, torch.Tensor):
                        action_chunk = action_chunk.cpu().numpy()
                    if action_chunk.ndim == 3:
                        action_chunk = action_chunk[0]
                    elif action_chunk.ndim == 1:
                        action_chunk = action_chunk[np.newaxis, :]

                    action_buffer = action_chunk
                    action_buffer_idx = 0

                # Get current action
                pred_action = action_buffer[action_buffer_idx]
                action_buffer_idx += 1

                # Execute action
                actions[0] = torch.from_numpy(pred_action).to(device)
                obs_dict, reward, terminated, truncated, info = env.step(actions)

        # Episode done
        if final_dist is None:
            final_dist = dist

        metrics = {
            "episode": ep_idx,
            "success": success,
            "final_distance": float(final_dist),
            "steps": step + 1,
        }
        all_metrics.append(metrics)

        if success:
            success_count += 1

        print(f"  Steps: {step + 1}, Success: {success}, Final dist: {final_dist:.4f}m")

        # Save video
        video_path = out_dir / f"ep{ep_idx:02d}.mp4"
        save_video(video_frames, video_path, args.fps)

    # Final results
    elapsed = time.time() - start_time
    success_rate = success_count / len(all_metrics) if all_metrics else 0.0
    avg_dist = np.mean([m["final_distance"] for m in all_metrics]) if all_metrics else 0.0
    avg_steps = np.mean([m["steps"] for m in all_metrics]) if all_metrics else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes: {len(all_metrics)}")
    print(f"  Success Rate: {success_rate * 100:.1f}% ({success_count}/{len(all_metrics)})")
    print(f"  Avg Final Distance: {avg_dist:.4f}m")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Total Time: {elapsed:.1f}s")
    print("=" * 60)

    # Save results
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "dataset": args.lerobot_dataset,
            "num_episodes": len(all_metrics),
            "horizon": args.horizon,
            "n_action_steps": n_action_steps,
            "success_radius": args.success_radius,
        },
        "summary": {
            "success_rate": success_rate,
            "avg_final_distance": avg_dist,
            "avg_steps": avg_steps,
            "total_successes": success_count,
        },
        "episodes": all_metrics,
        "time_s": elapsed,
    }

    results_file = out_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {results_file}")

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
