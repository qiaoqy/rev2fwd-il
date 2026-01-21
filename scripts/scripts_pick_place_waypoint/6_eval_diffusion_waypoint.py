#!/usr/bin/env python3
"""Step 6: Evaluate diffusion policy using waypoint-based execution.

This script implements a waypoint-based execution strategy for the diffusion
policy. Instead of executing actions step-by-step at a fixed rate, it:

1. Predicts a full action chunk (e.g., 8 waypoints)
2. Sends each waypoint as a target position to the robot
3. Waits until the robot reaches each waypoint (within threshold)
4. Only moves to the next waypoint after the current one is reached
5. Only re-infers a new action chunk after all waypoints are reached
6. Only updates the observation queue when waypoints are reached

This approach is more aligned with real robot execution where:
- Actions represent target positions, not velocities
- The robot controller handles the motion between waypoints
- Observations should reflect the state at meaningful milestones

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic waypoint evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_waypoint/6_eval_diffusion_waypoint.py \
    --checkpoint runs/diffusion_A_pick_place/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_pick_place/videos_waypoint \
    --num_episodes 5 --headless

# With custom waypoint threshold (default: 0.01m = 1cm)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_waypoint/6_eval_diffusion_waypoint.py \
    --checkpoint runs/diffusion_A_pick_place/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_pick_place/videos_waypoint \
    --position_threshold 0.02 --num_episodes 5 --headless

# With timeout (max steps per waypoint before forcing next)
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_waypoint/6_eval_diffusion_waypoint.py \
    --checkpoint runs/diffusion_A_pick_place/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_pick_place/videos_waypoint \
    --max_steps_per_waypoint 50 --num_episodes 5 --headless

CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place_waypoint/6_eval_diffusion_waypoint.py \
    --checkpoint runs/diffusion_A_mark/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_mark/videos_waypoint \
    --position_threshold 0.01 --max_steps_per_waypoint 150 \
    --num_episodes 1 --visualize_action_chunk --headless

=============================================================================
KEY DIFFERENCES FROM 5_eval_diffusion.py
=============================================================================
1. Uses WaypointExecutor instead of policy.select_action()
2. Executes each waypoint until reached (position threshold based)
3. Re-infers only after all n_action_steps waypoints are reached
4. Observation queue only updated at waypoint arrivals
5. Additional statistics: waypoints reached, timeouts, steps per waypoint

=============================================================================
NOTES
=============================================================================
- This execution mode may be slower (more sim steps per inference)
- But it's more robust to action timing mismatches
- Useful for testing if the action chunk represents valid waypoints
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
# Camera utilities (same as script 5)
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
    q_xyzw = rot.as_quat()
    qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
    return (qw, qx, qy, qz)


def create_target_markers(num_envs: int, device: str):
    """Create visualization markers for start and goal positions."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    
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


def update_target_markers(start_markers, goal_markers, start_xy: tuple, goal_xy: tuple, marker_z: float, env):
    """Update the positions of start and goal markers."""
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
    import isaaclab_tasks
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
        description="Evaluate diffusion policy with waypoint-based execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model",
        help="Path to LeRobot pretrained_model directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/diffusion_A/videos_waypoint",
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
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2000,
        help="Max simulation steps per episode (higher than before since waypoint exec is slower).",
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
        "--min_init_dist",
        type=float,
        default=0.15,
        help="Minimum initial distance from goal.",
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
        help="Number of waypoints in each action chunk.",
    )
    
    # Waypoint-specific arguments
    parser.add_argument(
        "--position_threshold",
        type=float,
        default=0.01,
        help="Position threshold (meters) to consider waypoint reached. Default: 0.01m (1cm).",
    )
    parser.add_argument(
        "--max_steps_per_waypoint",
        type=int,
        default=100,
        help="Maximum sim steps to wait for reaching a waypoint before timeout. Default: 100.",
    )
    parser.add_argument(
        "--min_steps_per_waypoint",
        type=int,
        default=3,
        help="Minimum sim steps at each waypoint before checking if reached. Default: 3.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Enable overfit testing mode.",
    )
    parser.add_argument(
        "--init_obj_xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Custom initial XY position for the object.",
    )
    parser.add_argument(
        "--visualize_action_chunk",
        action="store_true",
        help="Generate action chunk visualization videos showing predicted waypoints at each inference. "
             "Saves to {out_dir}/action_chunks/ with one frame per inference step.",
    )

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


# ---------------------------------------------------------------------------
# Policy loading helpers (same as script 5)
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
    
    # Check if obj_pose is included:
    # state_dim=15 means ee_pose(7) + obj_pose(7) + gripper(1)
    # state_dim=8 means ee_pose(7) + gripper(1)
    # state_dim=14 means ee_pose(7) + obj_pose(7) (legacy, no gripper)
    # state_dim=7 means ee_pose(7) only (legacy, no gripper)
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


def load_overfit_env_init(checkpoint_dir: str) -> Dict[str, Any] | None:
    """Load overfit environment initialization parameters."""
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
            return overfit_env_init
    
    print(f"[Overfit Mode] WARNING: overfit_env_init.json not found!")
    return None


def load_diffusion_policy(
    pretrained_dir: str,
    device: str,
    image_height: int = 128,
    image_width: int = 128,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
) -> Tuple[Any, Any, Any, int, int]:
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
    
    policy = DiffusionPolicy(cfg)
    
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
    
    actual_inference_steps = cfg.num_inference_steps if cfg.num_inference_steps else cfg.num_train_timesteps
    actual_n_action_steps = cfg.n_action_steps
    
    print(f"[load_policy] Policy loaded. (num_inference_steps={actual_inference_steps}, n_action_steps={actual_n_action_steps})")
    return policy, preprocessor, postprocessor, actual_inference_steps, actual_n_action_steps


# ---------------------------------------------------------------------------
# Waypoint-based episode execution
# ---------------------------------------------------------------------------
def run_episode_waypoint(
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
    overfit_env_init: dict | None = None,
    n_action_steps: int = 8,
    init_obj_xy: tuple | None = None,
    start_markers=None,
    goal_markers=None,
    marker_z: float = 0.002,
    # Waypoint-specific parameters
    position_threshold: float = 0.01,
    max_steps_per_waypoint: int = 100,
    min_steps_per_waypoint: int = 3,
    # Visualization
    action_chunk_visualizer=None,
) -> dict:
    """Run one episode using waypoint-based execution.
    
    Key differences from run_episode in script 5:
    1. Uses WaypointExecutor instead of policy.select_action()
    2. Sends waypoints and waits for them to be reached
    3. Only re-infers after all waypoints in chunk are reached
    4. Observation queue only updated at waypoint arrivals
    
    Args:
        env: Isaac Lab environment.
        policy: Diffusion policy.
        preprocessor: Input preprocessor pipeline.
        postprocessor: Output postprocessor pipeline.
        horizon: Maximum simulation steps per episode.
        writer: Video writer (or None).
        goal_xy: Goal XY position (plate center).
        success_radius: Success radius in meters.
        min_init_dist: Minimum initial distance from goal.
        max_reset_attempts: Max attempts to find valid initial position.
        has_wrist: Whether policy expects wrist camera input.
        include_obj_pose: Whether policy expects object pose in state.
        include_gripper: Whether policy expects gripper state in obs.
        overfit_env_init: Optional dict with initial poses for overfit testing.
        n_action_steps: Number of waypoints per action chunk.
        init_obj_xy: Optional custom initial object XY position.
        start_markers: Optional start position markers.
        goal_markers: Optional goal position markers.
        marker_z: Z height for markers.
        position_threshold: Threshold to consider waypoint reached.
        max_steps_per_waypoint: Max steps before forcing next waypoint.
        min_steps_per_waypoint: Min steps before checking waypoint reached.
        action_chunk_visualizer: Optional ActionChunkVisualizer for visualizing
            predicted action chunks at each inference step.
        
    Returns:
        Dictionary with episode statistics including waypoint info.
    """
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.eval.waypoint_executor import WaypointExecutor, WaypointConfig

    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None) if has_wrist else None
    device = env.unwrapped.device

    # Create waypoint executor
    waypoint_config = WaypointConfig(
        position_threshold=position_threshold,
        max_steps_per_waypoint=max_steps_per_waypoint,
        min_steps_per_waypoint=min_steps_per_waypoint,
    )
    executor = WaypointExecutor(config=waypoint_config, device=device)
    executor.set_policy(policy, preprocessor, postprocessor, n_action_steps)
    executor.reset()

    # Reset environment
    obs_dict, _ = env.reset()
    
    # Handle object initialization (same as script 5)
    if overfit_env_init is not None:
        print(f"  [Overfit Mode] Teleporting object to saved initial pose...")
        initial_obj_pose = torch.tensor(
            overfit_env_init["initial_obj_pose"], 
            dtype=torch.float32, 
            device=device
        ).unsqueeze(0)
        teleport_object_to_pose(env, initial_obj_pose, name="object")
        for _ in range(5):
            env.unwrapped.sim.step()
        init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        init_dist = np.linalg.norm(init_obj_pose[:2] - np.array(goal_xy))
    elif init_obj_xy is not None:
        print(f"  [Custom Init] Teleporting object to XY: [{init_obj_xy[0]:.3f}, {init_obj_xy[1]:.3f}]...")
        current_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        init_obj_z = 0.0022
        new_obj_pose = torch.tensor(
            [init_obj_xy[0], init_obj_xy[1], init_obj_z,
             current_obj_pose[3], current_obj_pose[4], current_obj_pose[5], current_obj_pose[6]],
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        teleport_object_to_pose(env, new_obj_pose, name="object")
        for _ in range(5):
            env.unwrapped.sim.step()
        init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        init_dist = np.linalg.norm(init_obj_pose[:2] - np.array(goal_xy))
    else:
        goal_xy_arr = np.array(goal_xy)
        for attempt in range(max_reset_attempts):
            if attempt > 0:
                obs_dict, _ = env.reset()
            init_obj_pose = get_object_pose_w(env)[0].cpu().numpy()
            init_obj_xy_arr = init_obj_pose[:2]
            init_dist = np.linalg.norm(init_obj_xy_arr - goal_xy_arr)
            if init_dist >= min_init_dist:
                break
    
    print(f"  Initial object XY: [{init_obj_pose[0]:.3f}, {init_obj_pose[1]:.3f}]")
    print(f"  Goal XY: [{goal_xy[0]:.3f}, {goal_xy[1]:.3f}]")
    print(f"  Initial distance: {init_dist:.3f}m")
    
    # Update visual markers
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
    success_step = None
    last_action = None
    final_dist = None
    last_gripper_state = 1.0  # Initialize to open gripper (+1)

    for t in range(horizon):
        steps = t + 1
        
        # Progress logging
        if t % 100 == 0:
            stats = executor.get_statistics()
            print(f"[Step {t+1}/{horizon}] inferences={stats['inference_count']}, "
                  f"waypoints={stats['waypoints_completed']}, "
                  f"timeouts={stats['waypoints_timeout']}", flush=True)

        # Acquire camera images
        table_rgb = table_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        table_rgb_np = table_rgb.cpu().numpy().astype(np.uint8)
        table_rgb_frame = table_rgb_np[0]
        
        if wrist_camera is not None:
            wrist_rgb = wrist_camera.data.output["rgb"]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]
            wrist_rgb_np = wrist_rgb.cpu().numpy().astype(np.uint8)
            wrist_rgb_frame = wrist_rgb_np[0]
            combined_frame = np.concatenate([table_rgb_frame, wrist_rgb_frame], axis=1)
        else:
            combined_frame = table_rgb_frame
        
        # Prepare observation for policy (without preprocessing - executor handles it)
        table_rgb_chw = torch.from_numpy(table_rgb_frame).float() / 255.0
        table_rgb_chw = table_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)
        
        ee_pose = get_ee_pose_w(env)[0:1]
        
        # Build observation.state based on whether obj_pose and gripper are included
        if include_obj_pose:
            obj_pose_tensor = get_object_pose_w(env)[0:1]  # (1, 7)
            if include_gripper:
                gripper_tensor = torch.tensor([[last_gripper_state]], device=device)  # (1, 1)
                state = torch.cat([ee_pose, obj_pose_tensor, gripper_tensor], dim=-1)  # (1, 15)
            else:
                state = torch.cat([ee_pose, obj_pose_tensor], dim=-1)  # (1, 14)
        else:
            if include_gripper:
                gripper_tensor = torch.tensor([[last_gripper_state]], device=device)  # (1, 1)
                state = torch.cat([ee_pose, gripper_tensor], dim=-1)  # (1, 8)
            else:
                state = ee_pose  # (1, 7)
        
        policy_inputs: Dict[str, torch.Tensor] = {
            "observation.image": table_rgb_chw,
            "observation.state": state,
        }
        
        if wrist_camera is not None:
            wrist_rgb_chw = torch.from_numpy(wrist_rgb_frame).float() / 255.0
            wrist_rgb_chw = wrist_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)
            policy_inputs["observation.wrist_image"] = wrist_rgb_chw
        
        # Get current EE pose for waypoint checking
        current_ee_pose = ee_pose[0].cpu().numpy()
        
        # Execute waypoint step
        with torch.no_grad():
            action_np, info = executor.step(policy_inputs, current_ee_pose)
        
        # Log inference events and visualize action chunk
        if info['inference_this_step']:
            print(f"  [t={t}] New action chunk generated (inference #{info['inference_count']})")
            
            # Add frame to action chunk visualizer
            if action_chunk_visualizer is not None and 'action_chunk' in info:
                action_chunk_raw = info['action_chunk']  # (n_action_steps, action_dim)
                wrist_img = wrist_rgb_frame if wrist_camera is not None else None
                action_chunk_visualizer.add_frame(
                    ee_pose_raw=current_ee_pose[:3],  # XYZ only
                    ee_pose_norm=current_ee_pose[:3],  # For waypoint, we use raw (no separate norm)
                    action_chunk_norm=action_chunk_raw[:, :3],  # XYZ only (n_steps, 3)
                    action_chunk_raw=action_chunk_raw[:, :3],   # XYZ only
                    gt_chunk_norm=None,  # No ground truth during evaluation
                    gt_chunk_raw=None,
                    table_image=table_rgb_frame,
                    wrist_image=wrist_img,
                )
                
        if info.get('waypoint_reached', False):
            timeout_str = " (TIMEOUT)" if info.get('reached_by_timeout', False) else ""
            steps_taken = info.get('steps_taken_for_waypoint', info['steps_at_waypoint'])
            print(f"  [t={t}] Waypoint {info['waypoint_idx']-1} reached{timeout_str} in {steps_taken} steps")
        
        # Convert action to tensor
        action = torch.from_numpy(action_np).float().to(device)
        last_action = action
        
        # Get poses for text overlay
        ee_pose_for_text = current_ee_pose
        obj_pose_for_text = get_object_pose_w(env)[0].cpu().numpy()
        action_for_text = action_np
        
        # Add text overlay to video
        if writer is not None:
            frame_with_text = combined_frame.copy()
            ee_text = f"EE:  [{ee_pose_for_text[0]:.3f}, {ee_pose_for_text[1]:.3f}, {ee_pose_for_text[2]:.3f}]"
            obj_text = f"Obj: [{obj_pose_for_text[0]:.3f}, {obj_pose_for_text[1]:.3f}, {obj_pose_for_text[2]:.3f}]"
            act_text = f"WP{info['waypoint_idx']}: [{action_for_text[0]:.3f}, {action_for_text[1]:.3f}, {action_for_text[2]:.3f}] G:{action_for_text[-1]:.2f}"
            inf_text = f"Inf:{info['inference_count']} WP:{info['waypoint_idx']}/{n_action_steps}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            y_offset = 5
            for text in [ee_text, obj_text, act_text, inf_text]:
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(frame_with_text, (3, y_offset-2), (6 + text_w, y_offset + text_h + baseline), bg_color, -1)
                cv2.putText(frame_with_text, text, (4, y_offset + text_h), font, font_scale, color, thickness, cv2.LINE_AA)
                y_offset += text_h + baseline + 4
            
            writer.append_data(frame_with_text)

        # Prepare action for env
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and num_envs > 1:
            action = action.repeat(num_envs, 1)

        obs_dict, _, terminated, truncated, _ = env.step(action)
        
        # Update last_gripper_state from the action just executed
        last_gripper_state = action[0, -1].item()

        # Check success
        obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        obj_xy = obj_pose[:2]
        dist_to_goal = np.linalg.norm(obj_xy - np.array(goal_xy))
        final_dist = dist_to_goal
        
        if not success and dist_to_goal < success_radius:
            success = True
            success_step = t + 1
            print(f"  ✓ SUCCESS at step {t+1}! Distance: {dist_to_goal:.4f}m")
        
        if terminated[0] or truncated[0]:
            break

    # Get final waypoint statistics
    waypoint_stats = executor.get_statistics()
    
    return {
        "steps": steps,
        "success": success,
        "success_step": success_step,
        "final_dist": final_dist,
        "last_action": None if last_action is None else last_action.detach().cpu().numpy(),
        # Waypoint-specific stats
        "inference_count": waypoint_stats['inference_count'],
        "waypoints_completed": waypoint_stats['waypoints_completed'],
        "waypoints_reached": waypoint_stats['waypoints_reached'],
        "waypoints_timeout": waypoint_stats['waypoints_timeout'],
        "avg_steps_per_waypoint": waypoint_stats['avg_steps_per_waypoint'],
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

        # Load policy config
        print(f"\n{'='*60}")
        print("Loading policy configuration...")
        print(f"{'='*60}")
        policy_info = load_policy_config(args.checkpoint)
        has_wrist = policy_info["has_wrist"]
        include_obj_pose = policy_info["include_obj_pose"]
        include_gripper = policy_info["include_gripper"]
        
        print(f"  Policy checkpoint: {args.checkpoint}")
        print(f"  Requires wrist camera: {has_wrist}")
        print(f"  Includes obj_pose: {include_obj_pose}")
        print(f"  Includes gripper: {include_gripper}")
        
        if policy_info["image_shape"] is not None:
            policy_h, policy_w = policy_info["image_shape"][1], policy_info["image_shape"][2]
            if policy_h != args.image_height or policy_w != args.image_width:
                args.image_height = policy_h
                args.image_width = policy_w
        print(f"{'='*60}\n")

        # Create environment
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

        print("[DEBUG] Env created")

        table_camera = env.unwrapped.scene.sensors.get("table_cam", None)
        if table_camera is None:
            raise RuntimeError("Camera sensor 'table_cam' not found!")
        
        wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None)
        if has_wrist and wrist_camera is None:
            raise RuntimeError("Policy expects wrist camera but not found in env!")
        
        # Create visual markers
        start_markers, goal_markers, marker_z = create_target_markers(num_envs=1, device=device)

        # Load policy
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

        # Load overfit init if needed
        overfit_env_init = None
        if args.overfit:
            overfit_env_init = load_overfit_env_init(args.checkpoint)

        goal_xy = (0.5, 0.0)
        success_radius = 0.05
        min_init_dist = args.min_init_dist

        print(f"\n{'='*60}")
        print(f"Waypoint-Based Evaluation Settings:")
        print(f"  Goal XY: {goal_xy}")
        print(f"  Success radius: {success_radius}m")
        print(f"  Min initial distance: {min_init_dist}m")
        print(f"  Horizon: {args.horizon} sim steps")
        print(f"  Num episodes: {args.num_episodes}")
        print(f"  N action steps (waypoints per chunk): {n_action_steps}")
        print(f"  Position threshold: {args.position_threshold}m")
        print(f"  Max steps per waypoint: {args.max_steps_per_waypoint}")
        print(f"  Min steps per waypoint: {args.min_steps_per_waypoint}")
        print(f"{'='*60}")

        stats = []
        video_paths = []
        action_chunk_video_paths = []
        
        # Create action chunks directory if visualization is enabled
        action_chunks_dir = None
        if args.visualize_action_chunk:
            action_chunks_dir = out_dir / "action_chunks"
            action_chunks_dir.mkdir(parents=True, exist_ok=True)
            print(f"Action chunk visualizations will be saved to: {action_chunks_dir}/")
        
        for ep in range(args.num_episodes):
            print(f"\nEpisode {ep+1}/{args.num_episodes}")
            
            video_path = out_dir / f"ep{ep}.mp4"
            writer = imageio.get_writer(video_path, fps=args.fps)
            
            # Create action chunk visualizer for this episode if enabled
            action_chunk_visualizer = None
            if args.visualize_action_chunk:
                from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer
                action_chunk_visualizer = ActionChunkVisualizer(
                    output_dir=action_chunks_dir,
                    step_id=ep,  # Use episode id as step_id
                    fps=args.fps,
                )
            
            result = run_episode_waypoint(
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
                overfit_env_init=overfit_env_init,
                n_action_steps=n_action_steps,
                init_obj_xy=tuple(args.init_obj_xy) if args.init_obj_xy is not None else None,
                start_markers=start_markers,
                goal_markers=goal_markers,
                marker_z=marker_z,
                position_threshold=args.position_threshold,
                max_steps_per_waypoint=args.max_steps_per_waypoint,
                min_steps_per_waypoint=args.min_steps_per_waypoint,
                action_chunk_visualizer=action_chunk_visualizer,
            )
            
            writer.close()
            
            # Rename video based on success
            suffix = "success" if result['success'] else "failed"
            new_video_path = out_dir / f"ep{ep}_{suffix}.mp4"
            video_path.rename(new_video_path)
            video_paths.append(new_video_path)
            
            # Generate action chunk video if enabled
            if action_chunk_visualizer is not None:
                action_chunk_video_path = action_chunk_visualizer.generate_video(
                    filename_prefix=f"eval_action_chunk_{suffix}"
                )
                if action_chunk_video_path:
                    action_chunk_video_paths.append(action_chunk_video_path)
            
            stats.append(result)
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"  Result: {status} | steps={result['steps']} | final_dist={result['final_dist']:.4f}m")
            print(f"  Waypoint stats: inferences={result['inference_count']}, "
                  f"waypoints={result['waypoints_completed']}, "
                  f"reached={result['waypoints_reached']}, "
                  f"timeout={result['waypoints_timeout']}, "
                  f"avg_steps={result['avg_steps_per_waypoint']:.1f}")
            print(f"  Video saved: {new_video_path}")
            if action_chunk_visualizer is not None and action_chunk_video_path:
                print(f"  Action chunk video saved: {action_chunk_video_path}")

        print(f"\nSaved {len(video_paths)} videos to {out_dir}/")
        if action_chunk_video_paths:
            print(f"Saved {len(action_chunk_video_paths)} action chunk videos to {action_chunks_dir}/")

        # Summary statistics
        num_success = sum(1 for s in stats if s["success"])
        avg_steps = np.mean([s["steps"] for s in stats])
        avg_dist = np.mean([s["final_dist"] for s in stats])
        avg_inferences = np.mean([s["inference_count"] for s in stats])
        avg_waypoints = np.mean([s["waypoints_completed"] for s in stats])
        total_timeouts = sum(s["waypoints_timeout"] for s in stats)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"  Success rate: {num_success}/{len(stats)} ({100*num_success/len(stats):.1f}%)")
        print(f"  Average sim steps: {avg_steps:.1f}")
        print(f"  Average final distance: {avg_dist:.4f}m")
        print(f"  Average inferences per episode: {avg_inferences:.1f}")
        print(f"  Average waypoints per episode: {avg_waypoints:.1f}")
        print(f"  Total waypoint timeouts: {total_timeouts}")
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
