#!/usr/bin/env python3
"""Step 6: Alternating test A→B→A→B... loop with rollout data collection.

This script executes an alternating loop of Task A and Task B policies on a
single robot arm in one Isaac environment, WITHOUT resetting between tasks.
It collects rollout data for both tasks that can be used for finetuning.

=============================================================================
OVERVIEW
=============================================================================
Task A: Pick object from arbitrary position on table → Place at fixed target (ring center)
Task B: Pick object from fixed target → Place at arbitrary position on table

The key insight is that when Task A completes successfully, the object is at
the fixed target position - exactly where Task B starts. Similarly, when Task B
completes, the object is at some position on the table - ready for Task A.

=============================================================================
SUCCESS CRITERIA
=============================================================================
Task A Success:
    - Object z-position > HEIGHT_THRESHOLD (object lifted)
    - Gripper state < 0.5 (gripper closed, holding object)
    - Object XY within DISTANCE_THRESHOLD of goal

Task B Success:
    - Distance(object_pos, target_pos) < DISTANCE_THRESHOLD (object placed)
    - Gripper state > 0.5 (gripper open, object released)

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic usage
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/PP_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/PP_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_circle_iter1.npz \
    --out_B data/rollout_B_circle_iter1.npz \
    --max_cycles 5 --save_video --visualize_action_chunk --action_chunk 16 --headless

# With custom thresholds
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_iter1.npz \
    --out_B data/rollout_B_iter1.npz \
    --max_cycles 100 \
    --height_threshold 0.12 \
    --distance_threshold 0.05 \
    --save_video --headless

# With action chunk visualization
CUDA_VISIBLE_DEVICES=0 python scripts/scripts_pick_place/6_test_alternating.py \
    --policy_A runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model \
    --policy_B runs/diffusion_B_circle/checkpoints/checkpoints/last/pretrained_model \
    --out_A data/rollout_A_circle_iter1.npz \
    --out_B data/rollout_B_circle_iter1.npz \
    --max_cycles 10 --save_video --visualize_action_chunk --action_chunk 16 --headless

=============================================================================
VISUALIZATION OPTIONS
=============================================================================
--visualize_action_chunk:  Per-inference visualization with input (camera images,
                           EE pose) and output (predicted action chunk XYZ curves).
                           Generates separate videos for Task A and Task B.
--action_chunk_out_dir:    Custom output directory for action chunk videos.
                           Default: {out_A parent}/action_chunks/

=============================================================================
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run alternating A→B→A→B... test and collect rollout data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Policy checkpoints
    parser.add_argument(
        "--policy_A",
        type=str,
        required=True,
        help="Path to Task A policy checkpoint (e.g., runs/diffusion_A_circle/checkpoints/checkpoints/last/pretrained_model).",
    )
    parser.add_argument(
        "--policy_B",
        type=str,
        required=True,
        help="Path to Task B policy checkpoint.",
    )

    # Output paths
    parser.add_argument(
        "--out_A",
        type=str,
        default="data/rollout_A_iter.npz",
        help="Output path for Task A rollout data.",
    )
    parser.add_argument(
        "--out_B",
        type=str,
        default="data/rollout_B_iter.npz",
        help="Output path for Task B rollout data.",
    )

    # Test parameters
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=50,
        help="Maximum number of complete A→B cycles to attempt. Default: 50.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="Maximum steps per task attempt.",
    )
    parser.add_argument(
        "--height_threshold",
        type=float,
        default=0.15,
        help="Minimum object z-position to consider lifted. Default: 0.15m.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.05,
        help="Maximum distance from target for success. Default: 0.05m.",
    )

    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab Gym task ID.",
    )
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
        "--goal_xy",
        type=float,
        nargs=2,
        default=[0.5, 0.0],
        help="Goal XY position (plate center). Default: 0.5 0.0.",
    )

    # Action chunk settings
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Number of action steps to execute per inference. Default: use training config.",
    )

    # Video saving
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video of the test execution.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Output path for video. Default: derived from out_A path.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Video frame rate. Default: 30.",
    )
    
    # Action chunk visualization
    parser.add_argument(
        "--visualize_action_chunk",
        action="store_true",
        help="Generate action chunk visualization videos showing model input and predicted 16-step chunk. "
             "Saves separate videos for Task A and Task B.",
    )
    parser.add_argument(
        "--action_chunk_out_dir",
        type=str,
        default=None,
        help="Output directory for action chunk videos. Default: derived from out_A path.",
    )

    # Isaac Lab AppLauncher arguments
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True  # Required for headless camera rendering
    return args


# ---------------------------------------------------------------------------
# Camera and environment utilities (from 5_eval_diffusion.py)
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


def create_target_markers(num_envs: int, device: str):
    """Create visualization markers for place target and goal positions.
    
    Creates two sets of flat cylinder markers on the table surface:
    - Red markers: Place target positions (where Task B will place the object)
    - Green markers: Goal positions (fixed at plate center)
    
    These are visual-only markers with no physics interaction.
    
    Args:
        num_envs: Number of parallel environments.
        device: Torch device string.
        
    Returns:
        Tuple of (place_markers, goal_markers, marker_z) VisualizationMarkers objects.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    
    # Marker parameters
    marker_radius = 0.05  # 5cm radius
    marker_height = 0.002  # 2mm height (flat disk)
    table_z = 0.0  # Table surface height
    marker_z = table_z + marker_height / 2 + 0.001  # Slightly above table
    
    # Red marker for place target positions (Task B target)
    place_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PlaceMarkers",
        markers={
            "place": sim_utils.CylinderCfg(
                radius=marker_radius,
                height=marker_height,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # Red
                ),
            ),
        },
    )
    place_markers = VisualizationMarkers(place_marker_cfg)
    
    # Green marker for goal positions (fixed at plate center)
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
    
    return place_markers, goal_markers, marker_z


def update_target_markers(
    place_markers,
    goal_markers,
    place_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    marker_z: float,
    env,
):
    """Update the positions of place (red) and goal (green) markers.
    
    This is the unified update function that updates both markers at once,
    consistent with scripts 1 and 5.
    
    Args:
        place_markers: VisualizationMarkers for place positions (red).
        goal_markers: VisualizationMarkers for goal positions (green).
        place_xy: Tuple (x, y) for the place target position.
        goal_xy: Tuple (x, y) for the goal position (plate center).
        marker_z: Z height for markers.
        env: Isaac Lab environment (for env_origins).
    """
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    env_origins = env.unwrapped.scene.env_origins  # (num_envs, 3)
    
    # Convert to Python float to handle numpy.float32 types
    place_x = float(place_xy[0])
    place_y = float(place_xy[1])
    goal_x = float(goal_xy[0])
    goal_y = float(goal_xy[1])
    
    # Build place marker positions (red - Task B target)
    place_positions = torch.zeros((num_envs, 3), device=device)
    place_positions[:, 0] = place_x
    place_positions[:, 1] = place_y
    place_positions[:, 2] = marker_z
    place_positions_w = place_positions + env_origins
    
    # Build goal marker positions (green - fixed at plate center)
    goal_positions = torch.zeros((num_envs, 3), device=device)
    goal_positions[:, 0] = goal_x
    goal_positions[:, 1] = goal_y
    goal_positions[:, 2] = marker_z
    goal_positions_w = goal_positions + env_origins
    
    # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_envs, 1)
    
    # Update marker visualizations
    place_markers.visualize(place_positions_w, identity_quat)
    goal_markers.visualize(goal_positions_w, identity_quat)


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
    """Create Isaac Lab gym environment with camera sensors."""
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
# Policy loading utilities (from 5_eval_diffusion.py)
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
) -> Tuple[Any, Any, Any, int, int]:
    """Load LeRobot diffusion policy from checkpoint directory."""
    import json
    from pathlib import Path

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
    for key in ["crop_shape", "optimizer_betas", "down_dims"]:
        if key in config_dict and isinstance(config_dict[key], list):
            config_dict[key] = tuple(config_dict[key])

    # Override settings if specified
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

    actual_inference_steps = cfg.num_inference_steps or cfg.num_train_timesteps
    actual_n_action_steps = cfg.n_action_steps

    return policy, preprocessor, postprocessor, actual_inference_steps, actual_n_action_steps


# ---------------------------------------------------------------------------
# Alternating Tester Class
# ---------------------------------------------------------------------------
class AlternatingTester:
    """Execute alternating A/B policies and collect rollout data."""

    def __init__(
        self,
        env,
        policy_A,
        preprocessor_A,
        postprocessor_A,
        policy_B,
        preprocessor_B,
        postprocessor_B,
        n_action_steps_A: int,
        n_action_steps_B: int,
        goal_xy: Tuple[float, float] = (0.5, 0.0),
        height_threshold: float = 0.15,
        distance_threshold: float = 0.05,
        horizon: int = 400,
        has_wrist_A: bool = True,
        has_wrist_B: bool = True,
        include_obj_pose_A: bool = True,
        include_obj_pose_B: bool = True,
        include_gripper_A: bool = True,
        include_gripper_B: bool = True,
    ):
        """Initialize the alternating tester."""
        self.env = env
        self.device = env.unwrapped.device

        # Task A policy (pick from table → place at goal)
        self.policy_A = policy_A
        self.preprocessor_A = preprocessor_A
        self.postprocessor_A = postprocessor_A
        self.n_action_steps_A = n_action_steps_A
        self.has_wrist_A = has_wrist_A
        self.include_obj_pose_A = include_obj_pose_A
        self.include_gripper_A = include_gripper_A

        # Task B policy (pick from goal → place on table)
        self.policy_B = policy_B
        self.preprocessor_B = preprocessor_B
        self.postprocessor_B = postprocessor_B
        self.n_action_steps_B = n_action_steps_B
        self.has_wrist_B = has_wrist_B
        self.include_obj_pose_B = include_obj_pose_B
        self.include_gripper_B = include_gripper_B

        # Task parameters
        self.goal_xy = np.array(goal_xy)
        self.height_threshold = height_threshold
        self.distance_threshold = distance_threshold
        self.horizon = horizon

        # Get camera references
        self.table_camera = env.unwrapped.scene.sensors["table_cam"]
        self.wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None)

        # Storage for collected episodes
        self.episodes_A: list[dict] = []
        self.episodes_B: list[dict] = []
        
        # Current place target for Task B (random position on table)
        self.current_place_xy: Tuple[float, float] | None = None
        
        # Visualization markers
        self.place_markers = None
        self.goal_markers = None
        self.marker_z = None
        
        # Random number generator for sampling place positions
        self.rng = np.random.default_rng()
        
        # Track current gripper state (starts open)
        self.current_gripper_state = 1.0  # 1.0 = open, 0.0 = closed
        
        # Video recording
        self.video_frames: list[np.ndarray] = []
        
        # Action chunk visualization
        self.action_chunk_visualizer_A = None
        self.action_chunk_visualizer_B = None

    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Get current observation (images, ee_pose, obj_pose, gripper_state)."""
        from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w

        # Get table camera RGB
        table_rgb = self.table_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        table_rgb_np = table_rgb.cpu().numpy().astype(np.uint8)[0]  # (H, W, 3)

        # Get wrist camera RGB
        wrist_rgb_np = None
        if self.wrist_camera is not None:
            wrist_rgb = self.wrist_camera.data.output["rgb"]
            if wrist_rgb.shape[-1] > 3:
                wrist_rgb = wrist_rgb[..., :3]
            wrist_rgb_np = wrist_rgb.cpu().numpy().astype(np.uint8)[0]

        # Get poses
        ee_pose = get_ee_pose_w(self.env)[0].cpu().numpy()
        obj_pose = get_object_pose_w(self.env)[0].cpu().numpy()
        
        # Get gripper state (tracked from last action)
        gripper_state = self.current_gripper_state

        return table_rgb_np, wrist_rgb_np, ee_pose, obj_pose, gripper_state

    def _prepare_policy_input(
        self,
        table_rgb: np.ndarray,
        wrist_rgb: np.ndarray | None,
        ee_pose: np.ndarray,
        obj_pose: np.ndarray,
        gripper_state: float,
        include_obj_pose: bool,
        include_gripper: bool,
        has_wrist: bool,
    ) -> Dict[str, torch.Tensor]:
        """Prepare input for policy inference.
        
        Args:
            table_rgb: Table camera RGB image (H, W, 3).
            wrist_rgb: Wrist camera RGB image (H, W, 3) or None.
            ee_pose: End-effector pose (7,).
            obj_pose: Object pose (7,).
            gripper_state: Gripper state (1.0=open, 0.0=closed).
            include_obj_pose: Whether to include obj_pose in state.
            include_gripper: Whether to include gripper_state in state.
            has_wrist: Whether policy expects wrist camera input.
        """
        # Convert table image to float32 [0, 1] and BCHW format
        table_rgb_chw = torch.from_numpy(table_rgb).float() / 255.0
        table_rgb_chw = table_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Build observation.state based on include_obj_pose and include_gripper:
        # state_dim=7:  ee_pose(7) only
        # state_dim=8:  ee_pose(7) + gripper(1)
        # state_dim=14: ee_pose(7) + obj_pose(7)
        # state_dim=15: ee_pose(7) + obj_pose(7) + gripper(1)
        ee_pose_t = torch.from_numpy(ee_pose).float().unsqueeze(0).to(self.device)
        state_parts = [ee_pose_t]
        if include_obj_pose:
            obj_pose_t = torch.from_numpy(obj_pose).float().unsqueeze(0).to(self.device)
            state_parts.append(obj_pose_t)
        if include_gripper:
            gripper_t = torch.tensor([[gripper_state]], dtype=torch.float32, device=self.device)
            state_parts.append(gripper_t)
        state = torch.cat(state_parts, dim=-1)

        policy_inputs = {
            "observation.image": table_rgb_chw,
            "observation.state": state,
        }

        # Add wrist image if available and policy expects it
        if wrist_rgb is not None and has_wrist:
            wrist_rgb_chw = torch.from_numpy(wrist_rgb).float() / 255.0
            wrist_rgb_chw = wrist_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(self.device)
            policy_inputs["observation.wrist_image"] = wrist_rgb_chw

        return policy_inputs

    def _run_task(
        self,
        policy,
        preprocessor,
        postprocessor,
        n_action_steps: int,
        check_success_fn,
        task_name: str,
        include_obj_pose: bool,
        include_gripper: bool,
        has_wrist: bool,
        place_pose: np.ndarray | None = None,
        goal_pose: np.ndarray | None = None,
        action_chunk_visualizer=None,
    ) -> Tuple[dict, bool]:
        """Run a single task and collect trajectory data.
        
        Args:
            policy: The policy to use.
            preprocessor: Preprocessor for policy input.
            postprocessor: Postprocessor for policy output.
            n_action_steps: Number of action steps.
            check_success_fn: Function to check success.
            task_name: Name of the task for logging.
            include_obj_pose: Whether to include obj_pose in state.
            include_gripper: Whether to include gripper_state in state.
            has_wrist: Whether policy expects wrist camera input.
            place_pose: Target place position (7D pose) for this episode.
            goal_pose: Goal position (7D pose) for this episode.
            action_chunk_visualizer: Optional ActionChunkVisualizer for visualizing
                model input and predicted action chunks at each inference step.
        """
        from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w

        # Reset policy action queue for new task
        policy.reset()

        # Recording buffers
        images_list = []
        wrist_images_list = []
        ee_pose_list = []
        obj_pose_list = []
        action_list = []

        success = False
        success_step = None

        for t in range(self.horizon):
            # Get observation
            table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state = self._get_observation()

            # Record data
            images_list.append(table_rgb)
            if wrist_rgb is not None:
                wrist_images_list.append(wrist_rgb)
            ee_pose_list.append(ee_pose)
            obj_pose_list.append(obj_pose)
            
            # Record video frame (table camera view)
            self.video_frames.append(table_rgb.copy())

            # Prepare policy input with per-policy settings
            policy_inputs = self._prepare_policy_input(
                table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state,
                include_obj_pose=include_obj_pose,
                include_gripper=include_gripper,
                has_wrist=has_wrist,
            )
            
            # DEBUG: Print observation details on first step
            if t == 0:
                state_dim = policy_inputs['observation.state'].shape[-1]
                print(f"    [DEBUG] {task_name} Step 0: state_dim={state_dim}, "
                      f"include_obj_pose={include_obj_pose}, include_gripper={include_gripper}, has_wrist={has_wrist}")
                print(f"    [DEBUG] {task_name} Step 0: observation.state={policy_inputs['observation.state'][0, :7].cpu().numpy()}")
            
            # Store raw EE pose before normalization for action chunk visualization
            ee_pose_raw_np = ee_pose.copy()

            # Preprocess
            if preprocessor is not None:
                policy_inputs = preprocessor(policy_inputs)
            
            # Store normalized EE pose for action chunk visualization
            ee_pose_norm_np = policy_inputs['observation.state'][0, :7].cpu().numpy()
            
            # Determine if this is an inference step (start of new action chunk)
            is_inference_step = (t % n_action_steps == 0)

            # Get action from policy
            action_chunk_norm = None
            action_chunk_raw = None
            with torch.no_grad():
                action = policy.select_action(policy_inputs)
                raw_action = action.clone()
                
                # Get full action chunk for visualization at inference steps
                if is_inference_step and action_chunk_visualizer is not None:
                    try:
                        # For diffusion policy, prepare the batch correctly
                        n_obs_steps_required = policy.config.n_obs_steps if hasattr(policy, 'config') else 2
                        
                        inference_batch = {}
                        
                        # Copy state - replicate to fill n_obs_steps dimension
                        if 'observation.state' in policy_inputs:
                            state = policy_inputs['observation.state']  # (B, state_dim)
                            if state.dim() == 2:
                                state = state.unsqueeze(1).repeat(1, n_obs_steps_required, 1)
                            inference_batch['observation.state'] = state
                        
                        # Stack image features into observation.images
                        if hasattr(policy, 'config') and hasattr(policy.config, 'image_features'):
                            image_features_list = []
                            for key in policy.config.image_features:
                                if key in policy_inputs:
                                    img = policy_inputs[key]  # (B, C, H, W)
                                    if img.dim() == 4:
                                        img = img.unsqueeze(1).repeat(1, n_obs_steps_required, 1, 1, 1)
                                    image_features_list.append(img)
                            if image_features_list:
                                inference_batch['observation.images'] = torch.stack(image_features_list, dim=2)
                        
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
                            
                            # Unnormalize each action in the chunk
                            if postprocessor is not None:
                                unnorm_actions = []
                                for i in range(action_chunk_norm.shape[0]):
                                    single_action = torch.from_numpy(action_chunk_norm[i]).float().to(self.device)
                                    unnorm_action = postprocessor(single_action)
                                    unnorm_actions.append(unnorm_action.cpu().numpy())
                                action_chunk_raw = np.array(unnorm_actions)  # (horizon, action_dim)
                            else:
                                action_chunk_raw = action_chunk_norm.copy()
                                
                    except Exception as e:
                        import traceback
                        if t == 0:
                            print(f"[WARNING] Failed to get action chunk for visualization: {e}")
                            traceback.print_exc()

            # Postprocess (unnormalize)
            if postprocessor is not None:
                action = postprocessor(action)
            
            # DEBUG: Print action details on first step
            if t == 0:
                print(f"    [DEBUG] {task_name} Step 0: Raw action (normalized): {raw_action[0].cpu().numpy()}")
                print(f"    [DEBUG] {task_name} Step 0: Unnormalized action: {action[0].cpu().numpy()}")
            
            # Add frame to action chunk visualizer at inference steps
            if action_chunk_visualizer is not None and is_inference_step and action_chunk_norm is not None:
                action_chunk_visualizer.add_frame(
                    ee_pose_raw=ee_pose_raw_np[:3],  # XYZ only
                    ee_pose_norm=ee_pose_norm_np[:3],  # XYZ only
                    action_chunk_norm=action_chunk_norm[:, :3],  # XYZ only (horizon, 3)
                    action_chunk_raw=action_chunk_raw[:, :3] if action_chunk_raw is not None else None,
                    gt_chunk_norm=None,  # No ground truth during evaluation
                    gt_chunk_raw=None,
                    table_image=table_rgb,
                    wrist_image=wrist_rgb,
                )

            action_np = action[0].cpu().numpy()
            
            # =========================================================================
            # Force gripper open when XY aligned and Z is low enough
            # =========================================================================
            # Check if we should force gripper open:
            # - XY distance to target < threshold
            # - Object Z height < 0.15m (approaching table)
            # This helps ensure the object is released at the right position.
            from rev2fwd_il.sim.scene_api import get_object_pose_w as _get_obj_pose
            _obj_pose = _get_obj_pose(self.env)[0].cpu().numpy()
            _obj_z = _obj_pose[2]
            _obj_xy = _obj_pose[:2]
            
            # Determine target XY based on task
            if task_name == "Task A":
                _target_xy = self.goal_xy
            else:  # Task B
                _target_xy = np.array(self.current_place_xy) if self.current_place_xy is not None else self.goal_xy
            
            _dist_to_target = np.linalg.norm(_obj_xy - _target_xy)
            _xy_aligned = _dist_to_target < self.distance_threshold
            _z_low_enough = _obj_z < 0.15
            
            if _xy_aligned and _z_low_enough and action_np[7] < 0.5:
                # Force gripper open (1.0 = open)
                print(f"    [{task_name}] Step {t+1}: Forcing gripper open (XY dist={_dist_to_target:.4f}, Z={_obj_z:.4f})")
                action_np[7] = 1.0
            
            action_list.append(action_np)
            
            # Update gripper state from action (last dimension)
            self.current_gripper_state = float(action_np[7])

            # Execute action in environment
            action_t = torch.from_numpy(action_np).float().unsqueeze(0).to(self.device)
            num_envs = self.env.unwrapped.num_envs
            if action_t.ndim == 1:
                action_t = action_t.unsqueeze(0)
            if action_t.shape[0] == 1 and num_envs > 1:
                action_t = action_t.repeat(num_envs, 1)

            obs_dict, _, terminated, truncated, _ = self.env.step(action_t)

            # Check success
            if not success and check_success_fn():
                success = True
                success_step = t + 1
                print(f"    [{task_name}] ✓ SUCCESS at step {t+1}")
                # Success! Execute 20 more frames and record them.
                # IMPORTANT: Keep gripper open during all post-success frames
                print(f"    [{task_name}] Recording 20 additional frames after success (gripper forced open)...")
                for extra_t in range(20):
                    # Get observation
                    table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state = self._get_observation()
                    
                    # Record data
                    images_list.append(table_rgb)
                    if wrist_rgb is not None:
                        wrist_images_list.append(wrist_rgb)
                    ee_pose_list.append(ee_pose)
                    obj_pose_list.append(obj_pose)
                    
                    # Record video frame
                    self.video_frames.append(table_rgb.copy())
                    
                    # Prepare policy input
                    policy_inputs = self._prepare_policy_input(
                        table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state,
                        include_obj_pose=include_obj_pose,
                        include_gripper=include_gripper,
                        has_wrist=has_wrist,
                    )
                    
                    # Preprocess
                    if preprocessor is not None:
                        policy_inputs = preprocessor(policy_inputs)
                    
                    # Get action from policy
                    with torch.no_grad():
                        action = policy.select_action(policy_inputs)
                    
                    # Postprocess (unnormalize)
                    if postprocessor is not None:
                        action = postprocessor(action)
                    
                    action_np = action[0].cpu().numpy()
                    
                    # FORCE GRIPPER OPEN: After success, always keep gripper open
                    action_np[7] = 1.0
                    
                    action_list.append(action_np)
                    
                    # Update gripper state (always open after success)
                    self.current_gripper_state = 1.0
                    
                    # Execute action in environment (with forced open gripper)
                    action_t = torch.from_numpy(action_np).float().unsqueeze(0).to(self.device)
                    num_envs = self.env.unwrapped.num_envs
                    if action_t.ndim == 1:
                        action_t = action_t.unsqueeze(0)
                    if action_t.shape[0] == 1 and num_envs > 1:
                        action_t = action_t.repeat(num_envs, 1)
                    
                    self.env.step(action_t)
                
                # Now stop recording. Transition frames will be handled by the caller.
                break

            # Print progress
            if (t + 1) % 100 == 0:
                print(f"    [{task_name}] Step {t+1}/{self.horizon}")

        # Build episode dict
        episode_data = {
            "images": np.array(images_list, dtype=np.uint8),
            "ee_pose": np.array(ee_pose_list, dtype=np.float32),
            "obj_pose": np.array(obj_pose_list, dtype=np.float32),
            "action": np.array(action_list, dtype=np.float32),
            "success": success,
            "success_step": success_step,
        }
        if wrist_images_list:
            episode_data["wrist_images"] = np.array(wrist_images_list, dtype=np.uint8)
        
        # Add place_pose and goal_pose if provided (for compatibility with training data format)
        if place_pose is not None:
            episode_data["place_pose"] = place_pose
        if goal_pose is not None:
            episode_data["goal_pose"] = goal_pose

        return episode_data, success

    def check_task_A_success(self) -> bool:
        """Check if Task A succeeded (object placed on table at goal position).
        
        Success criteria:
        - Object XY position is near the goal position (green marker)
        - Object Z height < 0.15m (approaching or on table)
        - Gripper is open (object has been released) - forced open when conditions met
        
        Note: When XY is aligned and Z < 0.15m, the gripper is forced open in _run_task,
        so checking gripper state here confirms the release action was taken.
        """
        from rev2fwd_il.sim.scene_api import get_object_pose_w

        obj_pose = get_object_pose_w(self.env)[0].cpu().numpy()
        obj_z = obj_pose[2]
        obj_xy = obj_pose[:2]

        # Check object Z is low enough (approaching table or on table)
        is_z_low = obj_z < 0.15

        # Check distance to goal
        dist_to_goal = np.linalg.norm(obj_xy - self.goal_xy)
        is_at_goal = dist_to_goal < self.distance_threshold

        # Check gripper is open (object released)
        is_gripper_open = self.current_gripper_state > 0.5

        return is_z_low and is_at_goal and is_gripper_open

    def check_task_B_success(self) -> bool:
        """Check if Task B succeeded (object placed at target position).
        
        Success criteria:
        - Object XY position is near the target place position (red marker)
        - Object Z height < 0.15m (approaching or on table)
        - Gripper is open (object has been released) - forced open when conditions met
        """
        from rev2fwd_il.sim.scene_api import get_object_pose_w

        obj_pose = get_object_pose_w(self.env)[0].cpu().numpy()
        obj_z = obj_pose[2]
        obj_xy = obj_pose[:2]

        # Object Z should be low (approaching table or on table)
        is_z_low = obj_z < 0.15

        # Check if object is near the target place position (red marker)
        if self.current_place_xy is not None:
            target_xy = np.array(self.current_place_xy)
            dist_to_target = np.linalg.norm(obj_xy - target_xy)
            is_at_target = dist_to_target < self.distance_threshold
        else:
            # Fallback: just check if away from goal
            dist_from_goal = np.linalg.norm(obj_xy - self.goal_xy)
            is_at_target = dist_from_goal > 0.08

        # Check gripper is open (object released)
        is_gripper_open = self.current_gripper_state > 0.5

        return is_z_low and is_at_target and is_gripper_open
    
    def _sample_new_place_target(self) -> Tuple[float, float]:
        """Sample a new random place target position on the table.
        
        The position must be at least 0.1m away from the goal position.
        Uses the same range as task_spec.py (scripts 1 and 5).
        
        Returns:
            Tuple (x, y) for the new place target.
        """
        min_dist_from_goal = 0.1
        # Use same table bounds as task_spec.py for consistency with training data
        # task_spec: table_xy_min = (0.35, -0.25), table_xy_max = (0.65, 0.25)
        while True:
            # Sample random position within table bounds (matching training data)
            x = self.rng.uniform(0.37, 0.63)
            y = self.rng.uniform(-0.23, 0.23)
            
            # Check distance from goal
            dist = np.sqrt((x - self.goal_xy[0])**2 + (y - self.goal_xy[1])**2)
            if dist >= min_dist_from_goal:
                return (x, y)
    
    def _update_place_marker(self, place_xy: Tuple[float, float]):
        """Update the red place marker to a new position.
        
        Uses the unified update_target_markers function for consistency with
        scripts 1 and 5. This updates both markers but keeps goal marker fixed.
        """
        self.current_place_xy = place_xy
        if self.place_markers is not None:
            update_target_markers(
                self.place_markers,
                self.goal_markers,
                place_xy,
                tuple(self.goal_xy),
                self.marker_z,
                self.env,
            )
            print(f"    [Marker] Updated red place marker to: [{place_xy[0]:.3f}, {place_xy[1]:.3f}]")

    def _run_transition(self, n_frames: int, policy, preprocessor, postprocessor,
                         n_action_steps: int, include_obj_pose: bool, 
                         include_gripper: bool, has_wrist: bool, task_name: str) -> None:
        """Execute transition frames without recording data.
        
        This is used after a task succeeds to allow the robot to continue
        moving before switching to the next task.
        
        IMPORTANT: Gripper is forced open during all transition frames to ensure
        the object remains released.
        
        Args:
            n_frames: Number of frames to execute.
            policy: The policy to use.
            preprocessor: Preprocessor for policy input.
            postprocessor: Postprocessor for policy output.
            n_action_steps: Number of action steps per inference.
            include_obj_pose: Whether to include obj_pose in state.
            include_gripper: Whether to include gripper_state in state.
            has_wrist: Whether policy expects wrist camera input.
            task_name: Name of the task for logging.
        """
        print(f"    [{task_name}] Running {n_frames} transition frames (gripper forced open, not recording)...")
        
        for t in range(n_frames):
            # Get observation
            table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state = self._get_observation()
            
            # Record video frame (but not data)
            self.video_frames.append(table_rgb.copy())
            
            # Prepare policy input
            policy_inputs = self._prepare_policy_input(
                table_rgb, wrist_rgb, ee_pose, obj_pose, gripper_state,
                include_obj_pose=include_obj_pose,
                include_gripper=include_gripper,
                has_wrist=has_wrist,
            )
            
            # Preprocess
            if preprocessor is not None:
                policy_inputs = preprocessor(policy_inputs)
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(policy_inputs)
            
            # Postprocess (unnormalize)
            if postprocessor is not None:
                action = postprocessor(action)
            
            action_np = action[0].cpu().numpy()
            
            # FORCE GRIPPER OPEN: During transition, always keep gripper open
            action_np[7] = 1.0
            
            # Update gripper state (always open during transition)
            self.current_gripper_state = 1.0
            
            # Execute action in environment (with forced open gripper)
            action_t = torch.from_numpy(action_np).float().unsqueeze(0).to(self.device)
            num_envs = self.env.unwrapped.num_envs
            if action_t.ndim == 1:
                action_t = action_t.unsqueeze(0)
            if action_t.shape[0] == 1 and num_envs > 1:
                action_t = action_t.repeat(num_envs, 1)
            
            self.env.step(action_t)
        
        print(f"    [{task_name}] Transition complete.")

    def run_task_A(self) -> Tuple[dict, bool]:
        """Execute Task A (pick from table → place at goal)."""
        # For Task A: place_pose is the goal position (green marker)
        goal_pose = np.array([
            self.goal_xy[0], self.goal_xy[1], 0.055,
            1.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        return self._run_task(
            policy=self.policy_A,
            preprocessor=self.preprocessor_A,
            postprocessor=self.postprocessor_A,
            n_action_steps=self.n_action_steps_A,
            check_success_fn=self.check_task_A_success,
            task_name="Task A",
            include_obj_pose=self.include_obj_pose_A,
            include_gripper=self.include_gripper_A,
            has_wrist=self.has_wrist_A,
            place_pose=goal_pose,  # Task A places at goal
            goal_pose=goal_pose,
            action_chunk_visualizer=self.action_chunk_visualizer_A,
        )

    def run_task_B(self) -> Tuple[dict, bool]:
        """Execute Task B (pick from goal → place at red marker)."""
        # For Task B: place_pose is the current red marker position
        # goal_pose is the fixed green marker position
        goal_pose = np.array([
            self.goal_xy[0], self.goal_xy[1], 0.055,
            1.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        place_pose = None
        if self.current_place_xy is not None:
            place_pose = np.array([
                self.current_place_xy[0], self.current_place_xy[1], 0.055,
                1.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
        
        return self._run_task(
            policy=self.policy_B,
            preprocessor=self.preprocessor_B,
            postprocessor=self.postprocessor_B,
            n_action_steps=self.n_action_steps_B,
            check_success_fn=self.check_task_B_success,
            task_name="Task B",
            include_obj_pose=self.include_obj_pose_B,
            include_gripper=self.include_gripper_B,
            has_wrist=self.has_wrist_B,
            place_pose=place_pose,  # Task B places at red marker
            goal_pose=goal_pose,
            action_chunk_visualizer=self.action_chunk_visualizer_B,
        )

    def run_alternating_test(self, max_cycles: int) -> dict:
        """Run alternating A→B→A→B... test loop."""
        from rev2fwd_il.sim.scene_api import get_object_pose_w, teleport_object_to_pose

        consecutive_success = 0

        # First, reset environment and place object at random position
        print("Resetting environment for alternating test...")
        self.env.reset()
        
        # Create visualization markers
        print("Creating visualization markers...")
        self.place_markers, self.goal_markers, self.marker_z = create_target_markers(
            num_envs=1, device=self.device
        )
        
        # Sample initial place target (red marker) for Task B
        # This will be the first target position after Task A completes
        first_place_xy = self._sample_new_place_target()
        self.current_place_xy = first_place_xy
        
        # Initialize both markers using the unified function
        update_target_markers(
            self.place_markers,
            self.goal_markers,
            first_place_xy,
            tuple(self.goal_xy),
            self.marker_z,
            self.env,
        )
        print(f"  [Marker] Green goal marker at: [{self.goal_xy[0]:.3f}, {self.goal_xy[1]:.3f}]")
        print(f"  [Marker] Red place marker at: [{first_place_xy[0]:.3f}, {first_place_xy[1]:.3f}]")

        # Teleport object to the red marker position (initial position for Task A)
        init_x = first_place_xy[0]
        init_y = first_place_xy[1]
        init_pose = torch.tensor(
            [init_x, init_y, 0.022, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        teleport_object_to_pose(self.env, init_pose, name="object")

        # Settle
        zero_action = torch.zeros(1, self.env.action_space.shape[-1], device=self.device)
        for _ in range(10):
            self.env.step(zero_action)

        obj_pose = get_object_pose_w(self.env)[0].cpu().numpy()
        print(f"Initial object position: [{obj_pose[0]:.3f}, {obj_pose[1]:.3f}, {obj_pose[2]:.3f}]")

        for cycle in range(max_cycles):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{max_cycles}")
            print(f"{'='*50}")

            # Execute Task A (pick from table → place at goal)
            print(f"  Running Task A (pick → place at goal)...")
            print(f"    Target: green marker at [{self.goal_xy[0]:.3f}, {self.goal_xy[1]:.3f}]")
            ep_A, success_A = self.run_task_A()
            self.episodes_A.append(ep_A)

            if not success_A:
                print(f"  ✗ Task A FAILED at cycle {cycle + 1}")
                break
            
            # After Task A succeeds:
            # 1. Sample a NEW place target for Task B (update red marker)
            new_place_xy = self._sample_new_place_target()
            self._update_place_marker(new_place_xy)
            
            # 2. Run 100 transition frames with Policy A (not recording data)
            self._run_transition(
                n_frames=100,
                policy=self.policy_A,
                preprocessor=self.preprocessor_A,
                postprocessor=self.postprocessor_A,
                n_action_steps=self.n_action_steps_A,
                include_obj_pose=self.include_obj_pose_A,
                include_gripper=self.include_gripper_A,
                has_wrist=self.has_wrist_A,
                task_name="Task A (transition)",
            )

            # 3. Execute Task B (pick from goal → place at red marker)
            print(f"  Running Task B (pick from goal → place at red marker)...")
            print(f"    Target: red marker at [{self.current_place_xy[0]:.3f}, {self.current_place_xy[1]:.3f}]")
            ep_B, success_B = self.run_task_B()
            self.episodes_B.append(ep_B)

            if not success_B:
                print(f"  ✗ Task B FAILED at cycle {cycle + 1}")
                break
            
            # After Task B succeeds:
            # Run 100 transition frames with Policy B (not recording data)
            # Red marker position stays the same (will be used as start for next Task A)
            self._run_transition(
                n_frames=100,
                policy=self.policy_B,
                preprocessor=self.preprocessor_B,
                postprocessor=self.postprocessor_B,
                n_action_steps=self.n_action_steps_B,
                include_obj_pose=self.include_obj_pose_B,
                include_gripper=self.include_gripper_B,
                has_wrist=self.has_wrist_B,
                task_name="Task B (transition)",
            )

            consecutive_success += 1
            print(f"  ✓ Cycle {cycle + 1} complete! Consecutive successes: {consecutive_success}")

        return {
            "consecutive_success": consecutive_success,
            "total_A_episodes": len(self.episodes_A),
            "total_B_episodes": len(self.episodes_B),
            "A_success_rate": sum(1 for ep in self.episodes_A if ep["success"]) / max(1, len(self.episodes_A)),
            "B_success_rate": sum(1 for ep in self.episodes_B if ep["success"]) / max(1, len(self.episodes_B)),
        }

    def save_data(self, out_A: str, out_B: str) -> None:
        """Save collected rollout data to NPZ files."""
        out_A_path = Path(out_A)
        out_B_path = Path(out_B)
        out_A_path.parent.mkdir(parents=True, exist_ok=True)
        out_B_path.parent.mkdir(parents=True, exist_ok=True)

        # Filter to only successful episodes for training
        success_A = [ep for ep in self.episodes_A if ep["success"]]
        success_B = [ep for ep in self.episodes_B if ep["success"]]

        if success_A:
            np.savez_compressed(out_A, episodes=np.array(success_A, dtype=object))
            print(f"Saved {len(success_A)} successful Task A episodes to {out_A}")
        else:
            print(f"No successful Task A episodes to save.")

        if success_B:
            np.savez_compressed(out_B, episodes=np.array(success_B, dtype=object))
            print(f"Saved {len(success_B)} successful Task B episodes to {out_B}")
        else:
            print(f"No successful Task B episodes to save.")

    def save_video(self, video_path: str, fps: int = 30) -> None:
        """Save collected video frames to an MP4 file.
        
        Args:
            video_path: Output path for the video file.
            fps: Frame rate for the video.
        """
        if not self.video_frames:
            print("No video frames to save.")
            return
        
        import imageio
        
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8 if needed
        frames = [f.astype(np.uint8) if f.dtype != np.uint8 else f for f in self.video_frames]
        
        # Save video
        imageio.mimsave(str(video_path), frames, fps=fps)
        print(f"Saved video with {len(frames)} frames to {video_path}")
    
    def set_action_chunk_visualizers(self, visualizer_A, visualizer_B):
        """Set the action chunk visualizers for Task A and Task B.
        
        Args:
            visualizer_A: ActionChunkVisualizer for Task A.
            visualizer_B: ActionChunkVisualizer for Task B.
        """
        self.action_chunk_visualizer_A = visualizer_A
        self.action_chunk_visualizer_B = visualizer_B
    
    def generate_action_chunk_videos(self) -> Tuple[str, str]:
        """Generate action chunk visualization videos for both tasks.
        
        Returns:
            Tuple of (video_path_A, video_path_B) for the generated videos.
        """
        video_path_A = ""
        video_path_B = ""
        
        if self.action_chunk_visualizer_A is not None:
            video_path_A = self.action_chunk_visualizer_A.generate_video(
                filename_prefix="action_chunk_task_A"
            )
        
        if self.action_chunk_visualizer_B is not None:
            video_path_B = self.action_chunk_visualizer_B.generate_video(
                filename_prefix="action_chunk_task_B"
            )
        
        return video_path_A, video_path_B


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from rev2fwd_il.utils.seed import set_seed
        set_seed(args.seed)

        device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

        # =====================================================================
        # Load policy configs to check requirements
        # =====================================================================
        print(f"\n{'='*60}")
        print("Loading policy configurations...")
        print(f"{'='*60}")

        config_A = load_policy_config(args.policy_A)
        config_B = load_policy_config(args.policy_B)

        # Use OR logic only for environment setup (need wrist cam if either policy needs it)
        env_needs_wrist = config_A["has_wrist"] or config_B["has_wrist"]

        print(f"  Policy A: {args.policy_A}")
        print(f"    - state_dim: {config_A['state_dim']}, has_wrist: {config_A['has_wrist']}")
        print(f"    - include_obj_pose: {config_A['include_obj_pose']}, include_gripper: {config_A['include_gripper']}")
        print(f"  Policy B: {args.policy_B}")
        print(f"    - state_dim: {config_B['state_dim']}, has_wrist: {config_B['has_wrist']}")
        print(f"    - include_obj_pose: {config_B['include_obj_pose']}, include_gripper: {config_B['include_gripper']}")
        print(f"  Environment needs wrist camera: {env_needs_wrist}")

        # =====================================================================
        # Create environment
        # =====================================================================
        print(f"\n{'='*60}")
        print("Creating environment...")
        print(f"{'='*60}")

        env = make_env_with_camera(
            task_id=args.task,
            num_envs=1,
            device=device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=1000.0,  # Long episode to prevent auto-reset
            disable_terminations=True,
        )

        # =====================================================================
        # Load policies
        # =====================================================================
        print(f"\n{'='*60}")
        print("Loading policies...")
        print(f"{'='*60}")

        print("Loading Policy A...")
        policy_A, preprocessor_A, postprocessor_A, _, n_action_steps_A = load_diffusion_policy(
            args.policy_A, device,
            image_height=args.image_height,
            image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_A.eval()

        print("Loading Policy B...")
        policy_B, preprocessor_B, postprocessor_B, _, n_action_steps_B = load_diffusion_policy(
            args.policy_B, device,
            image_height=args.image_height,
            image_width=args.image_width,
            n_action_steps=args.n_action_steps,
        )
        policy_B.eval()

        # =====================================================================
        # Run alternating test
        # =====================================================================
        print(f"\n{'='*60}")
        print("Starting Alternating Test")
        print(f"{'='*60}")
        print(f"  Max cycles: {args.max_cycles}")
        print(f"  Horizon per task: {args.horizon}")
        print(f"  Height threshold: {args.height_threshold}m")
        print(f"  Distance threshold: {args.distance_threshold}m")
        print(f"  Goal XY: {args.goal_xy}")

        tester = AlternatingTester(
            env=env,
            policy_A=policy_A,
            preprocessor_A=preprocessor_A,
            postprocessor_A=postprocessor_A,
            policy_B=policy_B,
            preprocessor_B=preprocessor_B,
            postprocessor_B=postprocessor_B,
            n_action_steps_A=n_action_steps_A,
            n_action_steps_B=n_action_steps_B,
            goal_xy=tuple(args.goal_xy),
            height_threshold=args.height_threshold,
            distance_threshold=args.distance_threshold,
            horizon=args.horizon,
            has_wrist_A=config_A["has_wrist"],
            has_wrist_B=config_B["has_wrist"],
            include_obj_pose_A=config_A["include_obj_pose"],
            include_obj_pose_B=config_B["include_obj_pose"],
            include_gripper_A=config_A["include_gripper"],
            include_gripper_B=config_B["include_gripper"],
        )
        
        # Create action chunk visualizers if enabled
        if args.visualize_action_chunk:
            from rev2fwd_il.data.visualize_action_chunk import ActionChunkVisualizer
            
            # Determine output directory
            if args.action_chunk_out_dir is not None:
                action_chunk_dir = Path(args.action_chunk_out_dir)
            else:
                action_chunk_dir = Path(args.out_A).parent / "action_chunks"
            action_chunk_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Action chunk visualizations will be saved to: {action_chunk_dir}")
            
            visualizer_A = ActionChunkVisualizer(
                output_dir=action_chunk_dir,
                step_id=0,  # Use 0 as default step_id
                fps=args.video_fps,
            )
            visualizer_B = ActionChunkVisualizer(
                output_dir=action_chunk_dir,
                step_id=0,
                fps=args.video_fps,
            )
            tester.set_action_chunk_visualizers(visualizer_A, visualizer_B)

        start_time = time.time()
        results = tester.run_alternating_test(args.max_cycles)
        elapsed = time.time() - start_time

        # =====================================================================
        # Print results
        # =====================================================================
        print(f"\n{'='*60}")
        print("Alternating Test Results")
        print(f"{'='*60}")
        print(f"  Consecutive successes: {results['consecutive_success']}")
        print(f"  Total Task A episodes: {results['total_A_episodes']}")
        print(f"  Total Task B episodes: {results['total_B_episodes']}")
        print(f"  Task A success rate: {results['A_success_rate']:.1%}")
        print(f"  Task B success rate: {results['B_success_rate']:.1%}")
        print(f"  Total time: {elapsed:.1f}s")

        # =====================================================================
        # Save data
        # =====================================================================
        print(f"\n{'='*60}")
        print("Saving rollout data...")
        print(f"{'='*60}")
        tester.save_data(args.out_A, args.out_B)
        
        # =====================================================================
        # Save rollout statistics to JSON
        # =====================================================================
        print(f"\n{'='*60}")
        print("Saving rollout statistics...")
        print(f"{'='*60}")
        
        # Build detailed statistics
        import json
        from datetime import datetime
        
        rollout_stats = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "policy_A": args.policy_A,
                "policy_B": args.policy_B,
                "max_cycles": args.max_cycles,
                "horizon": args.horizon,
                "height_threshold": args.height_threshold,
                "distance_threshold": args.distance_threshold,
                "n_action_steps": args.n_action_steps,
                "goal_xy": args.goal_xy,
            },
            "summary": {
                "consecutive_successes": results['consecutive_success'],
                "total_task_A_episodes": results['total_A_episodes'],
                "total_task_B_episodes": results['total_B_episodes'],
                "task_A_success_count": sum(1 for ep in tester.episodes_A if ep["success"]),
                "task_B_success_count": sum(1 for ep in tester.episodes_B if ep["success"]),
                "task_A_success_rate": results['A_success_rate'],
                "task_B_success_rate": results['B_success_rate'],
                "total_elapsed_seconds": elapsed,
            },
            "episodes_A": [],
            "episodes_B": [],
        }
        
        # Add detailed per-episode statistics for Task A
        for i, ep in enumerate(tester.episodes_A):
            ep_stats = {
                "episode_index": i,
                "cycle": i + 1,  # 1-indexed cycle number
                "success": ep["success"],
                "success_step": ep.get("success_step"),
                "total_steps": len(ep["images"]),
                "total_actions": len(ep["action"]),
            }
            # Add initial and final positions if available
            if "ee_pose" in ep and len(ep["ee_pose"]) > 0:
                ep_stats["initial_ee_position"] = ep["ee_pose"][0][:3].tolist()
                ep_stats["final_ee_position"] = ep["ee_pose"][-1][:3].tolist()
            if "obj_pose" in ep and len(ep["obj_pose"]) > 0:
                ep_stats["initial_obj_position"] = ep["obj_pose"][0][:3].tolist()
                ep_stats["final_obj_position"] = ep["obj_pose"][-1][:3].tolist()
            if "place_pose" in ep and ep["place_pose"] is not None:
                ep_stats["target_place_position"] = ep["place_pose"][:3].tolist()
            rollout_stats["episodes_A"].append(ep_stats)
        
        # Add detailed per-episode statistics for Task B
        for i, ep in enumerate(tester.episodes_B):
            ep_stats = {
                "episode_index": i,
                "cycle": i + 1,  # 1-indexed cycle number
                "success": ep["success"],
                "success_step": ep.get("success_step"),
                "total_steps": len(ep["images"]),
                "total_actions": len(ep["action"]),
            }
            # Add initial and final positions if available
            if "ee_pose" in ep and len(ep["ee_pose"]) > 0:
                ep_stats["initial_ee_position"] = ep["ee_pose"][0][:3].tolist()
                ep_stats["final_ee_position"] = ep["ee_pose"][-1][:3].tolist()
            if "obj_pose" in ep and len(ep["obj_pose"]) > 0:
                ep_stats["initial_obj_position"] = ep["obj_pose"][0][:3].tolist()
                ep_stats["final_obj_position"] = ep["obj_pose"][-1][:3].tolist()
            if "place_pose" in ep and ep["place_pose"] is not None:
                ep_stats["target_place_position"] = ep["place_pose"][:3].tolist()
            rollout_stats["episodes_B"].append(ep_stats)
        
        # Save stats to JSON (same directory as out_A, with _stats.json suffix)
        stats_path = Path(args.out_A).with_suffix(".stats.json")
        with open(stats_path, 'w') as f:
            json.dump(rollout_stats, f, indent=2)
        print(f"  Saved rollout statistics to: {stats_path}")

        # =====================================================================
        # Save video
        # =====================================================================
        if args.save_video:
            print(f"\n{'='*60}")
            print("Saving video...")
            print(f"{'='*60}")
            # Derive video path from out_A if not specified
            if args.video_path is None:
                video_path = Path(args.out_A).with_suffix(".mp4")
            else:
                video_path = Path(args.video_path)
            tester.save_video(str(video_path), fps=args.video_fps)
        
        # =====================================================================
        # Generate action chunk visualization videos
        # =====================================================================
        if args.visualize_action_chunk:
            print(f"\n{'='*60}")
            print("Generating action chunk visualization videos...")
            print(f"{'='*60}")
            video_A, video_B = tester.generate_action_chunk_videos()
            if video_A:
                print(f"  Task A action chunk video saved: {video_A}")
            if video_B:
                print(f"  Task B action chunk video saved: {video_B}")

        # Cleanup
        env.close()

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
