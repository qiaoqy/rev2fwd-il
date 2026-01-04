#!/usr/bin/env python3
"""Evaluate and visualize the LeRobot Diffusion Policy in Isaac Lab.

This script loads a diffusion policy trained by ``31_train_A_diffusion.py``
and evaluates it on the forward pick-and-place task (cube starts at random
table position -> place to the plate center). A camera is injected into the
Isaac Lab scene (same as script 12) so the policy receives RGB + EE state.
The RGB stream is recorded to an MP4 video.

=============================================================================
OVERVIEW
=============================================================================
This script performs evaluation of a vision-based diffusion policy:
- Input: RGB image (128x128) + EE pose (7D: position + quaternion)
- Output: Action (8D: target EE pose + gripper command)
- Task: Pick cube from random table position and place at goal (plate center)

=============================================================================
USAGE EXAMPLES
=============================================================================
# Basic evaluation (headless mode, 5 episodes, auto-saves ep0.mp4 to ep4.mp4)
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A_2cam/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A_2cam/videos \
    --num_episodes 5 \
    --headless

# With GUI visualization (for debugging)
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A/videos

# Custom horizon and resolution
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model \
    --out_dir runs/diffusion_A/videos \
    --horizon 500 \
    --image_width 128 \
    --image_height 128 \
    --headless

# CUDA_VISIBLE_DEVICES=1,2,6 torchrun --nproc_per_node=3 scripts/31_train_A_diffusion.py \
#     --dataset data/A_forward_with_2images.npz \
#     --out runs/diffusion_A_2cam \
#     --batch_size 2048 \
#     --steps 200 \
#     --lr 0.0005
    
=============================================================================
CHECKPOINT STRUCTURE
=============================================================================
The checkpoint directory should contain:
    - config.json              Policy configuration
    - model.safetensors        Model weights
    - policy_preprocessor.json (optional) Preprocessor config
    - policy_postprocessor.json (optional) Postprocessor config
    - *_normalizer_*.safetensors (optional) Normalization stats

=============================================================================
NOTES
=============================================================================
- Video is encoded from camera frames (not viewer), works in headless mode.
- The policy internally handles input normalization via LeRobot's normalizers.
- Images are converted to float32 [0,1] before feeding to the policy.
- First inference step may be slow due to CUDA JIT compilation.
- Isaac Sim may hang during shutdown - use Ctrl+C if needed after video saves.
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
        default=300,
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

    # Isaac Lab AppLauncher flags (headless, device, etc.)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True  # required for headless camera rendering
    return args


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
    
    # Check if obj_pose is included (state_dim=14 means ee_pose(7) + obj_pose(7))
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
) -> Tuple[Any, Any, Any]:
    """Load LeRobot diffusion policy from checkpoint directory.
    
    Args:
        pretrained_dir: Path to the pretrained model directory.
        device: Device to load the model on.
        image_height: Image height (must match training data).
        image_width: Image width (must match training data).
        
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
    
    # Create DiffusionConfig
    print(f"[load_policy] Creating DiffusionConfig...", flush=True)
    cfg = DiffusionConfig(**config_dict)
    print(f"[load_policy] DiffusionConfig created", flush=True)

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
    
    print(f"[load_policy] Policy loading complete!", flush=True)
    return policy, None, None


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
) -> dict:
    """Run one episode, write frames to writer if provided, return summary.
    
    Args:
        env: Isaac Lab environment.
        policy: Diffusion policy.
        preprocessor: Optional preprocessor (unused).
        postprocessor: Optional postprocessor (unused).
        horizon: Maximum steps per episode.
        writer: Video writer (or None).
        goal_xy: Goal XY position (plate center).
        success_radius: Success radius in meters.
        min_init_dist: Minimum initial distance from goal (reject if closer).
        max_reset_attempts: Max attempts to find valid initial position.
        has_wrist: Whether the policy expects wrist camera input.
        include_obj_pose: Whether the policy expects object pose in state.
        
    Returns:
        Dictionary with episode statistics.
    """

    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w

    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors.get("wrist_cam", None) if has_wrist else None
    device = env.unwrapped.device

    # Reset until object is far enough from goal
    goal_xy_arr = np.array(goal_xy)
    for attempt in range(max_reset_attempts):
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
    
    steps = 0
    success = False
    last_action = None
    final_dist = None


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

        # EE state
        ee_pose = get_ee_pose_w(env)[0:1]
        
        # Build observation.state based on whether obj_pose is included
        if include_obj_pose:
            obj_pose = get_object_pose_w(env)[0:1]  # (1, 7)
            state = torch.cat([ee_pose, obj_pose], dim=-1)  # (1, 14)
        else:
            state = ee_pose  # (1, 7)

        policy_inputs: Dict[str, torch.Tensor] = {
            "observation.image": table_rgb_chw,
            "observation.state": state,
        }
        
        # Add wrist camera input if policy expects it
        if wrist_camera is not None:
            wrist_rgb_chw = torch.from_numpy(wrist_rgb_frame).float() / 255.0
            wrist_rgb_chw = wrist_rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)
            policy_inputs["observation.wrist_image"] = wrist_rgb_chw

        with torch.no_grad():
            action = policy.select_action(policy_inputs)

        action = action.to(device)
        last_action = action
        
        # Get poses and action for text overlay
        ee_pose_for_text = ee_pose[0].cpu().numpy()
        obj_pose_for_text = get_object_pose_w(env)[0].cpu().numpy()
        action_for_text = action.cpu().numpy().flatten()
        
        # Add text overlay with EE, object, and action XYZ coordinates
        if writer is not None:
            frame_with_text = combined_frame.copy()
            ee_text = f"EE:  [{ee_pose_for_text[0]:.3f}, {ee_pose_for_text[1]:.3f}, {ee_pose_for_text[2]:.3f}]"
            obj_text = f"Obj: [{obj_pose_for_text[0]:.3f}, {obj_pose_for_text[1]:.3f}, {obj_pose_for_text[2]:.3f}]"
            act_text = f"Act: [{action_for_text[0]:.3f}, {action_for_text[1]:.3f}, {action_for_text[2]:.3f}]"
            
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
            
            writer.append_data(frame_with_text)

        # Tile action for num_envs (vectorized env)
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and num_envs > 1:
            action = action.repeat(num_envs, 1)

        obs_dict, _, terminated, truncated, _ = env.step(action)

        # Check success: object XY distance to goal
        obj_pose = get_object_pose_w(env)[0].cpu().numpy()
        obj_xy = obj_pose[:2]
        dist_to_goal = np.linalg.norm(obj_xy - np.array(goal_xy))
        final_dist = dist_to_goal
        
        if dist_to_goal < success_radius:
            success = True
            print(f"  SUCCESS at step {t+1}! Distance: {dist_to_goal:.4f}m")
            break
            
        # Early termination (but not success)
        if terminated[0] or truncated[0]:
            break

    return {
        "steps": steps,
        "success": success,
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
        
        print(f"  Policy checkpoint: {args.checkpoint}")
        print(f"  Expected image shape: {policy_info['image_shape']} (C, H, W)")
        print(f"  Expected state dim: {policy_info['state_dim']}")
        print(f"  Expected action dim: {policy_info['action_dim']}")
        print(f"  Requires wrist camera: {has_wrist}")
        print(f"  Includes obj_pose in state: {include_obj_pose}")
        
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
        # Step 3: Load full policy model
        # =====================================================================
        print("Loading diffusion policy weights...")
        policy, preprocessor, postprocessor = load_diffusion_policy(
            args.checkpoint,
            device,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        policy.eval()
        print("Policy loaded.")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"{'='*60}")

        stats = []
        video_paths = []
        for ep in range(args.num_episodes):
            print(f"\nEpisode {ep+1}/{args.num_episodes}")
            
            # Create a separate video writer for each episode
            video_path = out_dir / f"ep{ep}.mp4"
            writer = imageio.get_writer(video_path, fps=args.fps)
            
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
            )
            
            writer.close()
            video_paths.append(video_path)
            
            stats.append(result)
            status = "SUCCESS" if result['success'] else "FAILED"
            print(f"  Result: {status} | steps={result['steps']} | final_dist={result['final_dist']:.4f}m")
            print(f"  Video saved: {video_path}")

        print(f"\nSaved {len(video_paths)} videos to {out_dir}/")

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

    finally:
        print("Closing simulation app...", flush=True)
        simulation_app.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
