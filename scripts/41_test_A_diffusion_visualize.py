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
# Basic evaluation (headless mode, 1 episode, save video)
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A/videos/ep0.mp4 \
    --headless

# Multiple episodes evaluation
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A/videos/ep0.mp4 \
    --num_episodes 10 \
    --headless

# With GUI visualization (for debugging)
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A/videos/ep0.mp4

# Custom horizon and resolution
CUDA_VISIBLE_DEVICES=1 python scripts/41_test_A_diffusion_visualize.py \
    --checkpoint runs/diffusion_A/checkpoints/checkpoints/last/pretrained_model \
    --out runs/diffusion_A/videos/ep0.mp4 \
    --horizon 500 \
    --image_width 128 \
    --image_height 128 \
    --headless

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
    """Inject a table-view camera into Isaac Lab env cfg."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    camera_eye = (1.6, 0.7, 0.8)
    camera_lookat = (0.4, 0.0, 0.2)
    camera_quat = compute_camera_quat_from_lookat(camera_eye, camera_lookat)

    env_cfg.scene.table_cam = TiledCameraCfg(
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
        offset=TiledCameraCfg.OffsetCfg(
            pos=camera_eye,
            rot=camera_quat,
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
        "--out",
        type=str,
        default="runs/diffusion_A/videos/ep0.mp4",
        help="Output MP4 path for the recorded episode.",
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
        default=1,
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

    # Isaac Lab AppLauncher flags (headless, device, etc.)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True  # required for headless camera rendering
    return args


# ---------------------------------------------------------------------------
# Policy loading helpers
# ---------------------------------------------------------------------------
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
) -> dict:
    """Run one episode, write frames to writer if provided, return summary."""

    from rev2fwd_il.sim.scene_api import get_ee_pose_w

    camera = env.unwrapped.scene.sensors["table_cam"]
    device = env.unwrapped.device

    obs_dict, _ = env.reset()
    steps = 0
    success = False
    last_action = None


    for t in range(horizon):
        steps = t + 1
        if t % 50 == 0:
            print(f"[Step {t+1}/{horizon}]", flush=True)

        # Acquire camera RGB (num_envs, H, W, 3) -> float32 BCHW
        rgb = camera.data.output["rgb"]
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        rgb_np = rgb.cpu().numpy().astype(np.uint8)
        rgb_frame = rgb_np[0]
        if writer is not None:
            writer.append_data(rgb_frame)

        # Convert to float32 [0, 1] and BCHW format for policy
        rgb_chw = torch.from_numpy(rgb_frame).float() / 255.0  # uint8 -> float [0,1]
        rgb_chw = rgb_chw.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)

        # EE state
        ee_pose = get_ee_pose_w(env)[0:1]

        policy_inputs: Dict[str, torch.Tensor] = {
            "observation.image": rgb_chw,
            "observation.state": ee_pose,
        }

        with torch.no_grad():
            action = policy.select_action(policy_inputs)

        action = action.to(device)
        last_action = action

        # Tile action for num_envs (vectorized env)
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == 1 and num_envs > 1:
            action = action.repeat(num_envs, 1)

        obs_dict, _, terminated, truncated, _ = env.step(action)

        if (terminated[0] or truncated[0]) and t > 10:
            success = True
            break

    return {
        "steps": steps,
        "success": success,
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

        print("[DEBUG] Env created; checking camera...")

        camera = env.unwrapped.scene.sensors.get("table_cam", None)
        if camera is None:
            raise RuntimeError("Camera sensor 'table_cam' not found; did add_camera_to_env_cfg run?")
        print(f"[DEBUG] Camera cfg: {camera.cfg.width}x{camera.cfg.height}, data_types={camera.cfg.data_types}")

        print("Loading diffusion policy...")
        policy, preprocessor, postprocessor = load_diffusion_policy(
            args.checkpoint,
            device,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        policy.eval()
        print("Policy loaded.")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Only first episode is recorded to MP4 to avoid huge files
        writer = imageio.get_writer(out_path, fps=args.fps)

        stats = []
        for ep in range(args.num_episodes):
            print(f"\nEpisode {ep+1}/{args.num_episodes}")
            result = run_episode(
                env=env,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                horizon=args.horizon,
                writer=writer if ep == 0 else None,
            )
            stats.append(result)
            print(f"  steps={result['steps']} success={result['success']}")

        writer.close()
        print(f"Saved video to {out_path}")

        avg_steps = np.mean([s["steps"] for s in stats])
        print(f"Average steps: {avg_steps:.1f}")

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
