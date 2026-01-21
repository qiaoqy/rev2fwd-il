#!/usr/bin/env python3
"""
Deploy trained Diffusion Policy to Isaac Sim for evaluation.

Usage:
    python deploy_dp_isaacsim.py \
        --checkpoint runs/checkpoints/last/pretrained_model \
        --dataset_dir runs/diffusion_from_zarr/lerobot_dataset \
        --num_episodes 10 \
        --headless
"""

import argparse
import os
import numpy as np


def dummy_npwarn_decorator_factory():
    def npwarn_decorator(x):
        return x

    return npwarn_decorator


np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)

from isaaclab.app import AppLauncher

# CLI arguments
parser = argparse.ArgumentParser(description="Deploy Diffusion Policy to Isaac Sim.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--checkpoint", type=str, default="runs_500/diffusion_from_zarr/checkpoints/checkpoints/last/pretrained_model",
    help="Path to pretrained Diffusion Policy model directory."
)
parser.add_argument(
    "--dataset_dir", type=str, default="runs/diffusion_from_zarr/lerobot_dataset",
    help="Path to LeRobot dataset directory (for normalization stats). Auto-detected if not specified."
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode.")
parser.add_argument("--n_action_steps", type=int, default=8, help="Action steps to execute per prediction.")
parser.add_argument("--success_radius", type=float, default=0.03, help="Success threshold distance.")
parser.add_argument("--out_dir", type=str, default="eval_results", help="Output directory for results.")
parser.add_argument("--save_video", action="store_true", help="Save video of rollouts.")

# AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest of imports after app launch."""

import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import gymnasium as gym
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Camera setup
from tools.camera_recorder import add_camera_to_env_cfg

wp.init()


# -----------------------------
# Policy Loading with Preprocessors
# -----------------------------
def load_policy_with_processors(checkpoint_path: str, dataset_dir: str, device: str):
    """
    Load pretrained DiffusionPolicy with matching preprocessors.

    Args:
        checkpoint_path: Path to pretrained model directory
        dataset_dir: Path to LeRobot dataset directory (for stats)
        device: Device for inference

    Returns:
        policy: The loaded policy
        preprocess: Preprocessing function for observations
        postprocess: Postprocessing function for actions
    """
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    print(f"[INFO] Loading policy from: {checkpoint_path}")
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(device)
    policy.eval()

    print(f"[INFO] Policy loaded. Config: n_obs_steps={policy.config.n_obs_steps}, "
          f"horizon={policy.config.horizon}, n_action_steps={policy.config.n_action_steps}")

    # # Find dataset directory
    # checkpoint_dir = Path(checkpoint_path)

    if dataset_dir is not None:
        dataset_path = Path(dataset_dir)

    # Load dataset metadata (same repo_id as training script: "local/diffusion_from_zarr")
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="local/diffusion_from_zarr",
        root=str(dataset_path),
    )

    print(f"[INFO] Dataset stats keys: {list(dataset_metadata.stats.keys())}")

    # Create preprocessors using the same stats as training
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint_path,
        dataset_stats=dataset_metadata.stats
    )

    print(f"[INFO] Preprocessors created with dataset stats from: {dataset_path}")

    return policy, preprocess, postprocess


# -----------------------------
# Observation Processing
# -----------------------------
def prepare_observation(rgb: np.ndarray, ee_pose: np.ndarray, device: str) -> dict:
    """
    Prepare raw observation dict (before preprocessing).

    Args:
        rgb: (H, W, 3) uint8 image
        ee_pose: (7,) float32 end-effector pose [x, y, z, qw, qx, qy, qz]
    Returns:
        dict with observation tensors (to be preprocessed by preprocess function)
    """
    # Image: (H, W, 3) uint8 -> (1, 3, H, W) float32 [0, 1]
    # LeRobot expects images normalized to [0, 1]
    img = rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # State: (7,) -> (1, 7)
    state_tensor = torch.from_numpy(ee_pose.astype(np.float32)).unsqueeze(0).to(device)

    return {
        "observation.image": img_tensor,
        "observation.state": state_tensor,
    }


# -----------------------------
# Action Processing
# -----------------------------
def process_action_chunk(action_chunk, device: str) -> np.ndarray:
    """
    Process action chunk from policy output.

    Args:
        action_chunk: Policy output, could be torch.Tensor or np.ndarray
    Returns:
        (horizon, action_dim) np.ndarray
    """
    if isinstance(action_chunk, torch.Tensor):
        action_chunk = action_chunk.cpu().numpy()

    # Remove batch dimension if present
    if action_chunk.ndim == 3:
        action_chunk = action_chunk[0]  # (1, horizon, action_dim) -> (horizon, action_dim)
    elif action_chunk.ndim == 1:
        action_chunk = action_chunk[np.newaxis, :]  # (action_dim,) -> (1, action_dim)

    return action_chunk


# -----------------------------
# Metrics
# -----------------------------
def compute_episode_metrics(
        obj_final_xy: np.ndarray,
        goal_xy: np.ndarray,
        success_radius: float,
        episode_length: int,
) -> dict:
    """Compute metrics for a single episode."""
    dist = float(np.linalg.norm(obj_final_xy - goal_xy))
    success = dist < success_radius

    return {
        "success": success,
        "final_distance": dist,
        "episode_length": episode_length,
    }


# -----------------------------
# Main Deployment Loop
# -----------------------------
def main():
    # Setup output directory
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse environment configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Add camera to environment
    print("[INFO] Adding camera to environment config...")
    add_camera_to_env_cfg(env_cfg, image_width=256, image_height=256)

    # Create environment
    print("[INFO] Creating environment...")
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    obs_dict, _ = env.reset()

    # Get camera reference
    camera = env.unwrapped.scene.sensors["table_cam"]

    # Load policy with preprocessors
    policy, preprocess, postprocess = load_policy_with_processors(
        args_cli.checkpoint,
        args_cli.dataset_dir,
        args_cli.device
    )

    # Goal pose (same as training)
    desired_position_world = torch.tensor([0.6, 0.0, 0.2], device=env.unwrapped.device)
    desired_position = desired_position_world - env.unwrapped.scene.env_origins
    goal_pose_np = np.array(
        [desired_position_world[0].item(), desired_position_world[1].item(), desired_position_world[2].item(),
         1.0, 0.0, 0.0, 0.0],
        dtype=np.float32
    )
    goal_xy = goal_pose_np[:2]

    # Initialize action buffer
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0  # quaternion w
    actions[:, 7] = 1.0  # gripper open

    # Evaluation metrics
    all_metrics = []
    success_count = 0

    print("\n" + "=" * 60)
    print("Diffusion Policy Deployment - Isaac Sim")
    print("=" * 60)
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print(f"  Dataset dir: {args_cli.dataset_dir}")
    print(f"  Num episodes: {args_cli.num_episodes}")
    print(f"  Max steps: {args_cli.max_steps}")
    print(f"  n_action_steps: {args_cli.n_action_steps}")
    print(f"  Success radius: {args_cli.success_radius}")
    print("=" * 60 + "\n")

    # Episode loop
    ep_idx = 0
    step_count = 0
    action_buffer = None
    action_buffer_idx = 0

    # Video recording (optional)
    video_frames = [] if args_cli.save_video else None

    # Reset policy internal state
    policy.reset()

    start_time = time.time()

    while simulation_app.is_running() and ep_idx < args_cli.num_episodes:
        with torch.inference_mode():
            # =========================================================
            # 1) Get current observation
            # =========================================================
            # RGB image
            rgb = camera.data.output["rgb"]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            rgb_np = rgb[0].detach().cpu().numpy().astype(np.uint8)  # env 0

            # End-effector pose
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_pos = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
            ee_pose_np = ee_pose[0].detach().cpu().numpy().astype(np.float32)  # env 0

            # Object pose (for metrics)
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            obj_xy = object_data.root_pos_w.detach().cpu().numpy()[0, :2].astype(np.float32)

            # Save video frame
            if video_frames is not None:
                video_frames.append(rgb_np.copy())

            # =========================================================
            # 2) Get action from policy (or use buffered action)
            # =========================================================
            if action_buffer is None or action_buffer_idx >= len(action_buffer):
                # Need new prediction from policy
                # Step 1: Prepare raw observation
                obs_raw = prepare_observation(rgb_np, ee_pose_np, args_cli.device)

                # Step 2: Apply preprocessing (normalization using dataset stats)
                obs = preprocess(obs_raw)

                with torch.no_grad():
                    # Step 3: Get action from policy
                    action_chunk = policy.select_action(obs)

                    # Step 4: Apply postprocessing (denormalization)
                    action_chunk = postprocess(action_chunk)

                action_buffer = process_action_chunk(action_chunk, args_cli.device)
                action_buffer_idx = 0

                # Debug print (first step only)
                if step_count == 0:
                    print(f"[DEBUG] First action chunk shape: {action_buffer.shape}")
                    print(f"[DEBUG] First action: {action_buffer[0]}")

            # Get current action from buffer
            pred_action = action_buffer[action_buffer_idx]
            action_buffer_idx += 1

            # Convert to torch and set to all envs
            actions[0] = torch.from_numpy(pred_action).to(env.unwrapped.device)

            # =========================================================
            # 3) Execute action in environment
            # =========================================================
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            done = bool(dones[0].item())
            step_count += 1

            # =========================================================
            # 4) Check for episode termination
            # =========================================================
            if done or step_count >= args_cli.max_steps:
                # Compute metrics
                metrics = compute_episode_metrics(
                    obj_final_xy=obj_xy,
                    goal_xy=goal_xy,
                    success_radius=args_cli.success_radius,
                    episode_length=step_count,
                )
                all_metrics.append(metrics)

                if metrics["success"]:
                    success_count += 1

                print(f"[Episode {ep_idx + 1}/{args_cli.num_episodes}] "
                      f"Steps: {step_count}, "
                      f"Success: {metrics['success']}, "
                      f"Distance: {metrics['final_distance']:.4f}")

                # Save video for this episode
                if video_frames is not None and len(video_frames) > 0:
                    save_video(video_frames, out_dir / f"episode_{ep_idx:04d}.mp4")
                    video_frames = []

                # Reset for next episode
                ep_idx += 1
                step_count = 0
                action_buffer = None
                action_buffer_idx = 0

                if ep_idx < args_cli.num_episodes:
                    obs_dict, _ = env.reset()
                    policy.reset()  # Reset policy internal queues
                    actions.zero_()
                    actions[:, 3] = 1.0
                    actions[:, 7] = 1.0

    # =========================================================
    # 5) Compute and save final results
    # =========================================================
    elapsed = time.time() - start_time

    # Aggregate metrics
    success_rate = success_count / len(all_metrics) if all_metrics else 0.0
    avg_distance = np.mean([m["final_distance"] for m in all_metrics]) if all_metrics else 0.0
    avg_length = np.mean([m["episode_length"] for m in all_metrics]) if all_metrics else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes: {len(all_metrics)}")
    print(f"  Success Rate: {success_rate * 100:.1f}% ({success_count}/{len(all_metrics)})")
    print(f"  Avg Final Distance: {avg_distance:.4f}")
    print(f"  Avg Episode Length: {avg_length:.1f}")
    print(f"  Total Time: {elapsed:.1f}s")
    print("=" * 60)

    # Save results to JSON
    results = {
        "config": {
            "checkpoint": args_cli.checkpoint,
            "dataset_dir": args_cli.dataset_dir,
            "num_episodes": len(all_metrics),
            "max_steps": args_cli.max_steps,
            "n_action_steps": args_cli.n_action_steps,
            "success_radius": args_cli.success_radius,
        },
        "aggregate": {
            "success_rate": success_rate,
            "avg_final_distance": avg_distance,
            "avg_episode_length": avg_length,
            "total_successes": success_count,
        },
        "episodes": all_metrics,
        "evaluation_time_s": elapsed,
    }

    results_file = out_dir / "deployment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to: {results_file}")

    # Close environment
    env.close()


def save_video(frames: list, output_path: Path, fps: int = 20):
    """Save frames as video using imageio."""
    try:
        import imageio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"[INFO] Video saved: {output_path}")
    except ImportError:
        print("[WARNING] imageio not installed, skipping video save.")
    except Exception as e:
        print(f"[WARNING] Failed to save video: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()