"""Rollout evaluation for policy A on the forward task."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    import gymnasium as gym
    from rev2fwd_il.models.mlp_policy import MLPPolicy
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec


def evaluate_A_forward(
    env: "gym.Env",
    policy: "MLPPolicy",
    mean: np.ndarray,
    std: np.ndarray,
    task_spec: "PickPlaceTaskSpec",
    num_rollouts: int = 50,
    horizon: int = 250,
    device: str = "cuda",
    verbose: bool = True,
    print_every: int = 1,  # 新增：控制打印频率
) -> dict:
    """Evaluate policy A on the forward pick-and-place task.

    Forward task:
        - Cube starts at a random table position (env.reset default)
        - Policy should pick it up and place it at the goal (plate center)
        - Success: cube XY within success_radius of goal

    Args:
        env: Isaac Lab gymnasium environment.
        policy: Trained MLP policy.
        mean: Observation mean for normalization, shape (obs_dim,).
        std: Observation std for normalization, shape (obs_dim,).
        task_spec: Task specification with goal position and success radius.
        num_rollouts: Number of evaluation rollouts.
        horizon: Maximum steps per rollout.
        device: Torch device for policy inference.
        verbose: Whether to print progress.
        print_every: Print every N rollouts (default 1 = print all).

    Returns:
        Dictionary with:
            - success_rate: Fraction of successful rollouts
            - avg_final_dist: Average final XY distance to goal
            - final_dists: List of final distances for each rollout
            - successes: List of booleans for each rollout
    """
    from rev2fwd_il.sim.scene_api import get_object_pose_w

    policy.eval()

    # Convert normalization to torch tensors
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    goal_xy = np.array(task_spec.goal_xy)
    success_radius = task_spec.success_radius

    # Results storage
    successes = []
    final_dists = []
    # 不再存储完整的 dist_curves 以节省内存，只存储最终距离

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Policy A on Forward Task")
        print(f"{'='*60}")
        print(f"Goal XY: {goal_xy}")
        print(f"Success radius: {success_radius}")
        print(f"Horizon: {horizon}")
        print(f"Num rollouts: {num_rollouts}")
        print(f"{'='*60}\n")
        sys.stdout.flush()

    for ep_idx in range(num_rollouts):
        try:
            # Reset environment (cube spawns at random table position)
            obs_dict, _ = env.reset()
            obs = obs_dict["policy"]  # (num_envs, obs_dim)

            # We only use env 0 for evaluation
            obs_np = obs[0].cpu().numpy()

            # 用于存储最后一个有效的 action
            last_act_np = None
            steps_taken = 0

            for step in range(horizon):
                # Normalize observation
                obs_t = torch.from_numpy(obs_np).float().to(device).unsqueeze(0)  # (1, obs_dim)
                obs_normed = (obs_t - mean_t) / std_t

                # Policy inference
                with torch.no_grad():
                    act_pred = policy(obs_normed)  # (1, 8)

                # Convert to action for env
                act_np = act_pred[0].cpu().numpy()  # (8,)
                last_act_np = act_np

                # Create action tensor for all envs (repeat for num_envs)
                num_envs = env.unwrapped.num_envs
                action = torch.from_numpy(act_np).float().to(env.unwrapped.device)
                action = action.unsqueeze(0).repeat(num_envs, 1)  # (num_envs, 8)

                # Step environment
                obs_dict, _, terminated, truncated, _ = env.step(action)
                obs_np = obs_dict["policy"][0].cpu().numpy()
                steps_taken += 1

                # Check if episode ended (但不立即 break，继续完成评估)
                if terminated[0] or truncated[0]:
                    break

            # Final evaluation
            obj_pose = get_object_pose_w(env)
            obj_xy = obj_pose[0, :2].cpu().numpy()
            final_dist = np.linalg.norm(obj_xy - goal_xy)

            # Get final gripper state from last action
            final_gripper = last_act_np[7] if last_act_np is not None else 1.0

            # Success: cube near goal
            is_success = (final_dist < success_radius)

            successes.append(is_success)
            final_dists.append(final_dist)

            # 打印每个 rollout 或按 print_every 频率打印
            if verbose and ((ep_idx + 1) % print_every == 0):
                print(
                    f"Rollout {ep_idx + 1:3d}/{num_rollouts} | "
                    f"Steps: {steps_taken:3d} | "
                    f"Final dist: {final_dist:.4f} | "
                    f"Gripper: {final_gripper:+.2f} | "
                    f"Success: {is_success}"
                )
                sys.stdout.flush()

        except Exception as e:
            print(f"ERROR in rollout {ep_idx + 1}: {e}")
            sys.stdout.flush()
            # 记录失败的 rollout
            successes.append(False)
            final_dists.append(float('inf'))
            continue

    # Compute statistics
    # 过滤掉无效的结果
    valid_dists = [d for d in final_dists if d != float('inf')]
    
    success_rate = np.mean(successes) if successes else 0.0
    avg_final_dist = np.mean(valid_dists) if valid_dists else float('inf')

    if verbose:
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Success rate: {100*success_rate:.1f}% ({sum(successes)}/{num_rollouts})")
        print(f"Average final distance: {avg_final_dist:.4f}m")
        if valid_dists:
            print(f"Min final distance: {min(valid_dists):.4f}m")
            print(f"Max final distance: {max(valid_dists):.4f}m")
        print(f"{'='*60}\n")
        sys.stdout.flush()

    return {
        "success_rate": success_rate,
        "avg_final_dist": avg_final_dist,
        "final_dists": final_dists,
        "successes": successes,
    }


def load_policy_and_norm(
    checkpoint_path: str,
    norm_path: str,
    device: str = "cuda",
):
    """Load trained policy and normalization statistics.

    Args:
        checkpoint_path: Path to model.pt checkpoint.
        norm_path: Path to norm.json file.
        device: Torch device for model.

    Returns:
        Tuple of (policy, mean, std).
    """
    from rev2fwd_il.models.mlp_policy import MLPPolicy
    from rev2fwd_il.models.resnet_policy import ResNetPolicy
    from rev2fwd_il.data.normalize import load_norm

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_dim = checkpoint["obs_dim"]
    act_dim = checkpoint["act_dim"]
    hidden = tuple(checkpoint["hidden"])
    arch = checkpoint.get("arch", "mlp")  # Default to mlp for backward compatibility

    # Create model based on architecture
    if arch == "mlp":
        policy = MLPPolicy(obs_dim=obs_dim, hidden=hidden, act_dim=act_dim)
    elif arch == "resnet":
        num_blocks = checkpoint.get("num_blocks", 3)
        dropout = checkpoint.get("dropout", 0.0)
        policy = ResNetPolicy(
            obs_dim=obs_dim,
            hidden_dim=hidden[0],
            num_blocks=num_blocks,
            act_dim=act_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    # Load normalization
    mean, std = load_norm(norm_path)

    return policy, mean, std
