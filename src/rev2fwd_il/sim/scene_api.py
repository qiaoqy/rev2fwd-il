"""Scene API utilities for Isaac Lab environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import gymnasium as gym


def get_env_origins(env: gym.Env) -> torch.Tensor:
    """Get environment origins for all envs.

    Args:
        env: Gymnasium environment (Isaac Lab).

    Returns:
        Tensor of shape (num_envs, 3) with xyz origins in world frame.
    """
    unwrapped = env.unwrapped
    return unwrapped.scene.env_origins.clone()


def get_ee_pose_w(env: gym.Env) -> torch.Tensor:
    """Get end-effector pose in local env frame.

    Uses env.unwrapped.scene["ee_frame"] to get the EE pose.

    Args:
        env: Gymnasium environment (Isaac Lab).

    Returns:
        Tensor of shape (num_envs, 7) with [x, y, z, qw, qx, qy, qz] in local env frame.
    """
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    device = unwrapped.device

    ee_frame = scene["ee_frame"]
    # target_pos_w and target_quat_w are in world frame
    # Shape: (num_envs, num_targets, 3) and (num_envs, num_targets, 4)
    # We take the first target (index 0)
    pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (num_envs, 4) - wxyz format

    # Convert to local env frame by subtracting origins
    env_origins = scene.env_origins  # (num_envs, 3)
    pos_local = pos_w - env_origins

    # Concatenate position and quaternion
    pose = torch.cat([pos_local, quat_w], dim=-1)  # (num_envs, 7)

    return pose


def get_object_pose_w(env: gym.Env, name: str = "object") -> torch.Tensor:
    """Get object pose in local env frame.

    Args:
        env: Gymnasium environment (Isaac Lab).
        name: Name of the object in the scene.

    Returns:
        Tensor of shape (num_envs, 7) with [x, y, z, qw, qx, qy, qz] in local env frame.
    """
    unwrapped = env.unwrapped
    scene = unwrapped.scene

    obj = scene[name]
    # root_pos_w: (num_envs, 3), root_quat_w: (num_envs, 4) - wxyz format
    pos_w = obj.data.root_pos_w  # (num_envs, 3)
    quat_w = obj.data.root_quat_w  # (num_envs, 4)

    # Convert to local env frame
    env_origins = scene.env_origins  # (num_envs, 3)
    pos_local = pos_w - env_origins

    # Concatenate position and quaternion
    pose = torch.cat([pos_local, quat_w], dim=-1)  # (num_envs, 7)

    return pose


def teleport_object_to_pose(
    env: gym.Env,
    pose_w: torch.Tensor,
    name: str = "object",
    env_ids: torch.Tensor | None = None,
) -> None:
    """Teleport object to a given pose in local env frame.

    This resets the object state by writing pose and zero velocity to simulation.

    Args:
        env: Gymnasium environment (Isaac Lab).
        pose_w: Pose tensor of shape (num_envs, 7) with [x, y, z, qw, qx, qy, qz]
                in LOCAL env frame (will be converted to world frame).
        name: Name of the object in the scene.
        env_ids: Optional tensor of environment indices to teleport.
                 If None, teleports all environments.
    """
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    device = unwrapped.device
    num_envs = unwrapped.num_envs

    obj = scene[name]

    # Determine which envs to update
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)

    # Get env origins and convert local pose to world frame
    env_origins = scene.env_origins[env_ids]  # (len(env_ids), 3)

    # Extract position and quaternion
    pos_local = pose_w[env_ids, :3]  # (len(env_ids), 3)
    quat = pose_w[env_ids, 3:7]  # (len(env_ids), 4)

    # Convert position to world frame
    pos_world = pos_local + env_origins  # (len(env_ids), 3)

    # Create root state tensor: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
    root_state = torch.zeros(len(env_ids), 13, device=device)
    root_state[:, :3] = pos_world
    root_state[:, 3:7] = quat
    # Velocities are already zero

    # Write to simulation
    obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    # Reset the object internal state
    obj.reset(env_ids)
