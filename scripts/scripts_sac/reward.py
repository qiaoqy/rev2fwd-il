"""Reward functions for SAC-based pick-place training.

All functions operate on tensors (batched over num_envs).
Independent copy from scripts_rl/reward.py for experiment isolation.
"""

from __future__ import annotations

import torch


def dense_reward(
    obj_pose: torch.Tensor,
    ee_pose: torch.Tensor,
    gripper: torch.Tensor,
    goal_xy: torch.Tensor,
    prev_obj_pose: torch.Tensor | None = None,
    distance_threshold: float = 0.03,
    height_threshold: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense reward with reaching + progress + grasp + success components.

    Args:
        obj_pose: (num_envs, 7) object pose [x,y,z,qw,qx,qy,qz].
        ee_pose: (num_envs, 7) ee pose.
        gripper: (num_envs,) gripper state (+1 open, -1 close).
        goal_xy: (2,) goal position or (num_envs, 2).
        prev_obj_pose: (num_envs, 7) previous object pose (for progress).
        distance_threshold: success threshold in meters.
        height_threshold: max z for "on table".

    Returns:
        (reward, success) — both (num_envs,) tensors.
    """
    num_envs = obj_pose.shape[0]
    device = obj_pose.device

    if goal_xy.dim() == 1:
        goal_xy = goal_xy.unsqueeze(0).expand(num_envs, -1)

    obj_xy = obj_pose[:, :2]
    obj_z = obj_pose[:, 2]
    dist_to_goal = torch.norm(obj_xy - goal_xy, dim=-1)

    # Sparse success
    success = (obj_z < height_threshold) & (gripper > 0.5) & (dist_to_goal < distance_threshold)
    r_success = torch.where(success, 10.0, 0.0)

    # Dense reaching (ee → object)
    ee_to_obj = torch.norm(ee_pose[:, :3] - obj_pose[:, :3], dim=-1)
    r_reach = -1.0 * ee_to_obj

    # Dense progress (object → goal)
    r_progress = torch.zeros(num_envs, device=device)
    if prev_obj_pose is not None:
        prev_dist = torch.norm(prev_obj_pose[:, :2] - goal_xy, dim=-1)
        r_progress = (prev_dist - dist_to_goal) * 5.0

    # Grasp bonus
    r_grasp = torch.where(
        (obj_z > 0.05) & (gripper < -0.5),
        torch.tensor(2.0, device=device),
        torch.tensor(0.0, device=device),
    )

    reward = r_success + r_reach + r_progress + r_grasp
    return reward, success


def sparse_reward(
    obj_pose: torch.Tensor,
    gripper: torch.Tensor,
    goal_xy: torch.Tensor,
    distance_threshold: float = 0.03,
    height_threshold: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse binary reward: only +10 on success."""
    num_envs = obj_pose.shape[0]
    if goal_xy.dim() == 1:
        goal_xy = goal_xy.unsqueeze(0).expand(num_envs, -1)

    obj_xy = obj_pose[:, :2]
    obj_z = obj_pose[:, 2]
    dist_to_goal = torch.norm(obj_xy - goal_xy, dim=-1)

    success = (obj_z < height_threshold) & (gripper > 0.5) & (dist_to_goal < distance_threshold)
    reward = torch.where(success, 10.0, 0.0)
    return reward, success


def distance_reward(
    obj_pose: torch.Tensor,
    goal_xy: torch.Tensor,
) -> torch.Tensor:
    """Negative distance to goal (no success bonus)."""
    if goal_xy.dim() == 1:
        goal_xy = goal_xy.unsqueeze(0).expand(obj_pose.shape[0], -1)
    return -torch.norm(obj_pose[:, :2] - goal_xy, dim=-1)
