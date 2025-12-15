"""Task specification for pick-and-place tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym


@dataclass
class PickPlaceTaskSpec:
    """Specification for a pick-and-place task.

    Attributes:
        goal_xy: Fixed goal position (plate center) in local env frame.
        table_xy_min: Minimum (x, y) for random placement on table.
        table_xy_max: Maximum (x, y) for random placement on table.
        hover_z: Height for hovering above objects.
        grasp_z_offset: Offset from object surface when grasping.
        success_radius: Radius for determining success (xy distance).
        settle_steps: Number of steps to wait after reaching a waypoint.
        object_height: Height of the object (for computing grasp z).
    """

    goal_xy: tuple[float, float] = (0.5, 0.0)
    table_xy_min: tuple[float, float] = (0.35, -0.25)
    table_xy_max: tuple[float, float] = (0.65, 0.25)
    hover_z: float = 0.25
    grasp_z_offset: float = 0.0  # offset above object top
    success_radius: float = 0.03
    settle_steps: int = 10
    object_height: float = 0.05  # approximate cube height

    def sample_table_xy(self, rng: np.random.Generator) -> tuple[float, float]:
        """Sample a random (x, y) position on the table, excluding goal vicinity.

        Args:
            rng: NumPy random generator.

        Returns:
            Tuple (x, y) sampled uniformly from valid table region.
        """
        exclusion_radius = self.success_radius * 2.0
        max_attempts = 100

        for _ in range(max_attempts):
            x = rng.uniform(self.table_xy_min[0], self.table_xy_max[0])
            y = rng.uniform(self.table_xy_min[1], self.table_xy_max[1])

            # Check if outside goal exclusion zone
            dist_to_goal = np.sqrt((x - self.goal_xy[0]) ** 2 + (y - self.goal_xy[1]) ** 2)
            if dist_to_goal > exclusion_radius:
                return (x, y)

        # Fallback: return a corner position if sampling fails
        return (self.table_xy_min[0], self.table_xy_min[1])


def make_goal_pose_w(
    env: gym.Env,
    goal_xy: tuple[float, float],
    z: float = 0.055,
) -> torch.Tensor:
    """Create goal pose tensor for all envs.

    Args:
        env: Gymnasium environment (Isaac Lab).
        goal_xy: (x, y) position in local env frame.
        z: Height of goal pose.

    Returns:
        Tensor of shape (num_envs, 7) with [x, y, z, qw, qx, qy, qz] (wxyz quaternion).
    """
    unwrapped = env.unwrapped
    device = unwrapped.device
    num_envs = unwrapped.num_envs

    pose = torch.zeros(num_envs, 7, device=device)
    pose[:, 0] = goal_xy[0]
    pose[:, 1] = goal_xy[1]
    pose[:, 2] = z
    # Identity quaternion (wxyz): [1, 0, 0, 0]
    pose[:, 3] = 1.0

    return pose


def make_place_pose_w(
    env: gym.Env,
    xy: tuple[float, float],
    z: float = 0.055,
) -> torch.Tensor:
    """Create place pose tensor for all envs.

    Args:
        env: Gymnasium environment (Isaac Lab).
        xy: (x, y) position in local env frame.
        z: Height of place pose.

    Returns:
        Tensor of shape (num_envs, 7) with [x, y, z, qw, qx, qy, qz] (wxyz quaternion).
    """
    unwrapped = env.unwrapped
    device = unwrapped.device
    num_envs = unwrapped.num_envs

    pose = torch.zeros(num_envs, 7, device=device)
    pose[:, 0] = xy[0]
    pose[:, 1] = xy[1]
    pose[:, 2] = z
    # Identity quaternion (wxyz): [1, 0, 0, 0]
    pose[:, 3] = 1.0

    return pose


def is_pose_close_xy(
    pose_w: torch.Tensor,
    xy: tuple[float, float],
    radius: float,
) -> torch.Tensor:
    """Check if poses are within radius of target xy.

    Args:
        pose_w: Pose tensor of shape (num_envs, 7) or (num_envs, 3+).
        xy: Target (x, y) position.
        radius: Success radius.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    device = pose_w.device
    target = torch.tensor([xy[0], xy[1]], device=device)
    dist = torch.norm(pose_w[:, :2] - target, dim=-1)
    return dist < radius
