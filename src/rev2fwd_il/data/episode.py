"""Episode data structure for trajectory recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Episode:
    """Episode data container for pick-and-place trajectories.

    All arrays use numpy for easy serialization with np.savez.

    Attributes:
        obs: Observation sequence from policy input, shape (T, obs_dim).
        ee_pose: End-effector pose sequence, shape (T, 7) [x, y, z, qw, qx, qy, qz].
        obj_pose: Object pose sequence, shape (T, 7) [x, y, z, qw, qx, qy, qz].
        gripper: Gripper action sequence, shape (T,) with values +1 (open) or -1 (close).
        place_pose: Target place pose for this episode (B's random table target), shape (7,).
        goal_pose: Fixed goal pose (plate center), shape (7,).
        success: Whether the episode was successful.
    """

    obs: np.ndarray  # (T, obs_dim)
    ee_pose: np.ndarray  # (T, 7)
    obj_pose: np.ndarray  # (T, 7)
    gripper: np.ndarray  # (T,)
    place_pose: np.ndarray  # (7,)
    goal_pose: np.ndarray  # (7,)
    success: bool = False

    def __post_init__(self):
        """Validate array shapes after initialization."""
        T = len(self.obs)
        assert self.ee_pose.shape[0] == T, f"ee_pose length {self.ee_pose.shape[0]} != obs length {T}"
        assert self.obj_pose.shape[0] == T, f"obj_pose length {self.obj_pose.shape[0]} != obs length {T}"
        assert self.gripper.shape[0] == T, f"gripper length {self.gripper.shape[0]} != obs length {T}"
        assert self.place_pose.shape == (7,), f"place_pose shape {self.place_pose.shape} != (7,)"
        assert self.goal_pose.shape == (7,), f"goal_pose shape {self.goal_pose.shape} != (7,)"

    @property
    def length(self) -> int:
        """Return episode length (number of timesteps)."""
        return len(self.obs)

    def to_dict(self) -> dict[str, Any]:
        """Convert episode to dictionary for serialization.

        Returns:
            Dictionary with all episode data.
        """
        return {
            "obs": self.obs,
            "ee_pose": self.ee_pose,
            "obj_pose": self.obj_pose,
            "gripper": self.gripper,
            "place_pose": self.place_pose,
            "goal_pose": self.goal_pose,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        """Create Episode from dictionary.

        Args:
            d: Dictionary with episode data.

        Returns:
            Episode instance.
        """
        return cls(
            obs=d["obs"],
            ee_pose=d["ee_pose"],
            obj_pose=d["obj_pose"],
            gripper=d["gripper"],
            place_pose=d["place_pose"],
            goal_pose=d["goal_pose"],
            success=d["success"],
        )
