"""Pick-and-place expert B using finite state machine.

This expert picks an object from its current location and places it at a target location,
then returns to the rest pose.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import gymnasium as gym


class ExpertState(IntEnum):
    """States for the pick-and-place finite state machine."""

    REST = 0
    GO_ABOVE_OBJ = 1
    GO_TO_OBJ = 2
    CLOSE = 3
    LIFT = 4
    GO_ABOVE_PLACE = 5
    GO_TO_PLACE = 6
    LOWER_TO_RELEASE = 7  # New: Lower to release height
    OPEN = 8
    LIFT_AFTER_RELEASE = 9  # New: Lift to hover height after release
    RETURN_REST = 10
    DONE = 11


# Gripper action values
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


class PickPlaceExpertB:
    """Finite state machine expert for pick-and-place tasks.

    This expert outputs absolute EE pose commands (position + quaternion) plus gripper action.
    Action format: [x, y, z, qw, qx, qy, qz, gripper] = 8 dims

    Gripper convention:
        - OPEN = +1.0
        - CLOSE = -1.0
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        hover_z: float = 0.25,
        grasp_z_offset: float = 0.02,  # grasp slightly into the object
        release_z_offset: float = -0.03,  # lower slightly into the object for stable release
        position_threshold: float = 0.01,
        wait_steps: int = 15,
    ):
        """Initialize the expert.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device.
            hover_z: Height for hovering above objects.
            grasp_z_offset: Offset from object top when grasping.
            release_z_offset: Offset for release height (negative = lower).
            position_threshold: Distance threshold for reaching waypoints.
            wait_steps: Steps to wait after reaching a waypoint.
        """
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.hover_z = hover_z
        self.grasp_z_offset = grasp_z_offset
        self.release_z_offset = release_z_offset
        self.position_threshold = position_threshold
        self.wait_steps = wait_steps

        # State tracking per environment
        self.state = torch.full((num_envs,), ExpertState.REST, dtype=torch.int32, device=self.device)
        self.wait_counter = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        # Store rest pose (recorded at reset)
        self.rest_pose = None  # (num_envs, 7)

        # Store place target pose
        self.place_pose = None  # (num_envs, 7)

        # Fixed grasp orientation: gripper pointing down
        # wxyz format: [w, x, y, z] = [0, 1, 0, 0] represents 180° rotation around x-axis
        self.grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

    def reset(
        self,
        ee_pose: torch.Tensor,
        place_xy: tuple[float, float],
        place_z: float = 0.055,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Reset the expert state.

        Args:
            ee_pose: Current EE pose (num_envs, 7) [x, y, z, qw, qx, qy, qz].
            place_xy: Target (x, y) for placing the object.
            place_z: Height for place pose.
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Store rest pose
        if self.rest_pose is None:
            self.rest_pose = ee_pose.clone()
        else:
            self.rest_pose[env_ids] = ee_pose[env_ids].clone()

        # Create place pose
        if self.place_pose is None:
            self.place_pose = torch.zeros(self.num_envs, 7, device=self.device)

        self.place_pose[env_ids, 0] = place_xy[0]
        self.place_pose[env_ids, 1] = place_xy[1]
        self.place_pose[env_ids, 2] = place_z
        self.place_pose[env_ids, 3:7] = self.grasp_quat

        # Reset state
        self.state[env_ids] = ExpertState.GO_ABOVE_OBJ
        self.wait_counter[env_ids] = 0

    def act(
        self,
        ee_pose: torch.Tensor,
        object_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action based on current state.

        Args:
            ee_pose: Current EE pose (num_envs, 7) [x, y, z, qw, qx, qy, qz].
            object_pose: Current object pose (num_envs, 7) [x, y, z, qw, qx, qy, qz].

        Returns:
            Action tensor (num_envs, 8) [x, y, z, qw, qx, qy, qz, gripper].
        """
        # Initialize action with current pose and open gripper
        action = torch.zeros(self.num_envs, 8, device=self.device)
        action[:, :7] = ee_pose.clone()
        action[:, 7] = GRIPPER_OPEN

        # Compute target positions for each state
        obj_xy = object_pose[:, :2]
        obj_z = object_pose[:, 2]

        # Above object position
        above_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        above_obj_pos[:, :2] = obj_xy
        above_obj_pos[:, 2] = self.hover_z

        # At object position (for grasping)
        at_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        at_obj_pos[:, :2] = obj_xy
        at_obj_pos[:, 2] = obj_z + self.grasp_z_offset

        # Above place position
        above_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        above_place_pos[:, :2] = self.place_pose[:, :2]
        above_place_pos[:, 2] = self.hover_z

        # At place position (approach height)
        at_place_pos = self.place_pose[:, :3].clone()

        # Release position (lower than place position for stable release)
        release_pos = self.place_pose[:, :3].clone()
        release_pos[:, 2] = self.place_pose[:, 2] + self.release_z_offset

        # Process each state
        for state_val in ExpertState:
            mask = self.state == state_val

            if not mask.any():
                continue

            if state_val == ExpertState.REST:
                # Stay at rest, open gripper
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

            elif state_val == ExpertState.GO_ABOVE_OBJ:
                # Move above object
                action[mask, :3] = above_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN

                # Check if reached
                dist = torch.norm(ee_pose[mask, :3] - above_obj_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                # Update wait counter for reached envs
                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                # Transition if waited enough
                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.GO_TO_OBJ
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.GO_TO_OBJ:
                # Move down to object
                action[mask, :3] = at_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN

                # Check if reached
                dist = torch.norm(ee_pose[mask, :3] - at_obj_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.CLOSE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.CLOSE:
                # Close gripper at object
                action[mask, :3] = at_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE

                # Wait for gripper to close
                self.wait_counter[mask] += 1
                transition = mask & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.LIFT
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.LIFT:
                # Lift object
                action[mask, :3] = above_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE

                # Check if reached hover height
                dist = torch.norm(ee_pose[mask, :3] - above_obj_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.GO_ABOVE_PLACE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.GO_ABOVE_PLACE:
                # Move above place position
                action[mask, :3] = above_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE

                dist = torch.norm(ee_pose[mask, :3] - above_place_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.GO_TO_PLACE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.GO_TO_PLACE:
                # Move down to place position (first stage)
                action[mask, :3] = at_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE

                dist = torch.norm(ee_pose[mask, :3] - at_place_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.LOWER_TO_RELEASE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.LOWER_TO_RELEASE:
                # Lower further for stable release
                action[mask, :3] = release_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE

                dist = torch.norm(ee_pose[mask, :3] - release_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                # 等待时间恢复正常
                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.OPEN
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.OPEN:
                # Open gripper to release object (stay at release height)
                action[mask, :3] = release_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN

                # Wait for gripper to open and object to settle
                self.wait_counter[mask] += 1
                transition = mask & (self.wait_counter >= self.wait_steps)  # 恢复正常等待时间
                self.state[transition] = ExpertState.LIFT_AFTER_RELEASE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.LIFT_AFTER_RELEASE:
                # Lift to hover height after release
                action[mask, :3] = above_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN

                dist = torch.norm(ee_pose[mask, :3] - above_place_pos[mask], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.RETURN_REST
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.RETURN_REST:
                # Return to rest pose
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

                dist = torch.norm(ee_pose[mask, :3] - self.rest_pose[mask, :3], dim=-1)
                reached = dist < self.position_threshold

                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1

                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.DONE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.DONE:
                # Stay at rest
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

        return action

    def get_state_names(self) -> list[str]:
        """Get state names for all environments.

        Returns:
            List of state names.
        """
        return [ExpertState(s.item()).name for s in self.state]

    def is_done(self) -> torch.Tensor:
        """Check if expert has finished for each environment.

        Returns:
            Boolean tensor (num_envs,).
        """
        return self.state == ExpertState.DONE
