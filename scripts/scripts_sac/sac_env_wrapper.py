"""SAC environment wrapper for Isaac Lab pick-place.

Wraps the Isaac Lab gym environment into a standard interface for SAC training,
with reward computation, observation preprocessing, and vectorized support.

Independent copy from scripts_rl/rl_env_wrapper.py for experiment isolation.

This module reuses camera/env creation from the existing eval scripts and adds:
- Dense/sparse reward computation
- Observation dict → tensor conversion
- Action chunk execution (predict N, execute M)
- Episode auto-reset with object teleportation
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import utilities from existing eval scripts
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts_pick_place"
_ALT_SPEC = importlib.util.spec_from_file_location(
    "test_alternating",
    str(_SCRIPTS_DIR / "6_test_alternating.py"),
)
_ALT_MOD = importlib.util.module_from_spec(_ALT_SPEC)
_ALT_SPEC.loader.exec_module(_ALT_MOD)

make_env_with_camera = _ALT_MOD.make_env_with_camera
load_policy_config = _ALT_MOD.load_policy_config
load_diffusion_policy = _ALT_MOD.load_diffusion_policy
create_target_markers = _ALT_MOD.create_target_markers
update_target_markers = _ALT_MOD.update_target_markers

from rev2fwd_il.sim.scene_api import (
    get_ee_pose_w,
    get_object_pose_w,
    teleport_object_to_pose,
    pre_position_gripper_down,
)

from . import reward as reward_module


class PickPlaceRLEnv:
    """RL wrapper around Isaac Lab pick-place environment.

    Manages:
    - Environment creation with cameras
    - Observation extraction (images + state) for policy input
    - Reward computation per step
    - Episode reset with random object placement
    - Vectorized execution for parallel envs (num_envs > 1)
    """

    def __init__(
        self,
        task_id: str = "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        num_envs: int = 1,
        device: str = "cuda",
        image_width: int = 128,
        image_height: int = 128,
        goal_xy: Tuple[float, float] = (0.5, -0.2),
        red_region_center_xy: Tuple[float, float] = (0.5, 0.2),
        red_region_size_xy: Tuple[float, float] = (0.3, 0.3),
        distance_threshold: float = 0.03,
        height_threshold: float = 0.15,
        horizon: int = 1500,
        reward_type: str = "dense",
        use_fabric: bool = True,
        headless: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.goal_xy = torch.tensor(goal_xy, dtype=torch.float32, device=device)
        self.red_region_center_xy = red_region_center_xy
        self.red_region_size_xy = red_region_size_xy
        self.distance_threshold = distance_threshold
        self.height_threshold = height_threshold
        self.horizon = horizon
        self.reward_type = reward_type

        # Create Isaac Lab environment
        self.env = make_env_with_camera(
            task_id=task_id,
            num_envs=num_envs,
            device=device,
            use_fabric=use_fabric,
            image_width=image_width,
            image_height=image_height,
            episode_length_s=1000.0,
            disable_terminations=True,
        )

        # Camera references
        unwrapped = self.env.unwrapped
        self.table_camera = unwrapped.scene["table_cam"]
        self.wrist_camera = unwrapped.scene["wrist_cam"]

        # State tracking
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.gripper_state = torch.ones(num_envs, device=device)  # start open
        self.prev_obj_pose = None

        # Sampling bounds for random object placement
        # (within the red_region for simplicity — can also use full table)
        rx, ry = red_region_center_xy
        sx, sy = red_region_size_xy
        self.obj_x_range = (rx - sx / 2, rx + sx / 2)
        self.obj_y_range = (ry - sy / 2, ry + sy / 2)

    def reset(self, env_ids: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Reset environments and teleport object to random position.

        Args:
            env_ids: Which envs to reset. None = all.

        Returns:
            Observation dict with keys: observation.image, observation.wrist_image,
            observation.state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Full env reset
        self.env.reset()
        pre_position_gripper_down(self.env)

        # Random object placement
        n = len(env_ids)
        rand_x = torch.rand(n, device=self.device) * (self.obj_x_range[1] - self.obj_x_range[0]) + self.obj_x_range[0]
        rand_y = torch.rand(n, device=self.device) * (self.obj_y_range[1] - self.obj_y_range[0]) + self.obj_y_range[0]

        obj_pose = torch.zeros(n, 7, device=self.device)
        obj_pose[:, 0] = rand_x
        obj_pose[:, 1] = rand_y
        obj_pose[:, 2] = 0.022  # on table
        obj_pose[:, 3] = 1.0    # identity quat

        # Need to broadcast for teleport if not resetting all envs
        full_pose = torch.zeros(self.num_envs, 7, device=self.device)
        full_pose[:, 3] = 1.0
        full_pose[env_ids] = obj_pose
        teleport_object_to_pose(self.env, full_pose, name="object", env_ids=env_ids)

        # Settle physics
        ee_hold = get_ee_pose_w(self.env)
        hold_action = torch.zeros(self.num_envs, self.env.action_space.shape[-1], device=self.device)
        hold_action[:, :7] = ee_hold[:, :7]
        hold_action[:, 7] = 1.0
        for _ in range(10):
            self.env.step(hold_action)

        self.step_count[env_ids] = 0
        self.gripper_state[env_ids] = 1.0
        self.prev_obj_pose = get_object_pose_w(self.env)

        return self._get_obs()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute action in environment.

        Args:
            action: (num_envs, 8) action tensor [ee_goal(7), gripper(1)].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Store previous object pose for progress reward
        prev_obj_pose = self.prev_obj_pose.clone() if self.prev_obj_pose is not None else None

        # Execute in sim
        obs_dict, _, terminated, truncated, info = self.env.step(action)

        # Update gripper state tracking
        self.gripper_state = action[:, 7].clone()

        # Get current poses
        ee_pose = get_ee_pose_w(self.env)
        obj_pose = get_object_pose_w(self.env)

        # Compute reward
        if self.reward_type == "dense":
            rew, success = reward_module.dense_reward(
                obj_pose=obj_pose,
                ee_pose=ee_pose,
                gripper=self.gripper_state,
                goal_xy=self.goal_xy,
                prev_obj_pose=prev_obj_pose,
                distance_threshold=self.distance_threshold,
                height_threshold=self.height_threshold,
            )
        elif self.reward_type == "sparse":
            rew, success = reward_module.sparse_reward(
                obj_pose=obj_pose,
                gripper=self.gripper_state,
                goal_xy=self.goal_xy,
                distance_threshold=self.distance_threshold,
                height_threshold=self.height_threshold,
            )
        else:
            rew = reward_module.distance_reward(obj_pose, self.goal_xy)
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.prev_obj_pose = obj_pose.clone()
        self.step_count += 1

        # Truncation by horizon
        truncated_horizon = self.step_count >= self.horizon
        terminated_flag = success
        truncated_flag = truncated_horizon & ~success

        info_out = {
            "success": success,
            "step_count": self.step_count.clone(),
            "ee_pose": ee_pose,
            "obj_pose": obj_pose,
        }

        obs = self._get_obs()
        return obs, rew, terminated_flag, truncated_flag, info_out

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """Extract observation dict from environment.

        Returns:
            Dict with:
                observation.image: (num_envs, 3, H, W) float32 [0, 1]
                observation.wrist_image: (num_envs, 3, H, W) float32 [0, 1]
                observation.state: (num_envs, 15) float32
        """
        # Table camera
        table_rgb = self.table_camera.data.output["rgb"]  # (num_envs, H, W, 3+)
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        table_float = table_rgb.float() / 255.0  # (num_envs, H, W, 3)
        table_float = table_float.permute(0, 3, 1, 2)  # (num_envs, 3, H, W)

        # Wrist camera
        wrist_rgb = self.wrist_camera.data.output["rgb"]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]
        wrist_float = wrist_rgb.float() / 255.0
        wrist_float = wrist_float.permute(0, 3, 1, 2)

        # State: ee_pose(7) + obj_pose(7) + gripper(1) = 15
        ee_pose = get_ee_pose_w(self.env)      # (num_envs, 7)
        obj_pose = get_object_pose_w(self.env)  # (num_envs, 7)
        gripper = self.gripper_state.unsqueeze(-1)  # (num_envs, 1)
        state = torch.cat([ee_pose, obj_pose, gripper], dim=-1)  # (num_envs, 15)

        return {
            "observation.image": table_float,
            "observation.wrist_image": wrist_float,
            "observation.state": state,
        }

    def close(self):
        self.env.close()


def load_pretrained_diffusion_policy(
    checkpoint_dir: str,
    device: str,
    n_action_steps: int = 8,
    image_height: int = 128,
    image_width: int = 128,
):
    """Load a pretrained Diffusion Policy for SAC finetuning.

    Returns:
        (policy, preprocessor, postprocessor, config_info)
    """
    policy, preprocessor, postprocessor, n_inference_steps, n_act = load_diffusion_policy(
        pretrained_dir=checkpoint_dir,
        device=device,
        image_height=image_height,
        image_width=image_width,
        n_action_steps=n_action_steps,
    )
    config_info = load_policy_config(checkpoint_dir)
    return policy, preprocessor, postprocessor, config_info
