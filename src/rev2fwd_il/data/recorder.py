"""Recorder utilities for collecting expert rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import gymnasium as gym
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec

from .episode import Episode


def rollout_expert_B_reverse(
    env: gym.Env,
    expert: "PickPlaceExpertB",
    task_spec: "PickPlaceTaskSpec",
    rng: np.random.Generator,
    horizon: int,
    settle_steps: int = 30,
) -> tuple[Episode, bool]:
    """Run a single reverse rollout with Expert B and record trajectory.

    Reverse task flow:
        1. Reset environment
        2. Teleport cube to goal position (plate center)
        3. Sample random place target on table
        4. Expert B picks from goal and places at random table position
        5. Add settle steps to let cube come to rest
        6. Check success (cube near place target)

    Args:
        env: Isaac Lab gymnasium environment.
        expert: PickPlaceExpertB instance.
        task_spec: Task specification with goal/table bounds.
        rng: NumPy random generator.
        horizon: Maximum steps for the episode.
        settle_steps: Extra steps after expert finishes to let cube settle.

    Returns:
        Tuple of (Episode, expert_completed) where expert_completed indicates
        if the expert FSM reached DONE state (completed full trajectory).
    """
    # Import here to avoid circular imports
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import make_goal_pose_w, is_pose_close_xy

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    assert num_envs == 1, "Recorder only supports single env for now"

    # =========================================================================
    # Step 1: Reset environment
    # =========================================================================
    obs_dict, _ = env.reset()

    # Get initial EE pose (rest pose)
    ee_pose = get_ee_pose_w(env)

    # =========================================================================
    # Step 2: Teleport cube to goal position
    # =========================================================================
    goal_pose = make_goal_pose_w(env, task_spec.goal_xy, z=0.055)
    teleport_object_to_pose(env, goal_pose, name="object")

    # Let physics settle after teleport
    zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    for _ in range(10):
        env.step(zero_action)

    # =========================================================================
    # Step 3: Sample random place target on table
    # =========================================================================
    place_xy = task_spec.sample_table_xy(rng)
    place_pose = np.array([place_xy[0], place_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal_pose_np = np.array([task_spec.goal_xy[0], task_spec.goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # =========================================================================
    # Step 4: Initialize expert
    # =========================================================================
    ee_pose = get_ee_pose_w(env)
    expert.reset(ee_pose, place_xy, place_z=0.055)

    # =========================================================================
    # Step 5: Run episode and record
    # =========================================================================
    obs_list = []
    ee_pose_list = []
    obj_pose_list = []
    gripper_list = []

    expert_completed = False  # 是否完成了完整的状态机流程

    for t in range(horizon):
        # Get current state
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)

        # Get observation vector (policy input)
        if isinstance(obs_dict, dict):
            # Isaac Lab returns dict with 'policy' key
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict

        # Record data
        obs_list.append(obs_vec[0].cpu().numpy())
        ee_pose_list.append(ee_pose[0].cpu().numpy())
        obj_pose_list.append(object_pose[0].cpu().numpy())

        # Compute expert action
        action = expert.act(ee_pose, object_pose)
        gripper_val = action[0, 7].item()
        gripper_list.append(gripper_val)

        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)

        # Check if expert finished (reached DONE state)
        if expert.is_done().all():
            expert_completed = True
            break

    # =========================================================================
    # Step 6: Settle steps (only if expert completed, keep recording)
    # =========================================================================
    if expert_completed:
        for t in range(settle_steps):
            ee_pose = get_ee_pose_w(env)
            object_pose = get_object_pose_w(env)

            if isinstance(obs_dict, dict):
                obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
                if obs_vec is None:
                    obs_vec = next(iter(obs_dict.values()))
            else:
                obs_vec = obs_dict

            obs_list.append(obs_vec[0].cpu().numpy())
            ee_pose_list.append(ee_pose[0].cpu().numpy())
            obj_pose_list.append(object_pose[0].cpu().numpy())
            gripper_list.append(1.0)  # Open gripper during settle

            # Use rest action during settle
            rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
            rest_action[:, :7] = expert.rest_pose
            rest_action[:, 7] = 1.0  # Open gripper

            obs_dict, _, _, _, _ = env.step(rest_action)

    # =========================================================================
    # Step 7: Check success (cube near target position)
    # =========================================================================
    object_pose = get_object_pose_w(env)
    success = is_pose_close_xy(object_pose, place_xy, task_spec.success_radius)
    success_bool = success[0].item()

    # =========================================================================
    # Step 8: Build Episode
    # =========================================================================
    episode = Episode(
        obs=np.array(obs_list, dtype=np.float32),
        ee_pose=np.array(ee_pose_list, dtype=np.float32),
        obj_pose=np.array(obj_pose_list, dtype=np.float32),
        gripper=np.array(gripper_list, dtype=np.float32),
        place_pose=place_pose,
        goal_pose=goal_pose_np,
        success=success_bool,
    )

    return episode, expert_completed
