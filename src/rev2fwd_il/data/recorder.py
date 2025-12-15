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


def rollout_expert_B_reverse_parallel(
    env: gym.Env,
    expert: "PickPlaceExpertB",
    task_spec: "PickPlaceTaskSpec",
    rng: np.random.Generator,
    horizon: int,
    settle_steps: int = 30,
) -> list[tuple[Episode, bool]]:
    """Run parallel reverse rollouts with Expert B and record trajectories.

    This function runs multiple environments in parallel for faster data collection.
    Each environment independently executes the reverse pick-and-place task.

    Args:
        env: Isaac Lab gymnasium environment with num_envs >= 1.
        expert: PickPlaceExpertB instance.
        task_spec: Task specification with goal/table bounds.
        rng: NumPy random generator.
        horizon: Maximum steps for each episode.
        settle_steps: Extra steps after expert finishes to let cube settle.

    Returns:
        List of (Episode, expert_completed) tuples, one per environment.
    """
    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import make_goal_pose_w, is_pose_close_xy

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # =========================================================================
    # Step 1: Reset environment
    # =========================================================================
    obs_dict, _ = env.reset()
    ee_pose = get_ee_pose_w(env)

    # =========================================================================
    # Step 2: Teleport cube to goal position for all envs
    # =========================================================================
    goal_pose = make_goal_pose_w(env, task_spec.goal_xy, z=0.055)
    teleport_object_to_pose(env, goal_pose, name="object")

    # Let physics settle after teleport
    zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    for _ in range(10):
        env.step(zero_action)

    # =========================================================================
    # Step 3: Sample random place targets for each env
    # =========================================================================
    place_xys = [task_spec.sample_table_xy(rng) for _ in range(num_envs)]
    place_poses_np = np.array([
        [xy[0], xy[1], 0.055, 1.0, 0.0, 0.0, 0.0] for xy in place_xys
    ], dtype=np.float32)
    goal_pose_np = np.array(
        [task_spec.goal_xy[0], task_spec.goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], 
        dtype=np.float32
    )

    # =========================================================================
    # Step 4: Initialize expert for each env with different place targets
    # =========================================================================
    ee_pose = get_ee_pose_w(env)
    # Reset expert with first place_xy, then update place_pose for each env
    expert.reset(ee_pose, place_xys[0], place_z=0.055)
    # Update place_pose for all envs
    for i, xy in enumerate(place_xys):
        expert.place_pose[i, 0] = xy[0]
        expert.place_pose[i, 1] = xy[1]

    # =========================================================================
    # Step 5: Initialize per-env recording buffers
    # =========================================================================
    obs_lists = [[] for _ in range(num_envs)]
    ee_pose_lists = [[] for _ in range(num_envs)]
    obj_pose_lists = [[] for _ in range(num_envs)]
    gripper_lists = [[] for _ in range(num_envs)]
    
    expert_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    done_at_step = torch.full((num_envs,), horizon + settle_steps, dtype=torch.int32, device=device)

    # =========================================================================
    # Step 6: Run episode and record
    # =========================================================================
    for t in range(horizon):
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)

        # Get observation vector
        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict

        # Record data for each env
        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        
        for i in range(num_envs):
            obs_lists[i].append(obs_np[i])
            ee_pose_lists[i].append(ee_pose_np[i])
            obj_pose_lists[i].append(obj_pose_np[i])

        # Compute expert action
        action = expert.act(ee_pose, object_pose)
        gripper_vals = action[:, 7].cpu().numpy()
        for i in range(num_envs):
            gripper_lists[i].append(gripper_vals[i])

        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)

        # Check which envs just finished
        just_done = expert.is_done() & ~expert_completed
        expert_completed = expert_completed | expert.is_done()
        done_at_step[just_done] = t + 1

        # Early exit if all envs are done
        if expert_completed.all():
            break

    # =========================================================================
    # Step 7: Settle steps (continue recording for all envs)
    # =========================================================================
    for t in range(settle_steps):
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)

        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict

        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()

        for i in range(num_envs):
            # Only record settle steps for envs that completed
            if expert_completed[i]:
                obs_lists[i].append(obs_np[i])
                ee_pose_lists[i].append(ee_pose_np[i])
                obj_pose_lists[i].append(obj_pose_np[i])
                gripper_lists[i].append(1.0)  # Open gripper during settle

        # Rest action during settle
        rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        rest_action[:, :7] = expert.rest_pose
        rest_action[:, 7] = 1.0  # Open gripper

        obs_dict, _, _, _, _ = env.step(rest_action)

    # =========================================================================
    # Step 8: Check success and build episodes
    # =========================================================================
    object_pose = get_object_pose_w(env)
    results = []

    for i in range(num_envs):
        # Check success for this env
        place_xy = place_xys[i]
        obj_xy = object_pose[i, :2].cpu().numpy()
        dist = np.sqrt((obj_xy[0] - place_xy[0])**2 + (obj_xy[1] - place_xy[1])**2)
        success_bool = dist < task_spec.success_radius

        episode = Episode(
            obs=np.array(obs_lists[i], dtype=np.float32),
            ee_pose=np.array(ee_pose_lists[i], dtype=np.float32),
            obj_pose=np.array(obj_pose_lists[i], dtype=np.float32),
            gripper=np.array(gripper_lists[i], dtype=np.float32),
            place_pose=place_poses_np[i],
            goal_pose=goal_pose_np,
            success=success_bool,
        )
        results.append((episode, expert_completed[i].item()))

    return results


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
