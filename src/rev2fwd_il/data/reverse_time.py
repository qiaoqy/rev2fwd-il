"""Time reversal utilities for building forward BC dataset from reverse rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec

from .episode import Episode


# Gripper action values
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


def infer_gripper_labels(
    ee_pos_fwd: np.ndarray,
    obj_pos_fwd: np.ndarray,
    goal_xy: tuple[float, float],
    table_z: float = 0.055,
    grasp_dist: float = 0.035,
    goal_release_radius: float = 0.05,
    z_margin: float = 0.03,
) -> np.ndarray:
    """Infer gripper labels for forward trajectory using heuristics.

    The forward task (A) should:
        - Start with OPEN gripper
        - CLOSE when near the object (grasp)
        - Keep CLOSE while transporting
        - OPEN when object is at goal and lowered

    Args:
        ee_pos_fwd: Forward EE positions, shape (T, 3).
        obj_pos_fwd: Forward object positions, shape (T, 3).
        goal_xy: Goal (x, y) position.
        table_z: Table height (approximate using goal z).
        grasp_dist: Distance threshold for grasping.
        goal_release_radius: XY radius around goal for release.
        z_margin: Z margin above table for release detection.

    Returns:
        Gripper labels, shape (T,) with values +1 (OPEN) or -1 (CLOSE).
    """
    T = len(ee_pos_fwd)
    gripper = np.full(T, GRIPPER_OPEN, dtype=np.float32)

    # Find grasp_start: earliest t where EE is close to object
    grasp_start = None
    for t in range(T):
        dist = np.linalg.norm(ee_pos_fwd[t] - obj_pos_fwd[t])
        if dist < grasp_dist:
            grasp_start = t
            break

    if grasp_start is None:
        # Fallback: no grasp detected, return all OPEN
        return gripper

    # Find release_start: earliest t (after half of trajectory) where:
    # - object XY is close to goal
    # - object Z is low (near table)
    release_start = None
    search_start = max(grasp_start + 1, int(T * 0.5))

    for t in range(search_start, T):
        obj_xy = obj_pos_fwd[t, :2]
        obj_z = obj_pos_fwd[t, 2]
        dist_to_goal = np.linalg.norm(obj_xy - np.array(goal_xy))

        if dist_to_goal < goal_release_radius and obj_z < (table_z + z_margin):
            release_start = t
            break

    if release_start is None:
        # Fallback: force OPEN for last 20 steps
        release_start = max(grasp_start + 1, T - 20)

    # Assign gripper labels
    # OPEN before grasp
    gripper[:grasp_start] = GRIPPER_OPEN
    # CLOSE from grasp to release
    gripper[grasp_start:release_start] = GRIPPER_CLOSE
    # OPEN from release onwards
    gripper[release_start:] = GRIPPER_OPEN

    return gripper


def reverse_episode_build_forward_pairs(
    ep: Episode,
    task_spec: "PickPlaceTaskSpec",
) -> dict:
    """Reverse a B episode in time and build forward action labels for A.

    The reverse operation:
        1. Reverse observation/pose sequences in time
        2. Build EE action labels as "next-step EE pose"
        3. Infer gripper labels using heuristics (not simply reversing B's gripper)

    Args:
        ep: Episode from Expert B (reverse task).
        task_spec: Task specification with goal position.

    Returns:
        Dictionary with:
            - obs: Forward observations, shape (T-1, obs_dim)
            - act: Forward action labels, shape (T-1, 8) [ee_pose(7) + gripper(1)]
    """
    # A) Time reversal
    obs_fwd = ep.obs[::-1].copy()  # (T, obs_dim)
    ee_fwd = ep.ee_pose[::-1].copy()  # (T, 7)
    obj_fwd = ep.obj_pose[::-1].copy()  # (T, 7)

    T = len(obs_fwd)

    # B) EE action label: next-step EE pose
    # For training, obs[t] -> act[t] where act[t] is the target EE pose at t+1
    ee_label = ee_fwd[1:]  # (T-1, 7)

    # C) Infer gripper labels for forward trajectory
    ee_pos_fwd = ee_fwd[:, :3]  # (T, 3)
    obj_pos_fwd = obj_fwd[:, :3]  # (T, 3)

    # Note: In forward task, goal is ep.goal_pose (plate center)
    # In reverse task, goal is ep.place_pose (random table position)
    # For forward A, we want to reach goal_pose
    goal_xy = (ep.goal_pose[0], ep.goal_pose[1])
    table_z = ep.goal_pose[2]  # Use goal z as table height approximation

    gripper_full = infer_gripper_labels(
        ee_pos_fwd=ee_pos_fwd,
        obj_pos_fwd=obj_pos_fwd,
        goal_xy=goal_xy,
        table_z=table_z,
        grasp_dist=0.035,
        goal_release_radius=0.05,
        z_margin=0.03,
    )

    # Gripper label for action at time t is the gripper state at t+1
    gripper_label = gripper_full[1:]  # (T-1,)

    # D) Concatenate EE pose and gripper to form action
    act = np.concatenate([ee_label, gripper_label[:, None]], axis=-1)  # (T-1, 8)

    # Observations for training (exclude last timestep since no action for it)
    obs_out = obs_fwd[:-1]  # (T-1, obs_dim)

    return {
        "obs": obs_out.astype(np.float32),
        "act": act.astype(np.float32),
    }
