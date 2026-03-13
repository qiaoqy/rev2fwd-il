#!/usr/bin/env python3
"""Collect FORWARD Task A data: Pick from random table position → Place at goal.

This script is a variant of 1_collect_data_pick_place.py (Task B: goal→random)
adapted for direct forward Task A collection (random→goal).

Differences from script 1:
  - Object starts at a RANDOM table position (not at goal)
  - Place target is FIXED at goal_xy (0.5, 0.0) (not random)
  - Success: object near goal after placing

Uses the SAME PickPlaceExpertB FSM (it's a generic pick→place expert).

=============================================================================
USAGE
=============================================================================
CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place/1a_collect_forward_task_A.py \
    --headless --num_episodes 300 --num_envs 20 \
    --out data/pick_place_isaac_lab_simulation/exp8/A_forward_300.npz
"""

from __future__ import annotations

import argparse
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect forward Task A rollouts (random→goal) using FSM expert.",
    )
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/A_forward_300.npz")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def rollout_forward_task_A(
    env,
    expert,
    task_spec,
    rng,
    horizon: int,
    settle_steps: int = 40,
    markers=None,
):
    """Run parallel FORWARD rollouts: pick from random → place at goal.

    Key difference from script 1's rollout:
      - Object teleported to RANDOM table position (not goal)
      - Expert place target = goal_xy (not random)
      - Success = object near goal
    """
    import numpy as np
    import torch

    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import make_goal_pose_w

    # Reuse make_env helpers from script 1
    script1_dir = os.path.dirname(__file__)
    sys.path.insert(0, script1_dir)
    from importlib.util import spec_from_file_location, module_from_spec
    _spec = spec_from_file_location(
        "collect_b", os.path.join(script1_dir, "1_collect_data_pick_place.py"),
    )
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    get_fsm_goal_action = _mod.get_fsm_goal_action
    create_target_markers = _mod.create_target_markers
    update_target_markers = _mod.update_target_markers

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

    # === Reset environment ===
    obs_dict, _ = env.reset()
    ee_pose = get_ee_pose_w(env)

    # === Teleport cube to RANDOM table position (not goal!) ===
    min_dist_from_goal = 0.1
    start_xys = []
    for _ in range(num_envs):
        while True:
            xy = task_spec.sample_table_xy(rng)
            dist = np.sqrt((xy[0] - task_spec.goal_xy[0])**2 + (xy[1] - task_spec.goal_xy[1])**2)
            if dist >= min_dist_from_goal:
                start_xys.append(xy)
                break

    # Teleport each env's object to its random start position
    start_poses = torch.zeros(num_envs, 7, device=device)
    for i, xy in enumerate(start_xys):
        start_poses[i, 0] = xy[0]
        start_poses[i, 1] = xy[1]
        start_poses[i, 2] = 0.022  # cube half-height on table
        start_poses[i, 3] = 1.0    # identity quaternion w
    teleport_object_to_pose(env, start_poses, name="object")

    # Let physics settle
    zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    for _ in range(10):
        env.step(zero_action)

    # Prepare pose arrays for saving
    goal_xy = task_spec.goal_xy
    start_poses_np = np.array([
        [xy[0], xy[1], 0.055, 1.0, 0.0, 0.0, 0.0] for xy in start_xys
    ], dtype=np.float32)
    goal_pose_np = np.array(
        [goal_xy[0], goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    # === Initialize expert: place target = GOAL (not random!) ===
    ee_pose = get_ee_pose_w(env)
    expert.reset(ee_pose, goal_xy, place_z=0.055)
    # All envs have the same place target (goal), already set by reset

    # === Markers ===
    if markers is None:
        start_markers, goal_markers, marker_z = create_target_markers(num_envs, device)
        markers = (start_markers, goal_markers, marker_z)
    else:
        start_markers, goal_markers, marker_z = markers
    update_target_markers(
        start_markers, goal_markers,
        start_xys=start_xys,
        goal_xy=goal_xy,
        marker_z=marker_z,
        env=env,
    )

    # === Recording buffers ===
    obs_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    wrist_image_lists = [[] for _ in range(num_envs)]
    ee_pose_lists = [[] for _ in range(num_envs)]
    obj_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    gripper_lists = [[] for _ in range(num_envs)]
    fsm_state_lists = [[] for _ in range(num_envs)]

    expert_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # === Main loop ===
    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"  Step {t+1}/{horizon}")
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)

        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict

        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        goal_action = get_fsm_goal_action(expert, object_pose)

        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
        goal_action_np = goal_action.cpu().numpy()
        fsm_states_np = expert.state.cpu().numpy()

        for i in range(num_envs):
            obs_lists[i].append(obs_np[i])
            image_lists[i].append(table_images_np[i])
            wrist_image_lists[i].append(wrist_images_np[i])
            ee_pose_lists[i].append(ee_pose_np[i])
            obj_pose_lists[i].append(obj_pose_np[i])
            action_lists[i].append(goal_action_np[i])
            gripper_lists[i].append(goal_action_np[i, 7])
            fsm_state_lists[i].append(fsm_states_np[i])

        action = expert.act(ee_pose, object_pose)
        obs_dict, _, _, _, _ = env.step(action)

        just_done = expert.is_done() & ~expert_completed
        expert_completed = expert_completed | expert.is_done()

        if expert_completed.all():
            print(f"  All envs done at step {t+1}")
            break

    # === Settle steps ===
    for t in range(settle_steps):
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)
        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict

        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        goal_action = get_fsm_goal_action(expert, object_pose)

        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
        goal_action_np = goal_action.cpu().numpy()
        fsm_states_np = expert.state.cpu().numpy()

        for i in range(num_envs):
            if expert_completed[i]:
                obs_lists[i].append(obs_np[i])
                image_lists[i].append(table_images_np[i])
                wrist_image_lists[i].append(wrist_images_np[i])
                ee_pose_lists[i].append(ee_pose_np[i])
                obj_pose_lists[i].append(obj_pose_np[i])
                action_lists[i].append(goal_action_np[i])
                gripper_lists[i].append(1.0)
                fsm_state_lists[i].append(fsm_states_np[i])

        rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        rest_action[:, :7] = expert.rest_pose
        rest_action[:, 7] = 1.0
        obs_dict, _, _, _, _ = env.step(rest_action)

    # === Check success: object near GOAL ===
    object_pose = get_object_pose_w(env)
    results = []
    for i in range(num_envs):
        obj_xy = object_pose[i, :2].cpu().numpy()
        dist = np.sqrt((obj_xy[0] - goal_xy[0])**2 + (obj_xy[1] - goal_xy[1])**2)
        success_bool = dist < task_spec.success_radius

        episode_dict = {
            "obs": np.array(obs_lists[i], dtype=np.float32),
            "images": np.array(image_lists[i], dtype=np.uint8),
            "wrist_images": np.array(wrist_image_lists[i], dtype=np.uint8),
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "obj_pose": np.array(obj_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),
            "gripper": np.array(gripper_lists[i], dtype=np.float32),
            "fsm_state": np.array(fsm_state_lists[i], dtype=np.int32),
            "place_pose": goal_pose_np,       # place target = goal
            "goal_pose": goal_pose_np,        # goal = same as place for Task A
            "success": success_bool,
        }
        results.append((episode_dict, expert_completed[i].item()))

    return results, markers


def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    import torch

    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.utils.seed import set_seed

    # Reuse make_env_with_camera and save function from script 1
    from importlib.util import spec_from_file_location, module_from_spec
    _spec = spec_from_file_location(
        "collect_b",
        os.path.join(os.path.dirname(__file__), "1_collect_data_pick_place.py"),
    )
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    make_env_with_camera = _mod.make_env_with_camera
    save_episodes_with_goal_actions = _mod.save_episodes_with_goal_actions

    try:
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        num_envs = args.num_envs
        env = make_env_with_camera(
            task_id=args.task,
            num_envs=num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,
            disable_terminations=True,
        )
        device = env.unwrapped.device

        task_spec = PickPlaceTaskSpec(
            goal_xy=(0.5, 0.0),
            hover_z=0.25,
            grasp_z_offset=0.0,
            success_radius=0.03,
            settle_steps=10,
        )

        expert = PickPlaceExpertB(
            num_envs=num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            release_z_offset=-0.04,
            position_threshold=0.015,
            wait_steps=task_spec.settle_steps,
        )

        episodes = []
        success_count = 0
        completed_count = 0
        markers = None

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} FORWARD Task A rollouts (random→goal)")
        print(f"  num_envs: {num_envs}")
        print(f"  horizon: {args.horizon}")
        print(f"  settle_steps: {args.settle_steps}")
        print(f"  goal_xy: {task_spec.goal_xy}")
        print(f"{'='*60}\n")

        start_time = time.time()
        batch_count = 0
        max_batches = (args.num_episodes // num_envs + 1) * 3

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            results, markers = rollout_forward_task_A(
                env=env,
                expert=expert,
                task_spec=task_spec,
                rng=rng,
                horizon=args.horizon,
                settle_steps=args.settle_steps,
                markers=markers,
            )

            batch_completed = 0
            batch_success = 0
            for episode_dict, expert_completed_flag in results:
                if expert_completed_flag:
                    completed_count += 1
                    batch_completed += 1
                    episodes.append(episode_dict)
                    if episode_dict["success"]:
                        success_count += 1
                        batch_success += 1
                    if len(episodes) >= args.num_episodes:
                        break

            elapsed = time.time() - start_time
            total_attempts = batch_count * num_envs
            rate = total_attempts / elapsed if elapsed > 0 else 0
            print(
                f"Batch {batch_count:3d} | "
                f"Saved: {len(episodes)}/{args.num_episodes} | "
                f"This batch: {batch_completed}/{num_envs} ok, {batch_success} success | "
                f"Rate: {rate:.1f} ep/s"
            )

        elapsed = time.time() - start_time
        total_attempts = batch_count * num_envs
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Total batches: {batch_count}, attempts: {total_attempts}")
        print(f"Saved episodes: {len(episodes)}")
        print(f"Success: {success_count}/{len(episodes)} "
              f"({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)
        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
