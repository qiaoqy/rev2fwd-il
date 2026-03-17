#!/usr/bin/env python3
"""Collect FORWARD Task A data: Pick from red region → Place at green (goal).

This script is the forward counterpart of 1_collect_data_pick_place.py (Task B).
Task B collects reverse trajectories (green→red), while Task A collects forward
trajectories (red→green).

Direction comparison:
  Script 1  (Task B): Object at GREEN (goal) → pick → place at RED  (random/region)
  Script 1a (Task A): Object at RED (random/region) → pick → place at GREEN (goal)

Action is recorded as next-frame ee_pose (ee_pose[t+1]) + gripper command,
identical to script 1's recording format.

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
For each episode, the following arrays are saved:
    - obs:           (T, 36)  Policy observation sequence
    - images:        (T, H, W, 3)  RGB images from table camera (uint8)
    - wrist_images:  (T, H, W, 3)  RGB images from wrist camera (uint8)
    - ee_pose:       (T, 7)   End-effector pose [x, y, z, qw, qx, qy, qz]
    - obj_pose:      (T, 7)   Object (cube) pose
    - action:        (T, 8)   Next-frame ee_pose + gripper [x,y,z,qw,qx,qy,qz, gripper]
    - gripper:       (T,)     Gripper action (+1=open, -1=close)
    - fsm_state:     (T,)     FSM state at each timestep (int)
    - place_pose:    (7,)     Target place position (= goal position for Task A)
    - goal_pose:     (7,)     Goal position (plate center, same as place_pose)
    - start_pose:    (7,)     Object start position (red marker / pick position)
    - success:       bool     Whether cube ended up near goal position

=============================================================================
MARKER MODES (three mutually exclusive modes)
=============================================================================
Green circle marker is always fixed at goal position (0.5, 0.0) — the PLACE target.
Red marker shows the PICK position (where cube starts):

Mode 1: Random red circle (DEFAULT)
  Cube starts at a randomly sampled table position each episode.
  Expert picks cube from red circle, places it at green goal.

  python scripts/scripts_pick_place/1a_collect_forward_task_A.py \
      --num_episodes 100 --headless

Mode 2: Fixed red circle
  Cube always starts at a fixed position (e.g., 0.45, 0.15).
  Expert picks from fixed red circle, places at green goal.

  python scripts/scripts_pick_place/1a_collect_forward_task_A.py \
      --fixed_start_xy 0.45 0.15 \
      --num_episodes 20 --headless

Mode 3: Red rectangle region
  Cube starts at random position within a rectangular region.
  Red rectangle marker shows the pick region.

  python scripts/scripts_pick_place/1a_collect_forward_task_A.py \
      --start_region_mode red_region \
      --red_region_center_xy 0.5 0.0 \
      --red_region_size_xy 0.12 0.10 \
      --fix_red_marker_pose 1 \
      --num_episodes 20 --headless

NOTE: --fixed_start_xy (Mode 2) and --start_region_mode red_region (Mode 3)
are mutually exclusive. Do not use them together.

=============================================================================
"""

from __future__ import annotations

import argparse
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect forward Task A rollouts (red→green) using FSM expert.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -----------------------------------------------------------------
    # Task and environment configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Abs-v0",
        help="Isaac Lab Gym task ID. Must use IK-Abs control.",
    )

    # -----------------------------------------------------------------
    # Image configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="Width of captured images.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=128,
        help="Height of captured images.",
    )

    # -----------------------------------------------------------------
    # Data collection parameters
    # -----------------------------------------------------------------
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=300,
        help="Number of episodes to collect.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="Maximum simulation steps per episode (before settle steps).",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=40,
        help="Additional steps after expert finishes to let the cube settle.",
    )

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    # -----------------------------------------------------------------
    # Start position / pick region modes
    # -----------------------------------------------------------------
    parser.add_argument(
        "--fixed_start_xy",
        type=float,
        nargs=2,
        default=None,
        help="[Mode 2] Fixed (x, y) for cube start position and red circle marker. "
             "Mutually exclusive with --start_region_mode red_region.",
    )
    parser.add_argument(
        "--start_region_mode",
        type=str,
        choices=["legacy", "red_region"],
        default="legacy",
        help="[Mode 3] Set to 'red_region' to sample cube start positions within "
             "a rectangular region. Requires --red_region_center_xy and "
             "--red_region_size_xy. Mutually exclusive with --fixed_start_xy.",
    )
    parser.add_argument(
        "--red_region_center_xy",
        type=float,
        nargs=2,
        default=None,
        help="[Mode 3] Center (cx, cy) of the red rectangle region. "
             "Required when --start_region_mode=red_region.",
    )
    parser.add_argument(
        "--red_region_size_xy",
        type=float,
        nargs=2,
        default=None,
        help="[Mode 3] Size (sx, sy) in meters of the red rectangle region. "
             "Required when --start_region_mode=red_region.",
    )
    parser.add_argument(
        "--red_marker_size_xy",
        type=float,
        nargs=2,
        default=None,
        help="[Mode 3 only] Red rectangle marker display size (sx, sy) in meters. "
             "Auto-defaults to --red_region_size_xy if not specified.",
    )
    parser.add_argument(
        "--fix_red_marker_pose",
        type=int,
        choices=[0, 1],
        default=0,
        help="[Mode 3] If 1, red rectangle marker stays at region center while "
             "cube start positions are randomly sampled within the region.",
    )

    # -----------------------------------------------------------------
    # Output configuration
    # -----------------------------------------------------------------
    parser.add_argument(
        "--out",
        type=str,
        default="data/A_forward_300.npz",
        help="Output path for the NPZ file containing collected episodes.",
    )

    # -----------------------------------------------------------------
    # Simulation backend options
    # -----------------------------------------------------------------
    parser.add_argument(
        "--disable_fabric",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, disables Fabric backend (PhysX GPU acceleration).",
    )

    # -----------------------------------------------------------------
    # Isaac Lab AppLauncher arguments (--headless, --device, etc.)
    # -----------------------------------------------------------------
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    args.enable_cameras = True
    return args


def _import_script1():
    """Import functions from script 1 (1_collect_data_pick_place.py)."""
    from importlib.util import spec_from_file_location, module_from_spec
    script1_path = os.path.join(
        os.path.dirname(__file__), "1_collect_data_pick_place.py",
    )
    _spec = spec_from_file_location("collect_b", script1_path)
    _mod = module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


def rollout_forward_task_A(
    env,
    expert,
    task_spec,
    rng,
    horizon: int,
    settle_steps: int = 40,
    markers=None,
):
    """Run parallel FORWARD rollouts: pick from red → place at green (goal).

    Key differences from script 1's rollout_expert_B_with_goal_actions:
      - Object teleported to RED position (random/fixed/region), NOT goal
      - Expert place target = goal_xy (green), NOT random
      - Success = object near goal (green)
      - Actions recorded as next-frame ee_pose (same format as script 1)
    """
    import numpy as np
    import torch

    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import make_goal_pose_w

    # Import reusable functions from script 1
    script1 = _import_script1()
    get_fsm_goal_action = script1.get_fsm_goal_action
    create_target_markers = script1.create_target_markers
    update_target_markers = script1.update_target_markers

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    print(f"[DEBUG] rollout: device={device}, num_envs={num_envs}")

    # Get camera sensor references
    table_camera = env.unwrapped.scene.sensors["table_cam"]
    wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

    # =========================================================================
    # Step 1: Reset environment
    # =========================================================================
    print("[DEBUG] rollout: Resetting environment...")
    obs_dict, _ = env.reset()
    ee_pose = get_ee_pose_w(env)

    # Test camera data access
    try:
        test_rgb = table_camera.data.output["rgb"]
        print(f"[DEBUG] rollout: Table camera RGB shape: {test_rgb.shape}")
        test_wrist_rgb = wrist_camera.data.output["rgb"]
        print(f"[DEBUG] rollout: Wrist camera RGB shape: {test_wrist_rgb.shape}")
    except Exception as e:
        print(f"[ERROR] rollout: Failed to access camera data: {e}")

    # =========================================================================
    # Step 2: Sample START positions for each env (cube pick positions)
    #         This is where the cube will be teleported — the RED marker position.
    # =========================================================================
    min_dist_from_goal = 0.1
    start_region_mode = getattr(task_spec, "start_region_mode", "legacy")
    fixed_start_xy = getattr(task_spec, "fixed_start_xy", None)
    red_center = getattr(task_spec, "red_region_center_xy", None)
    red_size = getattr(task_spec, "red_region_size_xy", None)

    # Cube half-size: DexCube default 0.05m * scale 0.8 = 0.04m edge, half = 0.02m
    cube_half_size = 0.02

    if start_region_mode == "red_region" and red_center is not None and red_size is not None:
        # Mode 3: Sample within rectangular red region, shrunk by cube_half_size
        # so the cube body stays fully inside the rectangle.
        cx, cy = float(red_center[0]), float(red_center[1])
        sx, sy = float(red_size[0]), float(red_size[1])
        sample_half_x = max(sx * 0.5 - cube_half_size, 0.0)
        sample_half_y = max(sy * 0.5 - cube_half_size, 0.0)
        start_xys = [
            (rng.uniform(cx - sample_half_x, cx + sample_half_x),
             rng.uniform(cy - sample_half_y, cy + sample_half_y))
            for _ in range(num_envs)
        ]
        print(f"[INFO] Mode 3 sampling: region=({sx},{sy}), cube_half={cube_half_size}, "
              f"effective sample range x=[{cx-sample_half_x:.4f},{cx+sample_half_x:.4f}], "
              f"y=[{cy-sample_half_y:.4f},{cy+sample_half_y:.4f}]")
    elif fixed_start_xy is not None:
        # Mode 2: Fixed start position
        fx = float(fixed_start_xy[0])
        fy = float(fixed_start_xy[1])
        dist = np.sqrt((fx - task_spec.goal_xy[0])**2 + (fy - task_spec.goal_xy[1])**2)
        if dist < min_dist_from_goal:
            raise ValueError(
                f"fixed_start_xy=({fx:.3f}, {fy:.3f}) violates min distance "
                f"{min_dist_from_goal:.3f} from goal {task_spec.goal_xy}."
            )
        start_xys = [(fx, fy) for _ in range(num_envs)]
    else:
        # Mode 1: Random table position
        start_xys = []
        for _ in range(num_envs):
            while True:
                xy = task_spec.sample_table_xy(rng)
                dist = np.sqrt((xy[0] - task_spec.goal_xy[0])**2 + (xy[1] - task_spec.goal_xy[1])**2)
                if dist >= min_dist_from_goal:
                    start_xys.append(xy)
                    break

    # =========================================================================
    # Step 3: Teleport cube to START (red) position
    # =========================================================================
    print("[DEBUG] rollout: Teleporting cube to start (red) positions...")
    start_poses = torch.zeros(num_envs, 7, device=device)
    for i, xy in enumerate(start_xys):
        start_poses[i, 0] = xy[0]
        start_poses[i, 1] = xy[1]
        start_poses[i, 2] = 0.055  # cube on table surface
        start_poses[i, 3] = 1.0    # identity quaternion w
    teleport_object_to_pose(env, start_poses, name="object")

    # Let physics settle — hold current ee_pose (not zero action)
    print("[DEBUG] rollout: Settling physics (10 steps, hold ee_pose)...")
    ee_hold = get_ee_pose_w(env)
    hold_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    hold_action[:, :7] = ee_hold[:, :7]
    hold_action[:, 7] = 1.0  # gripper open
    for _ in range(10):
        env.step(hold_action)
    print("[DEBUG] rollout: Physics settled")

    # Prepare pose arrays for saving
    goal_xy = task_spec.goal_xy
    start_poses_np = np.array([
        [xy[0], xy[1], 0.055, 1.0, 0.0, 0.0, 0.0] for xy in start_xys
    ], dtype=np.float32)
    goal_pose_np = np.array(
        [goal_xy[0], goal_xy[1], 0.055, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    # =========================================================================
    # Step 4: Initialize expert — place target = GOAL (green)
    # =========================================================================
    ee_pose = get_ee_pose_w(env)
    expert.reset(ee_pose, goal_xy, place_z=0.055)
    # All envs have the same place target (goal), already set by reset

    # =========================================================================
    # Step 4.1: Pre-position robot to rest pose (gripper-down) before recording
    # =========================================================================
    print("[DEBUG] rollout: Pre-positioning robot to gripper-down rest pose...")
    prepos_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    prepos_action[:, :7] = expert.rest_pose[:, :7]
    prepos_action[:, 7] = 1.0  # gripper open
    prepos_steps = 80
    for step_i in range(prepos_steps):
        obs_dict, _, _, _, _ = env.step(prepos_action)
        if (step_i + 1) % 20 == 0:
            cur_ee = get_ee_pose_w(env)
            pos_err = torch.norm(cur_ee[:, :3] - expert.rest_pose[:, :3], dim=-1).max().item()
            print(f"[DEBUG] rollout: Pre-position step {step_i+1}/{prepos_steps}, max pos error: {pos_err:.4f}")
    # Hold at rest pose for 10 extra steps to ensure zero velocity
    for _ in range(10):
        obs_dict, _, _, _, _ = env.step(prepos_action)
    print("[DEBUG] rollout: Pre-positioning done (robot at rest, zero velocity)")

    # =========================================================================
    # Step 4.5: Create/update visual markers
    # =========================================================================
    if markers is None:
        print("[DEBUG] rollout: Creating target markers (first time)...")
        start_markers, goal_markers, marker_z = create_target_markers(
            num_envs,
            device,
            red_marker_shape=getattr(task_spec, "red_marker_shape", "circle"),
            red_marker_size_xy=getattr(task_spec, "red_marker_size_xy", None),
        )
        markers = (start_markers, goal_markers, marker_z)
    else:
        print("[DEBUG] rollout: Reusing existing markers...")
        start_markers, goal_markers, marker_z = markers

    # Red marker = start/pick positions
    marker_xys = start_xys
    if bool(getattr(task_spec, "fix_red_marker_pose", False)) and red_center is not None:
        marker_xys = [(float(red_center[0]), float(red_center[1])) for _ in range(num_envs)]

    update_target_markers(
        start_markers, goal_markers,
        start_xys=marker_xys,
        goal_xy=task_spec.goal_xy,
        marker_z=marker_z,
        env=env,
    )
    print("[DEBUG] rollout: Target markers created and positioned")

    # =========================================================================
    # Step 5: Initialize recording buffers
    # =========================================================================
    obs_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    wrist_image_lists = [[] for _ in range(num_envs)]
    ee_pose_lists = [[] for _ in range(num_envs)]
    obj_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    gripper_lists = [[] for _ in range(num_envs)]
    fsm_state_lists = [[] for _ in range(num_envs)]

    expert_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    done_at_step = torch.full((num_envs,), horizon + settle_steps, dtype=torch.int32, device=device)

    # =========================================================================
    # Step 6: Main loop — record next-frame ee_pose actions
    # =========================================================================
    print(f"[DEBUG] rollout: Starting main loop (horizon={horizon})...")
    for t in range(horizon):
        if t == 0 or (t + 1) % 50 == 0:
            print(f"[DEBUG] rollout: Step {t+1}/{horizon}")
        ee_pose = get_ee_pose_w(env)
        object_pose = get_object_pose_w(env)

        # Get observation vector
        if isinstance(obs_dict, dict):
            obs_vec = obs_dict.get("policy", obs_dict.get("obs", None))
            if obs_vec is None:
                obs_vec = next(iter(obs_dict.values()))
        else:
            obs_vec = obs_dict

        # Get camera images
        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        # Get FSM goal action (for gripper command)
        goal_action = get_fsm_goal_action(expert, object_pose)

        # Record observation data (action filled after env.step)
        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
        gripper_np = goal_action.cpu().numpy()[:, 7]  # gripper from FSM
        fsm_states_np = expert.state.cpu().numpy()

        for i in range(num_envs):
            obs_lists[i].append(obs_np[i])
            image_lists[i].append(table_images_np[i])
            wrist_image_lists[i].append(wrist_images_np[i])
            ee_pose_lists[i].append(ee_pose_np[i])
            obj_pose_lists[i].append(obj_pose_np[i])
            gripper_lists[i].append(gripper_np[i])
            fsm_state_lists[i].append(fsm_states_np[i])

        # Compute expert action (this also updates FSM state)
        action = expert.act(ee_pose, object_pose)

        # Step environment
        obs_dict, _, _, _, _ = env.step(action)

        # Record next-frame ee_pose as action[t] = ee_pose[t+1]
        next_ee_pose = get_ee_pose_w(env)
        next_ee_pose_np = next_ee_pose.cpu().numpy()
        for i in range(num_envs):
            act = np.zeros(8, dtype=np.float32)
            act[:7] = next_ee_pose_np[i]
            act[7] = gripper_np[i]
            action_lists[i].append(act)

        # Check completion
        just_done = expert.is_done() & ~expert_completed
        expert_completed = expert_completed | expert.is_done()
        done_at_step[just_done] = t + 1

        if expert_completed.all():
            print(f"[DEBUG] rollout: All envs done at step {t+1}")
            break

    print(f"[DEBUG] rollout: Main loop finished. expert_completed={expert_completed}")

    # =========================================================================
    # Step 7: Settle steps (continue recording for completed envs)
    # =========================================================================
    print(f"[DEBUG] rollout: Starting settle steps ({settle_steps})...")
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

        obs_np = obs_vec.cpu().numpy()
        ee_pose_np = ee_pose.cpu().numpy()
        obj_pose_np = object_pose.cpu().numpy()
        table_images_np = table_rgb.cpu().numpy().astype(np.uint8)
        wrist_images_np = wrist_rgb.cpu().numpy().astype(np.uint8)
        fsm_states_np = expert.state.cpu().numpy()

        for i in range(num_envs):
            if expert_completed[i]:
                obs_lists[i].append(obs_np[i])
                image_lists[i].append(table_images_np[i])
                wrist_image_lists[i].append(wrist_images_np[i])
                ee_pose_lists[i].append(ee_pose_np[i])
                obj_pose_lists[i].append(obj_pose_np[i])
                gripper_lists[i].append(1.0)  # Open gripper during settle
                fsm_state_lists[i].append(fsm_states_np[i])

        # Rest action during settle
        rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        rest_action[:, :7] = expert.rest_pose
        rest_action[:, 7] = 1.0  # Open gripper

        obs_dict, _, _, _, _ = env.step(rest_action)

        # Record next-frame ee_pose as action[t]
        next_ee_pose = get_ee_pose_w(env)
        next_ee_pose_np = next_ee_pose.cpu().numpy()
        for i in range(num_envs):
            if expert_completed[i]:
                act = np.zeros(8, dtype=np.float32)
                act[:7] = next_ee_pose_np[i]
                act[7] = 1.0  # Open gripper during settle
                action_lists[i].append(act)

    print("[DEBUG] rollout: Settle steps finished")

    # =========================================================================
    # Step 8: Check success (object near GOAL / green) and build episode dicts
    # =========================================================================
    print("[DEBUG] rollout: Building episode dicts...")
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
            "place_pose": goal_pose_np,         # place target = goal (green)
            "goal_pose": goal_pose_np,          # goal = place for Task A
            "start_pose": start_poses_np[i],    # where cube started (red)
            "success": success_bool,
        }
        results.append((episode_dict, expert_completed[i].item()))

    print(f"[DEBUG] rollout: Done. Returning {len(results)} episodes")
    return results, markers


def main():
    args = _parse_args()

    # =====================================================================
    # Step 1: Launch Isaac Sim
    # =====================================================================
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # =====================================================================
    # Step 2: Import modules (must be after Isaac Sim initialization)
    # =====================================================================
    import numpy as np

    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.experts.pickplace_expert_b import PickPlaceExpertB
    from rev2fwd_il.utils.seed import set_seed

    # Import make_env and save from script 1
    script1 = _import_script1()
    make_env_with_camera = script1.make_env_with_camera
    save_episodes_with_goal_actions = script1.save_episodes_with_goal_actions

    try:
        # =================================================================
        # Step 3: Set random seeds
        # =================================================================
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        # =================================================================
        # Step 4: Create environment with camera
        # =================================================================
        print(f"[DEBUG] Creating environment: task={args.task}, num_envs={args.num_envs}")
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

        # =================================================================
        # Step 5: Define task specification
        # =================================================================
        task_spec = PickPlaceTaskSpec(
            goal_xy=(0.5, 0.0),
            hover_z=0.25,
            grasp_z_offset=0.0,
            success_radius=0.03,
            settle_steps=10,
        )

        # Apply mode-specific settings
        if args.fixed_start_xy is not None:
            task_spec.fixed_start_xy = (float(args.fixed_start_xy[0]), float(args.fixed_start_xy[1]))
            print(f"[INFO] Mode 2 (fixed red circle): start_xy={task_spec.fixed_start_xy}")
        else:
            task_spec.fixed_start_xy = None

        task_spec.start_region_mode = args.start_region_mode
        task_spec.red_region_center_xy = (
            tuple(args.red_region_center_xy) if args.red_region_center_xy is not None else None
        )
        task_spec.red_region_size_xy = (
            tuple(args.red_region_size_xy) if args.red_region_size_xy is not None else None
        )
        task_spec.red_marker_shape = "circle"
        task_spec.red_marker_size_xy = (
            tuple(args.red_marker_size_xy) if args.red_marker_size_xy is not None else None
        )
        task_spec.fix_red_marker_pose = bool(args.fix_red_marker_pose)

        # Mode validation
        if args.start_region_mode == "red_region":
            if args.red_region_center_xy is None or args.red_region_size_xy is None:
                raise ValueError(
                    "Mode 3 (red_region): both --red_region_center_xy and "
                    "--red_region_size_xy are required."
                )
            if args.fixed_start_xy is not None:
                raise ValueError(
                    "Cannot combine --fixed_start_xy (Mode 2) with "
                    "--start_region_mode red_region (Mode 3). Choose one mode."
                )
            task_spec.red_marker_shape = "rectangle"
            if task_spec.red_marker_size_xy is None:
                task_spec.red_marker_size_xy = task_spec.red_region_size_xy
            print(
                f"[INFO] Mode 3 (rectangle region): "
                f"center={task_spec.red_region_center_xy}, "
                f"size={task_spec.red_region_size_xy}, "
                f"marker_size={task_spec.red_marker_size_xy}, "
                f"fix_marker_pose={task_spec.fix_red_marker_pose}"
            )
        elif args.fixed_start_xy is None:
            print("[INFO] Mode 1 (random red circle): default mode")

        # =================================================================
        # Step 6: Create Expert B (generic pick→place FSM)
        # =================================================================
        expert = PickPlaceExpertB(
            num_envs=num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            release_z_offset=-0.04,
            position_threshold=0.015,
            wait_steps=task_spec.settle_steps,
        )

        # =================================================================
        # Step 7: Data collection loop
        # =================================================================
        episodes = []
        success_count = 0
        completed_count = 0
        markers = None

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} FORWARD Task A rollouts (red→green)")
        print(f"Settings:")
        print(f"  - num_envs (parallel): {num_envs}")
        print(f"  - horizon: {args.horizon}")
        print(f"  - settle_steps: {args.settle_steps}")
        print(f"  - image_size: {args.image_width}x{args.image_height}")
        print(f"  - Action type: next-frame ee_pose (ee_pose[t+1])")
        print(f"  - goal_xy: {task_spec.goal_xy}")
        print(f"  - Only saving episodes with completed FSM (DONE state)")
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
                f"Batch {batch_count:3d} ({num_envs} envs) | "
                f"Saved: {len(episodes)}/{args.num_episodes} | "
                f"This batch: {batch_completed}/{num_envs} completed, {batch_success} success | "
                f"Rate: {rate:.1f} ep/s"
            )

        # =================================================================
        # Step 8: Print summary statistics
        # =================================================================
        elapsed = time.time() - start_time
        total_attempts = batch_count * num_envs
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Parallel envs: {num_envs}")
        print(f"Total batches: {batch_count}")
        print(f"Total attempts: {total_attempts}")
        print(f"Completed FSM: {completed_count} ({100*completed_count/total_attempts:.1f}%)")
        print(f"Saved episodes: {len(episodes)}")
        print(f"Success (cube at goal): {success_count}/{len(episodes)} "
              f"({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"Effective rate: {len(episodes)/elapsed:.2f} episodes/s")
        print(f"{'='*60}\n")

        # =================================================================
        # Step 9: Save collected episodes
        # =================================================================
        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)
        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
