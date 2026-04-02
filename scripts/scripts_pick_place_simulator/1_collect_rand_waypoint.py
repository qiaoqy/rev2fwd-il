#!/usr/bin/env python3
"""Collect Task B data using PickPlaceExpertRandWaypoint (enhanced randomisation).

Expert picks the cube from the green goal position and places it at a random
position inside the red rectangular region, with enhanced waypoint randomness
and an independent departure path after release.

Action convention: action[t][:7] = ee_pose[t+1], action[t][7] = gripper[t]

Usage:
    conda activate rev2fwd_il
    python scripts/scripts_pick_place_simulator/1_collect_rand_waypoint.py \
        --out data/pick_place_isaac_lab_simulation/exp29/task_B_cube.npz \
        --num_episodes 10 --num_envs 1 --headless
"""

from __future__ import annotations

import argparse
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Task B data with PickPlaceExpertRandWaypoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--settle_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path.")

    # Region parameters
    parser.add_argument("--goal_xy", type=float, nargs=2, default=[0.5, -0.2])
    parser.add_argument("--red_region_center_xy", type=float, nargs=2, default=[0.5, 0.2])
    parser.add_argument("--red_region_size_xy", type=float, nargs=2, default=[0.3, 0.3])
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save_log", action="store_true", help="Save detailed waypoint debug log.")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- imports that require Isaac Sim running ----
    import importlib.util
    from pathlib import Path

    import numpy as np
    import torch

    from rev2fwd_il.sim.task_spec import PickPlaceTaskSpec
    from rev2fwd_il.experts.pickplace_expert_rand_waypoint import (
        PickPlaceExpertRandWaypoint,
        ExpertState,
        GRIPPER_OPEN,
        GRIPPER_CLOSE,
    )
    from rev2fwd_il.utils.seed import set_seed

    # Reuse env / marker utilities from the original collection script
    _orig_spec = importlib.util.spec_from_file_location(
        "collect_orig",
        str(Path(__file__).resolve().parent.parent / "scripts_pick_place" / "1_collect_data_pick_place.py"),
    )
    _orig = importlib.util.module_from_spec(_orig_spec)
    _orig_spec.loader.exec_module(_orig)

    make_env_with_camera = _orig.make_env_with_camera
    create_target_markers = _orig.create_target_markers
    update_target_markers = _orig.update_target_markers
    save_episodes_with_goal_actions = _orig.save_episodes_with_goal_actions

    from rev2fwd_il.sim.scene_api import get_ee_pose_w, get_object_pose_w, teleport_object_to_pose
    from rev2fwd_il.sim.task_spec import make_goal_pose_w

    # ------------------------------------------------------------------
    # get_fsm_goal_action adapted for the new expert (query-only, no state update)
    # ------------------------------------------------------------------
    def get_fsm_goal_action(expert: PickPlaceExpertRandWaypoint, object_pose: torch.Tensor) -> torch.Tensor:
        num_envs = expert.num_envs
        device = expert.device
        goal_action = torch.zeros(num_envs, 8, device=device)

        obj_xy = object_pose[:, :2]
        obj_z = object_pose[:, 2]

        from rev2fwd_il.experts.pickplace_expert_rand_waypoint import _quat_mul
        perturbed_quat = _quat_mul(expert.grasp_quat.unsqueeze(0), expert.quat_perturbation)

        # ---- pick side (5 levels) ----
        approach_obj_pos = torch.zeros(num_envs, 3, device=device)
        approach_obj_pos[:, :2] = obj_xy + expert.approach_obj_dxy
        approach_obj_pos[:, 2] = expert.episode_hover_z + expert.approach_obj_extra_z

        above_obj_pos = torch.zeros(num_envs, 3, device=device)
        above_obj_pos[:, :2] = obj_xy + expert.above_obj_dxy
        above_obj_pos[:, 2] = expert.episode_hover_z + expert.above_obj_extra_z

        correct_obj_pos = torch.zeros(num_envs, 3, device=device)
        correct_obj_pos[:, :2] = obj_xy + expert.correct_obj_dxy
        correct_obj_pos[:, 2] = obj_z + expert.grasp_z_offset + expert.correct_obj_extra_z

        align_obj_pos = torch.zeros(num_envs, 3, device=device)
        align_obj_pos[:, :2] = obj_xy + expert.align_obj_dxy
        align_obj_pos[:, 2] = obj_z + expert.grasp_z_offset + expert.align_obj_extra_z

        at_obj_pos = torch.zeros(num_envs, 3, device=device)
        at_obj_pos[:, 0] = obj_xy[:, 0] + expert.grasp_dx
        at_obj_pos[:, 1] = obj_xy[:, 1]
        at_obj_pos[:, 2] = obj_z + expert.grasp_z_offset

        # ---- lift side (4 levels, ascending from grasp) ----
        lift_base_xy = expert.grasp_obj_xy if expert.grasp_obj_xy is not None else obj_xy
        lift_base_z = expert.grasp_obj_z if expert.grasp_obj_z is not None else obj_z

        lift_align_pos = torch.zeros(num_envs, 3, device=device)
        lift_align_pos[:, :2] = lift_base_xy + expert.lift_align_dxy
        lift_align_pos[:, 2] = lift_base_z + expert.grasp_z_offset + expert.lift_align_extra_z

        lift_correct_pos = torch.zeros(num_envs, 3, device=device)
        lift_correct_pos[:, :2] = lift_base_xy + expert.lift_correct_dxy
        lift_correct_pos[:, 2] = lift_base_z + expert.grasp_z_offset + expert.lift_correct_extra_z

        lift_above_pos = torch.zeros(num_envs, 3, device=device)
        lift_above_pos[:, :2] = lift_base_xy + expert.lift_above_dxy
        lift_above_pos[:, 2] = expert.episode_hover_z + expert.lift_above_extra_z

        lift_approach_pos = torch.zeros(num_envs, 3, device=device)
        lift_approach_pos[:, :2] = lift_base_xy + expert.lift_approach_dxy
        lift_approach_pos[:, 2] = expert.episode_hover_z + expert.lift_approach_extra_z

        # ---- place side (5 levels, descent) ----
        approach_place_pos = torch.zeros(num_envs, 3, device=device)
        approach_place_pos[:, :2] = expert.place_pose[:, :2] + expert.approach_place_dxy
        approach_place_pos[:, 2] = expert.episode_hover_z + expert.approach_place_extra_z

        above_place_pos = torch.zeros(num_envs, 3, device=device)
        above_place_pos[:, :2] = expert.place_pose[:, :2] + expert.above_place_dxy
        above_place_pos[:, 2] = expert.episode_hover_z + expert.above_place_extra_z

        correct_place_pos = torch.zeros(num_envs, 3, device=device)
        correct_place_pos[:, :2] = expert.place_pose[:, :2] + expert.correct_place_dxy
        correct_place_pos[:, 2] = expert.place_pose[:, 2] + expert.correct_place_extra_z

        align_place_pos = torch.zeros(num_envs, 3, device=device)
        align_place_pos[:, :2] = expert.place_pose[:, :2] + expert.align_place_dxy
        align_place_pos[:, 2] = expert.place_pose[:, 2] + expert.align_place_extra_z

        at_place_pos = expert.place_pose[:, :3].clone()
        at_place_pos[:, 0] = at_place_pos[:, 0] + expert.place_dx

        release_pos = expert.place_pose[:, :3].clone()
        release_pos[:, 0] = release_pos[:, 0] + expert.place_dx
        release_pos[:, 2] = expert.place_pose[:, 2] + expert.release_z_offset

        # ---- departure (4 levels) ----
        depart_align_pos = torch.zeros(num_envs, 3, device=device)
        depart_align_pos[:, :2] = expert.place_pose[:, :2] + expert.depart_align_dxy
        depart_align_pos[:, 2] = expert.episode_hover_z + expert.depart_align_extra_z

        depart_correct_pos = torch.zeros(num_envs, 3, device=device)
        depart_correct_pos[:, :2] = expert.place_pose[:, :2] + expert.depart_correct_dxy
        depart_correct_pos[:, 2] = expert.episode_hover_z + expert.depart_correct_extra_z

        depart_above_pos = torch.zeros(num_envs, 3, device=device)
        depart_above_pos[:, :2] = expert.place_pose[:, :2] + expert.depart_above_dxy
        depart_above_pos[:, 2] = expert.episode_hover_z + expert.depart_above_extra_z

        depart_approach_pos = torch.zeros(num_envs, 3, device=device)
        depart_approach_pos[:, :2] = expert.place_pose[:, :2] + expert.depart_approach_dxy
        depart_approach_pos[:, 2] = expert.episode_hover_z + expert.depart_approach_extra_z

        for state_val in ExpertState:
            mask = expert.state == state_val
            if not mask.any():
                continue

            if state_val == ExpertState.REST:
                goal_action[mask, :3] = expert.rest_pose[mask, :3]
                goal_action[mask, 3:7] = expert.rest_pose[mask, 3:7]
                goal_action[mask, 7] = GRIPPER_OPEN
            # ==== PICK ====
            elif state_val == ExpertState.APPROACH_OBJ:
                goal_action[mask, :3] = approach_obj_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.GO_ABOVE_OBJ:
                goal_action[mask, :3] = above_obj_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.CORRECT_OBJ:
                goal_action[mask, :3] = correct_obj_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.ALIGN_TO_OBJ:
                goal_action[mask, :3] = align_obj_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.GO_TO_OBJ:
                goal_action[mask, :3] = at_obj_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.CLOSE:
                goal_action[mask, :3] = at_obj_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            # ==== LIFT ====
            elif state_val == ExpertState.LIFT_ALIGN:
                goal_action[mask, :3] = lift_align_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.LIFT_CORRECT:
                goal_action[mask, :3] = lift_correct_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.LIFT_ABOVE:
                goal_action[mask, :3] = lift_above_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.LIFT_APPROACH:
                goal_action[mask, :3] = lift_approach_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_CLOSE
            # ==== PLACE ====
            elif state_val == ExpertState.APPROACH_PLACE:
                goal_action[mask, :3] = approach_place_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.GO_ABOVE_PLACE:
                goal_action[mask, :3] = above_place_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.CORRECT_PLACE:
                goal_action[mask, :3] = correct_place_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.ALIGN_TO_PLACE:
                goal_action[mask, :3] = align_place_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.GO_TO_PLACE:
                goal_action[mask, :3] = at_place_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.LOWER_TO_RELEASE:
                goal_action[mask, :3] = release_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_CLOSE
            elif state_val == ExpertState.OPEN:
                goal_action[mask, :3] = release_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_OPEN
            # ==== DEPARTURE ====
            elif state_val == ExpertState.DEPART_ALIGN:
                goal_action[mask, :3] = depart_align_pos[mask]
                goal_action[mask, 3:7] = expert.grasp_quat
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.DEPART_CORRECT:
                goal_action[mask, :3] = depart_correct_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.DEPART_ABOVE:
                goal_action[mask, :3] = depart_above_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val == ExpertState.DEPART_APPROACH:
                goal_action[mask, :3] = depart_approach_pos[mask]
                goal_action[mask, 3:7] = perturbed_quat[mask]
                goal_action[mask, 7] = GRIPPER_OPEN
            elif state_val in (ExpertState.RETURN_REST, ExpertState.DONE):
                goal_action[mask, :3] = expert.rest_pose[mask, :3]
                goal_action[mask, 3:7] = expert.rest_pose[mask, 3:7]
                goal_action[mask, 7] = GRIPPER_OPEN

        return goal_action

    # ==================================================================
    # rollout function (adapted from the original for the new expert)
    # ==================================================================
    def rollout(env, expert, task_spec, rng, horizon, settle_steps, markers, place_z=0.055, goal_z=0.055):
        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs

        table_camera = env.unwrapped.scene.sensors["table_cam"]
        wrist_camera = env.unwrapped.scene.sensors["wrist_cam"]

        # ---- reset env ----
        obs_dict, _ = env.reset()
        ee_pose = get_ee_pose_w(env)

        # ---- teleport cube to goal ----
        goal_pose = make_goal_pose_w(env, task_spec.goal_xy, z=goal_z)
        teleport_object_to_pose(env, goal_pose, name="object")

        hold_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        hold_action[:, :7] = get_ee_pose_w(env)[:, :7]
        hold_action[:, 7] = 1.0
        for _ in range(10):
            env.step(hold_action)

        # ---- sample place targets ----
        target_mode = getattr(task_spec, "taskB_target_mode", "legacy")
        red_center = getattr(task_spec, "red_region_center_xy", None)
        red_size = getattr(task_spec, "red_region_size_xy", None)
        cube_half_size = 0.02

        if target_mode == "red_region" and red_center is not None and red_size is not None:
            cx, cy = float(red_center[0]), float(red_center[1])
            sx, sy = float(red_size[0]), float(red_size[1])
            shx = max(sx * 0.5 - cube_half_size, 0.0)
            shy = max(sy * 0.5 - cube_half_size, 0.0)
            place_xys = [
                (rng.uniform(cx - shx, cx + shx), rng.uniform(cy - shy, cy + shy))
                for _ in range(num_envs)
            ]
        else:
            min_dist = 0.1
            place_xys = []
            for _ in range(num_envs):
                while True:
                    xy = task_spec.sample_table_xy(rng)
                    dist = np.sqrt((xy[0] - task_spec.goal_xy[0])**2 + (xy[1] - task_spec.goal_xy[1])**2)
                    if dist >= min_dist:
                        place_xys.append(xy)
                        break

        place_poses_np = np.array(
            [[xy[0], xy[1], place_z, 1.0, 0.0, 0.0, 0.0] for xy in place_xys], dtype=np.float32
        )
        goal_pose_np = np.array(
            [task_spec.goal_xy[0], task_spec.goal_xy[1], goal_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        # ---- init expert ----
        ee_pose = get_ee_pose_w(env)
        expert.reset(ee_pose, place_xys[0], place_z=place_z)
        for i, xy in enumerate(place_xys):
            expert.place_pose[i, 0] = xy[0]
            expert.place_pose[i, 1] = xy[1]

        # ---- pre-position to gripper-down rest ----
        prepos_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        prepos_action[:, :7] = expert.rest_pose[:, :7]
        prepos_action[:, 7] = 1.0
        for _ in range(80):
            obs_dict, _, _, _, _ = env.step(prepos_action)
        for _ in range(10):
            obs_dict, _, _, _, _ = env.step(prepos_action)

        # ---- markers ----
        if markers is None:
            start_markers, goal_markers, marker_z = create_target_markers(
                num_envs, device,
                red_marker_shape=getattr(task_spec, "red_marker_shape", "circle"),
                red_marker_size_xy=getattr(task_spec, "red_marker_size_xy", None),
            )
            markers = (start_markers, goal_markers, marker_z)
        else:
            start_markers, goal_markers, marker_z = markers

        marker_xys = place_xys
        if bool(getattr(task_spec, "fix_red_marker_pose", False)) and red_center is not None:
            marker_xys = [(float(red_center[0]), float(red_center[1])) for _ in range(num_envs)]

        update_target_markers(
            start_markers, goal_markers,
            start_xys=marker_xys,
            goal_xy=task_spec.goal_xy,
            marker_z=marker_z,
            env=env,
        )
        settle_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
        settle_action[:, :7] = get_ee_pose_w(env)[:, :7]
        settle_action[:, 7] = 1.0
        for _ in range(3):
            obs_dict, _, _, _, _ = env.step(settle_action)

        # ---- recording buffers ----
        obs_lists = [[] for _ in range(num_envs)]
        image_lists = [[] for _ in range(num_envs)]
        wrist_image_lists = [[] for _ in range(num_envs)]
        ee_pose_lists = [[] for _ in range(num_envs)]
        obj_pose_lists = [[] for _ in range(num_envs)]
        action_lists = [[] for _ in range(num_envs)]
        gripper_lists = [[] for _ in range(num_envs)]
        fsm_state_lists = [[] for _ in range(num_envs)]

        expert_completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        prev_state = expert.state.clone()
        log_lines = []  # detailed debug log

        # ---- log all waypoint target positions (env 0) ----
        _e = expert
        _obj = get_object_pose_w(env)[0]
        _obj_xy, _obj_z = _obj[:2].cpu().numpy(), _obj[2].item()
        _hz = _e.episode_hover_z[0].item()
        _gz = _e.grasp_z_offset
        _pz = _e.place_pose[0, 2].item()
        _pxy = _e.place_pose[0, :2].cpu().numpy()
        _rz = _e.release_z_offset

        def _v2(t): return t[0].cpu().numpy()
        def _v1(t): return t[0].item()

        waypoint_info = (
            f"=== WAYPOINT TARGETS (env 0) ===\n"
            f"obj_xy=[{_obj_xy[0]:.4f}, {_obj_xy[1]:.4f}], obj_z={_obj_z:.4f}\n"
            f"place_xy=[{_pxy[0]:.4f}, {_pxy[1]:.4f}], place_z={_pz:.4f}\n"
            f"hover_z={_hz:.4f}, grasp_z_offset={_gz:.4f}, release_z_offset={_rz:.4f}\n"
            f"\n--- PICK (descending) ---\n"
            f"APPROACH_OBJ:      xy=[{_obj_xy[0]+_v1(_e.approach_obj_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.approach_obj_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.approach_obj_extra_z):.4f}  (hover_z + {_v1(_e.approach_obj_extra_z):.4f})\n"
            f"GO_ABOVE_OBJ:      xy=[{_obj_xy[0]+_v1(_e.above_obj_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.above_obj_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.above_obj_extra_z):.4f}  (hover_z + {_v1(_e.above_obj_extra_z):.4f})\n"
            f"CORRECT_OBJ:       xy=[{_obj_xy[0]+_v1(_e.correct_obj_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.correct_obj_dxy[0,1:2]):.4f}], z={_obj_z+_gz+_v1(_e.correct_obj_extra_z):.4f}  (obj_z+gz + {_v1(_e.correct_obj_extra_z):.4f})\n"
            f"ALIGN_TO_OBJ:      xy=[{_obj_xy[0]+_v1(_e.align_obj_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.align_obj_dxy[0,1:2]):.4f}], z={_obj_z+_gz+_v1(_e.align_obj_extra_z):.4f}  (obj_z+gz + {_v1(_e.align_obj_extra_z):.4f})\n"
            f"GO_TO_OBJ:         xy=[{_obj_xy[0]+_v1(_e.grasp_dx):.4f}, {_obj_xy[1]:.4f}], z={_obj_z+_gz:.4f}  (X+{_v1(_e.grasp_dx):.4f}, Y=0, Z=0)\n"
            f"\n--- LIFT (ascending, symmetric to pick) ---\n"
            f"LIFT_ALIGN:        xy=[{_obj_xy[0]+_v1(_e.lift_align_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.lift_align_dxy[0,1:2]):.4f}], z={_obj_z+_gz+_v1(_e.lift_align_extra_z):.4f}  (obj_z+gz + {_v1(_e.lift_align_extra_z):.4f})\n"
            f"LIFT_CORRECT:      xy=[{_obj_xy[0]+_v1(_e.lift_correct_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.lift_correct_dxy[0,1:2]):.4f}], z={_obj_z+_gz+_v1(_e.lift_correct_extra_z):.4f}  (obj_z+gz + {_v1(_e.lift_correct_extra_z):.4f})\n"
            f"LIFT_ABOVE:        xy=[{_obj_xy[0]+_v1(_e.lift_above_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.lift_above_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.lift_above_extra_z):.4f}  (hover_z + {_v1(_e.lift_above_extra_z):.4f})\n"
            f"LIFT_APPROACH:     xy=[{_obj_xy[0]+_v1(_e.lift_approach_dxy[0,0:1]):.4f}, {_obj_xy[1]+_v1(_e.lift_approach_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.lift_approach_extra_z):.4f}  (hover_z + {_v1(_e.lift_approach_extra_z):.4f})\n"
            f"\n--- PLACE (descending) ---\n"
            f"APPROACH_PLACE:    xy=[{_pxy[0]+_v1(_e.approach_place_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.approach_place_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.approach_place_extra_z):.4f}  (hover_z + {_v1(_e.approach_place_extra_z):.4f})\n"
            f"GO_ABOVE_PLACE:    xy=[{_pxy[0]+_v1(_e.above_place_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.above_place_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.above_place_extra_z):.4f}  (hover_z + {_v1(_e.above_place_extra_z):.4f})\n"
            f"CORRECT_PLACE:     xy=[{_pxy[0]+_v1(_e.correct_place_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.correct_place_dxy[0,1:2]):.4f}], z={_pz+_v1(_e.correct_place_extra_z):.4f}  (place_z + {_v1(_e.correct_place_extra_z):.4f})\n"
            f"ALIGN_TO_PLACE:    xy=[{_pxy[0]+_v1(_e.align_place_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.align_place_dxy[0,1:2]):.4f}], z={_pz+_v1(_e.align_place_extra_z):.4f}  (place_z + {_v1(_e.align_place_extra_z):.4f})\n"
            f"GO_TO_PLACE:       xy=[{_pxy[0]+_v1(_e.place_dx):.4f}, {_pxy[1]:.4f}], z={_pz:.4f}  (X+{_v1(_e.place_dx):.4f}, Y=0, Z=0)\n"
            f"LOWER_TO_RELEASE:  xy=[{_pxy[0]+_v1(_e.place_dx):.4f}, {_pxy[1]:.4f}], z={_pz+_rz:.4f}  (place_z + {_rz:.4f})\n"
            f"\n--- DEPARTURE (ascending, symmetric to place) ---\n"
            f"DEPART_ALIGN:      xy=[{_pxy[0]+_v1(_e.depart_align_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.depart_align_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.depart_align_extra_z):.4f}  (hover_z + {_v1(_e.depart_align_extra_z):.4f})\n"
            f"DEPART_CORRECT:    xy=[{_pxy[0]+_v1(_e.depart_correct_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.depart_correct_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.depart_correct_extra_z):.4f}  (hover_z + {_v1(_e.depart_correct_extra_z):.4f})\n"
            f"DEPART_ABOVE:      xy=[{_pxy[0]+_v1(_e.depart_above_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.depart_above_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.depart_above_extra_z):.4f}  (hover_z + {_v1(_e.depart_above_extra_z):.4f})\n"
            f"DEPART_APPROACH:   xy=[{_pxy[0]+_v1(_e.depart_approach_dxy[0,0:1]):.4f}, {_pxy[1]+_v1(_e.depart_approach_dxy[0,1:2]):.4f}], z={_hz+_v1(_e.depart_approach_extra_z):.4f}  (hover_z + {_v1(_e.depart_approach_extra_z):.4f})\n"
            f"REST:              xyz=[{_e.rest_pose[0,0].item():.4f}, {_e.rest_pose[0,1].item():.4f}, {_e.rest_pose[0,2].item():.4f}]\n"
        )
        print(waypoint_info, flush=True)
        log_lines.append(waypoint_info)
        log_lines.append(f"\n=== STEP-BY-STEP LOG ===\n")
        log_lines.append(f"{'step':>5s}  {'state':<24s}  {'ee_x':>7s} {'ee_y':>7s} {'ee_z':>7s}  {'tgt_x':>7s} {'tgt_y':>7s} {'tgt_z':>7s}  {'gripper':>7s}\n")

        # ---- main loop ----
        for t in range(horizon):
            cur_state = expert.state.clone()
            state_name = ExpertState(cur_state[0].item()).name
            is_transition = (t == 0 or cur_state[0] != prev_state[0])
            # Log state transitions
            if is_transition:
                ee = get_ee_pose_w(env)[0]
                print(f"  t={t:4d}/{horizon}  state={state_name:<24s}  ee=[{ee[0].item():.4f},{ee[1].item():.4f},{ee[2].item():.4f}]", flush=True)
            prev_state = cur_state

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
            gripper_np = goal_action.cpu().numpy()[:, 7]
            fsm_states_np = expert.state.cpu().numpy()

            for i in range(num_envs):
                obs_lists[i].append(obs_np[i])
                image_lists[i].append(table_images_np[i])
                wrist_image_lists[i].append(wrist_images_np[i])
                ee_pose_lists[i].append(ee_pose_np[i])
                obj_pose_lists[i].append(obj_pose_np[i])
                gripper_lists[i].append(gripper_np[i])
                fsm_state_lists[i].append(fsm_states_np[i])

            action = expert.act(ee_pose, object_pose)

            # Log every step to file (env 0 only)
            _ee = ee_pose_np[0]
            _act = action[0].cpu().numpy()
            log_lines.append(
                f"{t:5d}  {state_name:<24s}  {_ee[0]:7.4f} {_ee[1]:7.4f} {_ee[2]:7.4f}  {_act[0]:7.4f} {_act[1]:7.4f} {_act[2]:7.4f}  {_act[7]:7.1f}\n"
            )

            obs_dict, _, _, _, _ = env.step(action)

            next_ee_pose = get_ee_pose_w(env)
            next_ee_pose_np = next_ee_pose.cpu().numpy()
            for i in range(num_envs):
                act = np.zeros(8, dtype=np.float32)
                act[:7] = next_ee_pose_np[i]
                act[7] = gripper_np[i]
                action_lists[i].append(act)

            just_done = expert.is_done() & ~expert_completed
            expert_completed = expert_completed | expert.is_done()

            if expert_completed.all():
                print(f"  All envs done at step {t+1}")
                break

        # ---- settle steps ----
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
                obs_lists[i].append(obs_np[i])
                image_lists[i].append(table_images_np[i])
                wrist_image_lists[i].append(wrist_images_np[i])
                ee_pose_lists[i].append(ee_pose_np[i])
                obj_pose_lists[i].append(obj_pose_np[i])
                gripper_lists[i].append(1.0)
                fsm_state_lists[i].append(fsm_states_np[i])

            rest_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
            rest_action[:, :7] = expert.rest_pose
            rest_action[:, 7] = 1.0
            obs_dict, _, _, _, _ = env.step(rest_action)

            next_ee_pose = get_ee_pose_w(env)
            next_ee_pose_np = next_ee_pose.cpu().numpy()
            for i in range(num_envs):
                act = np.zeros(8, dtype=np.float32)
                act[:7] = next_ee_pose_np[i]
                act[7] = 1.0
                action_lists[i].append(act)

        # ---- build episodes ----
        object_pose = get_object_pose_w(env)
        results = []
        for i in range(num_envs):
            place_xy = place_xys[i]
            obj_xy = object_pose[i, :2].cpu().numpy()
            dist = np.sqrt((obj_xy[0] - place_xy[0])**2 + (obj_xy[1] - place_xy[1])**2)
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
                "place_pose": place_poses_np[i],
                "goal_pose": goal_pose_np,
                "success": success_bool,
            }
            results.append((episode_dict, expert_completed[i].item()))

        return results, markers, log_lines

    # ==================================================================
    # main
    # ==================================================================
    try:
        set_seed(args.seed)
        rng = np.random.default_rng(args.seed)

        env = make_env_with_camera(
            task_id=args.task,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
            episode_length_s=100.0,
            disable_terminations=True,
        )
        device = env.unwrapped.device

        task_spec = PickPlaceTaskSpec(
            goal_xy=tuple(args.goal_xy),
            hover_z=0.25,
            grasp_z_offset=0.0,
            success_radius=0.03,
            settle_steps=10,
        )
        task_spec.taskB_target_mode = "red_region"
        task_spec.red_region_center_xy = tuple(args.red_region_center_xy)
        task_spec.red_region_size_xy = tuple(args.red_region_size_xy)
        task_spec.red_marker_shape = "rectangle"
        task_spec.red_marker_size_xy = tuple(args.red_region_size_xy)
        task_spec.fix_red_marker_pose = True

        expert = PickPlaceExpertRandWaypoint(
            num_envs=args.num_envs,
            device=device,
            hover_z=task_spec.hover_z,
            grasp_z_offset=task_spec.grasp_z_offset,
            release_z_offset=-0.02,
            position_threshold=0.015,
            wait_steps=10,
        )

        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        markers = None
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} episodes (PickPlaceExpertRandWaypoint)")
        print(f"  num_envs={args.num_envs}, horizon={args.horizon}, settle={args.settle_steps}")
        print(f"  red_region_center={args.red_region_center_xy}, size={args.red_region_size_xy}")
        print(f"{'='*60}\n")

        all_log_lines = []

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            results, markers, log_lines = rollout(
                env=env,
                expert=expert,
                task_spec=task_spec,
                rng=rng,
                horizon=args.horizon,
                settle_steps=args.settle_steps,
                markers=markers,
            )
            all_log_lines.extend(log_lines)
            batch_completed = 0
            batch_success = 0
            for episode_dict, completed_flag in results:
                batch_completed += int(completed_flag)
                episodes.append(episode_dict)
                if episode_dict["success"]:
                    batch_success += 1
                if len(episodes) >= args.num_episodes:
                    break

            elapsed = time.time() - start_time
            total_attempts = batch_count * args.num_envs
            rate = total_attempts / elapsed if elapsed > 0 else 0
            print(
                f"Batch {batch_count:3d} | Saved: {len(episodes)}/{args.num_episodes} | "
                f"This batch: {batch_completed}/{args.num_envs} completed, {batch_success} success | "
                f"Rate: {rate:.1f} ep/s"
            )

        elapsed = time.time() - start_time
        success_count = sum(1 for ep in episodes if ep["success"])
        print(f"\n{'='*60}")
        print(f"Collection finished in {elapsed:.1f}s")
        print(f"Saved: {len(episodes)}, Success: {success_count} ({100*success_count/len(episodes) if episodes else 0:.1f}%)")
        print(f"{'='*60}\n")

        episodes = episodes[:args.num_episodes]
        save_episodes_with_goal_actions(args.out, episodes)

        # Save debug log (only when --save_log is set)
        if args.save_log:
            log_path = args.out.replace(".npz", "_debug.log")
            with open(log_path, "w") as f:
                f.writelines(all_log_lines)
            print(f"Debug log saved to: {log_path}")

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
