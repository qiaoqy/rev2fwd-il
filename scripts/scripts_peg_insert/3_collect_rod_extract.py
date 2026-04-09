#!/usr/bin/env python3
"""Collect rod EXTRACTION demonstration data (Task B) for Exp39.

Uses the Manager-Based rod-insert environment (Isaac-RodInsert-Franka-IK-Abs-v0).

Strategy: After env.reset() the rod is above the block.  We first run a
*setup phase* (not recorded) to insert the rod using the insert expert.
Then we record the extraction expert pulling the rod out + wobbling.

When time-reversed, the extract trajectory becomes an insert demonstration.

Gripper stays CLOSED throughout.

=============================================================================
USAGE
=============================================================================
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 python scripts/scripts_peg_insert/3_collect_rod_extract.py \
    --headless --num_episodes 3 --horizon 400 \
    --out data/pick_place_isaac_lab_simulation/exp39/task_B_rod_extract_3.npz
=============================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect rod extraction data (Task B).")
    parser.add_argument("--task", type=str, default="Isaac-RodInsert-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--settle_steps", type=int, default=200)
    parser.add_argument("--setup_horizon", type=int, default=350,
                        help="Max steps for setup insertion (not recorded).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="data/pick_place_isaac_lab_simulation/exp39/task_B_rod_extract_3.npz")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# =========================================================================
# Camera helpers  (same as 1_collect_rod_insert.py)
# =========================================================================

def compute_camera_quat_from_lookat(eye, target, up=(0, 0, 1)):
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    eye, target, up = (np.array(v, dtype=np.float64) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = R.from_matrix(np.column_stack([right, down, forward]))
    q = rot.as_quat()  # xyzw
    return (q[3], q[0], q[1], q[2])  # wxyz


def add_camera_to_env_cfg(env_cfg, image_width, image_height):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab.sensors.camera import TiledCameraCfg

    table_eye = (0.7, 0.25, 0.25)
    table_lookat = (0.5, 0.0, 0.04)
    table_quat = compute_camera_quat_from_lookat(table_eye, table_lookat)

    env_cfg.scene.table_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=3.0 / 90.0,
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.5),
        ),
        offset=CameraCfg.OffsetCfg(pos=table_eye, rot=table_quat, convention="ros"),
    )

    wrist_quat = (-0.70614, 0.03701, 0.03701, -0.70614)
    env_cfg.scene.wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=3.0 / 90.0,
        height=image_height,
        width=image_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15), rot=wrist_quat, convention="ros"),
    )

    env_cfg.scene.env_spacing = 5.0
    env_cfg.episode_length_s = 120.0

    # Disable debug visualization (coordinate frame arrows on gripper)
    if hasattr(env_cfg, 'commands') and hasattr(env_cfg.commands, 'object_pose'):
        env_cfg.commands.object_pose.debug_vis = False
    if hasattr(env_cfg.scene, 'ee_frame'):
        env_cfg.scene.ee_frame.debug_vis = False


def _get_rod_insert_env_cfg():
    """Import FrankaRodInsertEnvCfg from the workspace's isaaclab_tasks."""
    import importlib
    import importlib.util
    import os
    import sys
    import types

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    cfg_path = os.path.join(
        workspace_root,
        "isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/rod_insert_env_cfg.py",
    )

    ik_abs = importlib.import_module(
        "isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg"
    )

    parent_name = "isaaclab_tasks.manager_based.manipulation.lift.config.franka"
    parent_mod = sys.modules.get(parent_name)
    if parent_mod is None:
        parent_mod = types.ModuleType(parent_name)
        parent_mod.__path__ = [os.path.dirname(cfg_path)]
        parent_mod.__package__ = parent_name
        sys.modules[parent_name] = parent_mod
    parent_mod.ik_abs_env_cfg = ik_abs

    mod_name = f"{parent_name}.rod_insert_env_cfg"
    spec = importlib.util.spec_from_file_location(
        mod_name, cfg_path,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = parent_name
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.FrankaRodInsertEnvCfg


def make_env_with_camera(task_id, num_envs, device, use_fabric,
                         image_width, image_height):
    from isaaclab.envs import ManagerBasedRLEnv

    EnvCfg = _get_rod_insert_env_cfg()
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = device
    env_cfg.sim.use_fabric = bool(use_fabric)

    add_camera_to_env_cfg(env_cfg, image_width, image_height)

    # Override gripper open command to barely-open (gap 12mm, rod is 8mm)
    env_cfg.actions.gripper_action.open_command_expr = {"panda_finger_.*": 0.006}

    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "object_dropping"):
            env_cfg.terminations.object_dropping = None

    env_cfg.episode_length_s = 120.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


# =========================================================================
# FSM Expert: Rod Insertion  (same as 1_collect_rod_insert.py — used for setup)
# =========================================================================

class RodInsertExpert:
    """Scripted expert for rod insertion (used during setup phase)."""

    PHASE_NAMES = ["APPROACH", "ALIGN", "INSERT", "DONE"]
    BLOCK_TOP_Z = 0.040
    ROD_HALF_HEIGHT = 0.030
    EE_QUAT_DOWN = (0.0, 1.0, 0.0, 0.0)

    def __init__(self, num_envs, device,
                 approach_z=0.10, align_z=0.06,
                 insert_target_z=0.02, insert_speed=0.0005,
                 success_depth=0.025):
        import torch
        self.num_envs = num_envs
        self.device = device
        self.approach_z = approach_z
        self.align_z = align_z
        self.insert_target_z = insert_target_z
        self.insert_speed = insert_speed
        self.success_depth = success_depth

        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ee_quat = torch.tensor(self.EE_QUAT_DOWN, device=device).unsqueeze(0).expand(num_envs, -1)

    def reset(self, env_ids=None):
        import torch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.step_count[env_ids] = 0
        self.phase_step_count[env_ids] = 0

    def is_done(self):
        return self.phase >= 3

    def compute_action(self, ee_pose, rod_pose, block_pose):
        import torch
        N = self.num_envs
        action = torch.zeros(N, 8, device=self.device)
        self.step_count += 1
        self.phase_step_count += 1

        hole_x = block_pose[:, 0]
        hole_y = block_pose[:, 1]
        block_top_z = block_pose[:, 2] + 0.020

        phase_snap = self.phase.clone()
        next_phase = self.phase.clone()
        action[:, 7] = -1.0  # gripper closed
        action[:, 3:7] = self.ee_quat

        mask = phase_snap == 0
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            action[mask, 2] = self.approach_z
            dist = torch.norm(
                ee_pose[mask, :3] - torch.stack([hole_x[mask], hole_y[mask],
                    torch.full_like(hole_x[mask], self.approach_z)], dim=-1), dim=-1)
            arrived = mask.clone()
            arrived[mask] = dist < 0.005
            timeout = mask & (self.phase_step_count > 80)
            next_phase = torch.where(arrived | timeout,
                                     torch.full_like(next_phase, 1), next_phase)

        mask = phase_snap == 1
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            action[mask, 2] = self.align_z
            dist = torch.norm(
                ee_pose[mask, :3] - torch.stack([hole_x[mask], hole_y[mask],
                    torch.full_like(hole_x[mask], self.align_z)], dim=-1), dim=-1)
            arrived = mask.clone()
            arrived[mask] = dist < 0.003
            timeout = mask & (self.phase_step_count > 100)
            next_phase = torch.where(arrived | timeout,
                                     torch.full_like(next_phase, 2), next_phase)

        mask = phase_snap == 2
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            target_z = (self.align_z
                        - self.insert_speed * self.phase_step_count[mask].float())
            target_z = target_z.clamp(min=self.insert_target_z)
            action[mask, 2] = target_z
            rod_bottom_z = rod_pose[mask, 2] - self.ROD_HALF_HEIGHT
            depth = block_top_z[mask] - rod_bottom_z
            inserted = mask.clone()
            inserted[mask] = depth > self.success_depth
            timeout = mask & (self.phase_step_count > 250)
            next_phase = torch.where(inserted | timeout,
                                     torch.full_like(next_phase, 3), next_phase)

        mask = phase_snap == 3
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            action[mask, 2] = self.insert_target_z

        changed = next_phase != self.phase
        if changed.any():
            self.phase_step_count[changed] = 0
        self.phase = next_phase
        return action


# =========================================================================
# FSM Expert: Rod Extraction
# =========================================================================

class RodExtractExpert:
    """Scripted expert for rod extraction — exact time-reverse of RodInsertExpert.

    Insert trajectory:
        HOME → APPROACH (keep home XY, Z↓ to approach_z)
             → ALIGN    (XY to hole, keep approach_z)
             → INSERT   (Z↓ from approach_z into hole)
             → DONE     (hold)

    Extract trajectory (symmetric reverse):
        START (in hole) → EXTRACT  (keep hole XY, Z↑ to approach_z)
                        → DEALIGN (XY from hole to home, keep approach_z)
                        → ASCEND  (keep home XY, Z↑ to home_z)
                        → DONE    (hold at home)

    When time-reversed, extract becomes insert:
        DONE(home) → ASCEND↓ → DEALIGN(XY→hole) → EXTRACT↓ = insert
    """

    PHASE_NAMES = ["EXTRACT", "DEALIGN", "ASCEND", "DONE"]

    BLOCK_TOP_Z = 0.040
    ROD_HALF_HEIGHT = 0.030
    EE_QUAT_DOWN = (0.0, 1.0, 0.0, 0.0)

    HOME_Z_RANGE = (0.15, 0.18)
    HOME_XY_OFFSET_RANGE = 0.05
    APPROACH_Z_RANGE = (0.12, 0.14)

    def __init__(self, num_envs, device,
                 extract_speed=0.001,     # same as insert_speed for symmetry
                 done_hold_steps=10,      # hold DONE for this many steps then stop
                 ):
        import torch
        self.num_envs = num_envs
        self.device = device
        self.extract_speed = extract_speed
        self.done_hold_steps = done_hold_steps

        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ee_quat = torch.tensor(self.EE_QUAT_DOWN, device=device).unsqueeze(0).expand(num_envs, -1)

        # Per-env random home position and approach Z
        self.home_pos = torch.zeros(num_envs, 3, device=device)
        self.approach_z = torch.zeros(num_envs, device=device)
        # Start Z captured at first step of EXTRACT
        self.start_z = torch.zeros(num_envs, device=device)

    def set_home_pos(self, hole_xy, env_ids=None):
        """Generate random home positions and approach Z for given envs."""
        import torch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        r = self.HOME_XY_OFFSET_RANGE
        dx = torch.empty(n, device=self.device).uniform_(-r, r)
        dy = torch.empty(n, device=self.device).uniform_(-r, r)
        z_lo, z_hi = self.HOME_Z_RANGE
        dz = torch.empty(n, device=self.device).uniform_(z_lo, z_hi)
        self.home_pos[env_ids, 0] = hole_xy[env_ids, 0] + dx
        self.home_pos[env_ids, 1] = hole_xy[env_ids, 1] + dy
        self.home_pos[env_ids, 2] = dz
        az_lo, az_hi = self.APPROACH_Z_RANGE
        self.approach_z[env_ids] = torch.empty(n, device=self.device).uniform_(az_lo, az_hi)

    def reset(self, env_ids=None):
        import torch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.step_count[env_ids] = 0
        self.phase_step_count[env_ids] = 0
        self.start_z[env_ids] = 0.0

    def is_done(self):
        return (self.phase >= 3) & (self.phase_step_count >= self.done_hold_steps)

    def compute_action(self, ee_pose, rod_pose, block_pose):
        import torch

        N = self.num_envs
        action = torch.zeros(N, 8, device=self.device)

        self.step_count += 1
        self.phase_step_count += 1

        hole_x = block_pose[:, 0]
        hole_y = block_pose[:, 1]

        phase_snap = self.phase.clone()
        next_phase = self.phase.clone()

        action[:, 7] = -1.0  # gripper closed
        action[:, 3:7] = self.ee_quat

        # ---- PHASE 0: EXTRACT (keep hole XY, Z↑ from start_z to approach_z) ----
        mask = phase_snap == 0
        if mask.any():
            # Capture start Z on first step
            first = mask & (self.phase_step_count == 1)
            if first.any():
                self.start_z[first] = ee_pose[first, 2]

            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            target_z = (self.start_z[mask]
                        + self.extract_speed * self.phase_step_count[mask].float())
            target_z = target_z.clamp(max=self.approach_z[mask])
            action[mask, 2] = target_z

            # Transition: Z reached approach_z
            at_height = mask.clone()
            at_height[mask] = target_z >= self.approach_z[mask] - 0.005
            timeout = mask & (self.phase_step_count > 250)
            next_phase = torch.where(at_height | timeout,
                                     torch.full_like(next_phase, 1), next_phase)

        # ---- PHASE 1: DEALIGN (keep Z at approach_z, XY from hole to home) ----
        mask = phase_snap == 1
        if mask.any():
            action[mask, 0] = self.home_pos[mask, 0]
            action[mask, 1] = self.home_pos[mask, 1]
            action[mask, 2] = self.approach_z[mask]

            xy_dist = torch.norm(
                ee_pose[mask, :2] - self.home_pos[mask, :2], dim=-1)
            arrived = mask.clone()
            arrived[mask] = xy_dist < 0.003
            timeout = mask & (self.phase_step_count > 100)
            next_phase = torch.where(arrived | timeout,
                                     torch.full_like(next_phase, 2), next_phase)

        # ---- PHASE 2: ASCEND (keep home XY, Z↑ from approach_z to home_z) ----
        mask = phase_snap == 2
        if mask.any():
            action[mask, 0] = self.home_pos[mask, 0]
            action[mask, 1] = self.home_pos[mask, 1]
            action[mask, 2] = self.home_pos[mask, 2]

            dist = torch.norm(ee_pose[mask, :3] - self.home_pos[mask], dim=-1)
            arrived = mask.clone()
            arrived[mask] = dist < 0.005
            timeout = mask & (self.phase_step_count > 80)
            next_phase = torch.where(arrived | timeout,
                                     torch.full_like(next_phase, 3), next_phase)

        # ---- PHASE 3: DONE (hold at home) ----
        mask = phase_snap == 3
        if mask.any():
            action[mask, 0] = self.home_pos[mask, 0]
            action[mask, 1] = self.home_pos[mask, 1]
            action[mask, 2] = self.home_pos[mask, 2]

        changed = next_phase != self.phase
        if changed.any():
            self.phase_step_count[changed] = 0
        self.phase = next_phase

        return action


# =========================================================================
# Setup: insert rod first (not recorded)
# =========================================================================

def setup_insert_rod(env, settle_steps=200, insert_horizon=350):
    """Run the insert expert to get the rod into the hole before recording extraction.

    Uses the same settle/teleport approach as 1_collect_rod_insert.py:
    1. Move robot to approach position with fingers barely open
    2. Teleport rod between finger pads
    3. Close fingers to grip
    4. Run insert expert to push rod into hole
    """
    import torch

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    rl_env = env.unwrapped

    robot = rl_env.scene["robot"]
    rod = rl_env.scene["object"]
    block = rl_env.scene["block"]
    ee_frame = rl_env.scene["ee_frame"]

    insert_expert = RodInsertExpert(num_envs=num_envs, device=device)
    insert_expert.reset()

    DOWN_QUAT = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).expand(num_envs, -1)
    settle_target_pos = torch.tensor([[0.5, 0.0, 0.10]], device=device).expand(num_envs, -1)

    # Step 1: Pre-position robot with fingers barely open
    print(f"[{_ts()}] Setup: pre-positioning robot ({settle_steps} steps)...", flush=True)
    for s in range(settle_steps):
        action = torch.cat([settle_target_pos, DOWN_QUAT,
                            torch.full((num_envs, 1), 1.0, device=device)], dim=-1)
        env.step(action)
        if (s + 1) % 50 == 0:
            cur_ee_pos = ee_frame.data.target_pos_w[:, 0, :] - rl_env.scene.env_origins
            pos_err = torch.norm(cur_ee_pos - settle_target_pos, dim=-1).max().item()
            print(f"[{_ts()}]   setup settle step {s+1}/{settle_steps}, pos_err={pos_err:.4f}", flush=True)

    # Step 2: Teleport rod between finger pads
    # ROD_GRIP_Z_OFFSET: offset from EE (fingertip) to rod centre in world Z.
    # Gripper points DOWN, so finger pads are BELOW EE in world frame.
    # Negative offset places rod below EE, between the finger pads.
    # This value MUST match 1_collect_rod_insert.py.
    ROD_GRIP_Z_OFFSET = -0.02
    cur_ee_pos_w = ee_frame.data.target_pos_w[:, 0, :].clone()
    rod_target_pos = cur_ee_pos_w.clone()
    rod_target_pos[:, 2] += ROD_GRIP_Z_OFFSET  # offset down from EE (negative)
    rod_teleport_pose = torch.cat([rod_target_pos,
                                   torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1)],
                                  dim=-1)
    rod.write_root_pose_to_sim(rod_teleport_pose)
    rod.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))
    print(f"[{_ts()}] Setup: rod teleported between finger pads", flush=True)

    # Step 3: Close fingers to grip
    grip_steps = 80
    for s in range(grip_steps):
        action = torch.cat([settle_target_pos, DOWN_QUAT,
                            torch.full((num_envs, 1), -1.0, device=device)], dim=-1)
        env.step(action)
        if s < 20:
            rod.write_root_pose_to_sim(rod_teleport_pose)
            rod.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))
    print(f"[{_ts()}] Setup: gripper closed", flush=True)

    # Fully tighten gripper: set open_command to 0.0 so both OPEN and CLOSE
    # target fully-closed fingers, preventing any looseness.
    rl_env.action_manager._terms["gripper_action"]._open_command.fill_(0.0)
    print(f"[{_ts()}] Gripper fully tightened (open_command → 0.0)", flush=True)

    # Step 4: Run insert expert
    print(f"[{_ts()}] Setup: running insert expert ({insert_horizon} steps)...", flush=True)
    for t in range(insert_horizon):
        ee_pos = ee_frame.data.target_pos_w[:, 0, :] - rl_env.scene.env_origins
        ee_quat = ee_frame.data.target_quat_w[:, 0, :]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)

        rod_pos = rod.data.root_pos_w - rl_env.scene.env_origins
        rod_quat = rod.data.root_quat_w
        rod_full = torch.cat([rod_pos, rod_quat], dim=-1)

        block_pos = block.data.root_pos_w - rl_env.scene.env_origins
        block_quat = block.data.root_quat_w
        block_full = torch.cat([block_pos, block_quat], dim=-1)

        action = insert_expert.compute_action(ee_pose, rod_full, block_full)
        env.step(action)

        if t > 0 and (t + 1) % 100 == 0:
            rod_bottom_z = rod_pos[:, 2] - 0.030
            depth = 0.040 - rod_bottom_z
            print(f"[{_ts()}] Setup insert step {t+1}: "
                  f"depth={depth.mean().item()*1000:.1f}mm", flush=True)

        if insert_expert.is_done().all():
            rod_bottom_z = rod_pos[:, 2] - 0.030
            depth = 0.040 - rod_bottom_z
            print(f"[{_ts()}] Setup insertion done at step {t+1}: "
                  f"depth={depth.mean().item()*1000:.1f}mm", flush=True)
            return True

    rod_pos = rod.data.root_pos_w - rl_env.scene.env_origins
    rod_bottom_z = rod_pos[:, 2] - 0.030
    depth = 0.040 - rod_bottom_z
    print(f"[{_ts()}] Setup insertion finished (timeout): "
          f"depth={depth.mean().item()*1000:.1f}mm", flush=True)
    return bool((depth > 0.020).all())


# =========================================================================
# Rollout
# =========================================================================

def rollout_rod_extract(env, expert, horizon, settle_steps=30, setup_horizon=350):
    """Run extract rollout: setup insert → record extraction."""
    import numpy as np
    import torch

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    rl_env = env.unwrapped

    table_camera = rl_env.scene.sensors["table_cam"]
    wrist_camera = rl_env.scene.sensors["wrist_cam"]
    robot = rl_env.scene["robot"]
    rod = rl_env.scene["object"]
    block = rl_env.scene["block"]
    ee_frame = rl_env.scene["ee_frame"]

    # Reset env
    obs_dict, _ = env.reset()

    # Setup: insert the rod first (not recorded)
    print(f"[{_ts()}] Running setup insertion...", flush=True)
    inserted = setup_insert_rod(env, settle_steps, setup_horizon)
    if not inserted:
        print(f"[{_ts()}] WARNING: Setup insertion may not be complete!", flush=True)

    # Reset extract expert and generate random home position
    expert.reset()
    block_pos_local = block.data.root_pos_w - rl_env.scene.env_origins
    hole_xy = block_pos_local[:, :2]
    expert.set_home_pos(hole_xy)
    print(f"[{_ts()}] Extract home: {expert.home_pos[0].cpu().numpy()}, "
          f"approach_z: {expert.approach_z[0].item():.3f}", flush=True)

    # Recording buffers
    ee_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    wrist_image_lists = [[] for _ in range(num_envs)]
    joint_pos_lists = [[] for _ in range(num_envs)]
    rod_pose_lists = [[] for _ in range(num_envs)]
    block_pose_lists = [[] for _ in range(num_envs)]
    phase_lists = [[] for _ in range(num_envs)]

    print(f"[{_ts()}] Recording extraction (horizon={horizon})...", flush=True)

    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"[{_ts()}] Step {t+1}/{horizon}", flush=True)

        ee_pos = ee_frame.data.target_pos_w[:, 0, :] - rl_env.scene.env_origins
        ee_quat = ee_frame.data.target_quat_w[:, 0, :]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)

        rod_pos = rod.data.root_pos_w - rl_env.scene.env_origins
        rod_quat = rod.data.root_quat_w
        rod_full = torch.cat([rod_pos, rod_quat], dim=-1)

        block_pos = block.data.root_pos_w - rl_env.scene.env_origins
        block_quat = block.data.root_quat_w
        block_full = torch.cat([block_pos, block_quat], dim=-1)

        joint_positions = robot.data.joint_pos

        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        action = expert.compute_action(ee_pose, rod_full, block_full)

        for i in range(num_envs):
            ee_pose_lists[i].append(ee_pose[i].cpu().numpy())
            action_lists[i].append(action[i].cpu().numpy())
            image_lists[i].append(table_rgb[i].cpu().numpy().astype(np.uint8))
            wrist_image_lists[i].append(wrist_rgb[i].cpu().numpy().astype(np.uint8))
            joint_pos_lists[i].append(joint_positions[i].cpu().numpy())
            rod_pose_lists[i].append(rod_full[i].cpu().numpy())
            block_pose_lists[i].append(block_full[i].cpu().numpy())
            phase_lists[i].append(expert.phase[i].item())

        obs_dict, _, _, _, _ = env.step(action)

        # Early termination: all envs in DONE for enough steps
        if expert.is_done().all():
            print(f"[{_ts()}] All envs done at step {t+1}, stopping early.", flush=True)
            break

    # Build episodes
    episodes = []
    for i in range(num_envs):
        rod_poses_arr = np.array(rod_pose_lists[i])
        rod_bottom_z_final = rod_poses_arr[-1, 2] - 0.030
        success = bool(rod_bottom_z_final > 0.040 + 0.015)  # rod well above block

        episodes.append({
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),
            "images": np.array(image_lists[i], dtype=np.uint8),
            "wrist_images": np.array(wrist_image_lists[i], dtype=np.uint8),
            "joint_pos": np.array(joint_pos_lists[i], dtype=np.float32),
            "gripper": np.array(action_lists[i], dtype=np.float32)[:, 7],
            "phase": np.array(phase_lists[i], dtype=np.int32),
            "episode_length": len(ee_pose_lists[i]),
            "success": success,
            "task_type": "extract",
            "phase_names": RodExtractExpert.PHASE_NAMES,
        })

    return episodes


def save_episodes(path, episodes):
    import numpy as np
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, episodes=episodes)
    n_success = sum(1 for ep in episodes if ep["success"])
    print(f"[{_ts()}] Saved {len(episodes)} episodes to {path} "
          f"({n_success}/{len(episodes)} success)", flush=True)


# =========================================================================
# Main
# =========================================================================

def main():
    args = _parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import numpy as np
    import torch

    try:
        print(f"[{_ts()}] Creating environment...", flush=True)
        env = make_env_with_camera(
            task_id=args.task,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
        )
        device = env.unwrapped.device
        print(f"[{_ts()}] Environment created. device={device}", flush=True)

        expert = RodExtractExpert(num_envs=args.num_envs, device=device)

        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        start_time = time.time()

        print(f"\n[{_ts()}] {'='*60}")
        print(f"[{_ts()}] Collecting {args.num_episodes} rod extraction episodes")
        print(f"[{_ts()}]   num_envs={args.num_envs}, horizon={args.horizon}")
        print(f"[{_ts()}]   setup_horizon={args.setup_horizon}")
        print(f"[{_ts()}] {'='*60}\n", flush=True)

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            print(f"[{_ts()}] Batch {batch_count}...", flush=True)

            batch_episodes = rollout_rod_extract(
                env, expert, args.horizon, args.settle_steps, args.setup_horizon,
            )
            episodes.extend(batch_episodes)

            n_success = sum(1 for ep in batch_episodes if ep["success"])
            print(f"[{_ts()}] Batch {batch_count}: {n_success}/{len(batch_episodes)} success, "
                  f"total {len(episodes)}/{args.num_episodes}", flush=True)

            if len(episodes) >= args.num_episodes:
                break

        episodes = episodes[:args.num_episodes]
        elapsed = time.time() - start_time
        n_success = sum(1 for ep in episodes if ep["success"])
        print(f"\n[{_ts()}] Collection done in {elapsed:.1f}s: "
              f"{n_success}/{len(episodes)} success", flush=True)

        save_episodes(args.out, episodes)
        env.close()

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
