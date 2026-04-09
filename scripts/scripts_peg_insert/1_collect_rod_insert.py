#!/usr/bin/env python3
"""Collect rod INSERTION demonstration data (Task A) for Exp39.

Uses the Manager-Based rod-insert environment (Isaac-RodInsert-Franka-IK-Abs-v0).
An FSM expert drives the rod into a block's vertical hole from above.

The gripper stays CLOSED throughout — no gripper control needed.

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
For each episode:
    - ee_pose:          (T, 7)    End-effector pose [x, y, z, qw, qx, qy, qz]
    - action:           (T, 8)    Action = next-frame ee_pose(7) + gripper(1)
    - images:           (T, H, W, 3)   Table camera RGB (uint8)
    - wrist_wrist_cam:  (T, H, W, 3)   Wrist camera RGB (uint8)
    - joint_pos:        (T, 9)    Robot joint positions (7 arm + 2 finger)
    - rod_pose:         (T, 7)    Rod pose [x, y, z, qw, qx, qy, qz]
    - block_pose:       (T, 7)    Block pose
    - phase:            (T,)      FSM phase ID
    - episode_length:   int
    - success:          bool
    - task_type:        str ("insert")
    - phase_names:      list[str]

=============================================================================
USAGE
=============================================================================
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 python scripts/scripts_peg_insert/1_collect_rod_insert.py \
    --headless --num_episodes 3 --horizon 400 \
    --out data/pick_place_isaac_lab_simulation/exp39/task_A_rod_insert_3.npz
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect rod insertion data (Task A).")
    parser.add_argument("--task", type=str, default="Isaac-RodInsert-Franka-IK-Abs-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--settle_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default="data/pick_place_isaac_lab_simulation/exp39/task_A_rod_insert_3.npz")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# =========================================================================
# Camera helpers  (same pattern as pick-place collection scripts)
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

    # Table camera — third-person view
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

    # Wrist camera — attached to panda_hand
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
    """Import FrankaRodInsertEnvCfg from the workspace's isaaclab_tasks.

    The bundled isaaclab_tasks inside the conda env doesn't contain our
    custom rod_insert_env_cfg.  We load it with importlib.util, injecting
    the parent ``ik_abs_env_cfg`` (which IS in the bundled copy) into the
    module namespace so relative imports work.
    """
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

    # Import the sibling ik_abs_env_cfg from the bundled copy (it's already usable)
    ik_abs = importlib.import_module(
        "isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg"
    )

    # Create a fake parent package so "from . import ik_abs_env_cfg" works
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
    from isaaclab.sim import SimulationCfg

    EnvCfg = _get_rod_insert_env_cfg()
    env_cfg = EnvCfg()
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = device
    env_cfg.sim.use_fabric = bool(use_fabric)

    add_camera_to_env_cfg(env_cfg, image_width, image_height)

    # Override gripper open command to barely-open (gap 12mm, rod is 8mm)
    env_cfg.actions.gripper_action.open_command_expr = {"panda_finger_.*": 0.006}

    # Disable terminations for data collection
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "object_dropping"):
            env_cfg.terminations.object_dropping = None

    env_cfg.episode_length_s = 120.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


# =========================================================================
# FSM Expert: Rod Insertion
# =========================================================================

class RodInsertExpert:
    """Scripted expert for rod insertion.

    The robot starts at a random HOME position (set during settle/init,
    not a FSM phase).  Recording begins with APPROACH.

    Phases:
        0  APPROACH  — Keep home XY, descend to random Z in APPROACH_Z_RANGE
        1  ALIGN     — Keep current Z, move XY to align with hole centre
        2  INSERT    — Push rod downward into hole at insert_speed
        3  DONE      — Hold position

    Action semantics (IK-Abs):
        action[0:3] = target EE position [x, y, z]
        action[3:7] = target EE orientation [qw, qx, qy, qz]
        action[7]   = gripper (1 = close)
    """

    PHASE_NAMES = ["APPROACH", "ALIGN", "INSERT", "DONE"]

    # Block and rod geometry constants
    BLOCK_TOP_Z = 0.040        # block top surface z
    HOLE_CENTER_XY = (0.5, 0.0)
    ROD_HALF_HEIGHT = 0.030    # rod height / 2 (60mm / 2)

    # EE orientation: pointing straight down
    # wxyz = (0, 1, 0, 0) → 180° rotation around X → panda_hand Z axis points down
    EE_QUAT_DOWN = (0.0, 1.0, 0.0, 0.0)

    # HOME phase randomization ranges
    HOME_Z_RANGE = (0.15, 0.18)        # Z height at home
    HOME_XY_OFFSET_RANGE = 0.05        # ±50mm random XY offset from hole centre

    # APPROACH Z range (random per env)
    APPROACH_Z_RANGE = (0.12, 0.14)

    def __init__(self, num_envs, device,
                 insert_target_z=0.05,  # EE z at full insertion
                 insert_speed=0.001,    # z decrement per step
                 success_depth=0.015,   # rod must be this deep to succeed (lenient)
                 done_hold_steps=10,    # hold DONE for this many steps then stop
                 ):
        import torch
        self.num_envs = num_envs
        self.device = device
        self.insert_target_z = insert_target_z
        self.insert_speed = insert_speed
        self.success_depth = success_depth
        self.done_hold_steps = done_hold_steps

        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Fixed gripper-down quaternion
        self.ee_quat = torch.tensor(self.EE_QUAT_DOWN, device=device).unsqueeze(0).expand(num_envs, -1)
        # Per-env random home position (set in set_home_pos, used for settle)
        self.home_pos = torch.zeros(num_envs, 3, device=device)
        # Per-env random approach Z (set in set_home_pos)
        self.approach_z = torch.zeros(num_envs, device=device)
        # Per-env hold Z for DONE phase (captured at INSERT→DONE transition)
        self.hold_z = torch.zeros(num_envs, device=device)

    def set_home_pos(self, hole_xy, env_ids=None):
        """Generate random home positions and approach Z for given envs.

        Args:
            hole_xy: (N, 2) hole centre XY for each env.
            env_ids: which envs to randomise (default: all).
        """
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
        # Random approach Z per env
        az_lo, az_hi = self.APPROACH_Z_RANGE
        self.approach_z[env_ids] = torch.empty(n, device=self.device).uniform_(az_lo, az_hi)

    def reset(self, env_ids=None):
        import torch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.step_count[env_ids] = 0
        self.phase_step_count[env_ids] = 0
        self.hold_z[env_ids] = 0.0

    def is_done(self):
        return (self.phase >= 3) & (self.phase_step_count >= self.done_hold_steps)

    def compute_action(self, ee_pose, rod_pose, block_pose):
        """Compute IK-Abs action.

        Args:
            ee_pose:    (N, 7) current EE pose [x,y,z,qw,qx,qy,qz]
            rod_pose:   (N, 7) current rod pose
            block_pose: (N, 7) current block pose

        Returns:
            action: (N, 8)  [target_ee_pos(3), target_ee_quat(4), gripper(1)]
        """
        import torch

        N = self.num_envs
        action = torch.zeros(N, 8, device=self.device)

        self.step_count += 1
        self.phase_step_count += 1

        # Block hole centre (from block pose XY)
        hole_x = block_pose[:, 0]
        hole_y = block_pose[:, 1]
        block_top_z = block_pose[:, 2] + 0.020  # block half-height

        phase_snap = self.phase.clone()
        next_phase = self.phase.clone()

        # Gripper always closed (negative = close in BinaryJointPositionAction)
        action[:, 7] = -1.0

        # Orientation: keep current captured orientation (pointing down)
        action[:, 3:7] = self.ee_quat

        # ---- PHASE 0: APPROACH (keep home XY, descend to approach_z) ----
        mask = phase_snap == 0
        if mask.any():
            # Keep home XY, only change Z
            action[mask, 0] = self.home_pos[mask, 0]
            action[mask, 1] = self.home_pos[mask, 1]
            action[mask, 2] = self.approach_z[mask]

            # Transition when close to target Z
            target_pos = torch.stack([
                self.home_pos[mask, 0], self.home_pos[mask, 1],
                self.approach_z[mask]], dim=-1)
            dist = torch.norm(ee_pose[mask, :3] - target_pos, dim=-1)
            arrived = mask.clone()
            arrived[mask] = dist < 0.005
            timeout = mask & (self.phase_step_count > 80)
            next_phase = torch.where(arrived | timeout,
                                     torch.full_like(next_phase, 1), next_phase)

        # ---- PHASE 1: ALIGN (keep current Z, move XY to hole centre) ----
        mask = phase_snap == 1
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            # Keep Z at approach_z (current height)
            action[mask, 2] = self.approach_z[mask]

            # Transition when XY close to hole centre
            xy_dist = torch.norm(
                ee_pose[mask, :2] - torch.stack([hole_x[mask], hole_y[mask]], dim=-1),
                dim=-1,
            )
            arrived = mask.clone()
            arrived[mask] = xy_dist < 0.003
            timeout = mask & (self.phase_step_count > 100)
            next_phase = torch.where(arrived | timeout,
                                     torch.full_like(next_phase, 2), next_phase)

        # ---- PHASE 2: INSERT (push down from approach_z to insert_target_z) ----
        mask = phase_snap == 2
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]

            # Gradually lower target z from approach_z
            target_z = (self.approach_z[mask]
                        - self.insert_speed * self.phase_step_count[mask].float())
            target_z = target_z.clamp(min=self.insert_target_z)
            action[mask, 2] = target_z

            # Check insertion depth: rod bottom z = rod_centre_z - rod_half_height
            rod_bottom_z = rod_pose[mask, 2] - self.ROD_HALF_HEIGHT
            depth = block_top_z[mask] - rod_bottom_z  # positive = inserted
            inserted = mask.clone()
            inserted[mask] = depth > self.success_depth
            timeout = mask & (self.phase_step_count > 250)
            next_phase = torch.where(inserted | timeout,
                                     torch.full_like(next_phase, 3), next_phase)

        # Capture hold Z when transitioning from INSERT to DONE
        newly_done = (phase_snap == 2) & (next_phase == 3)
        if newly_done.any():
            self.hold_z[newly_done] = action[newly_done, 2]

        # ---- PHASE 3: DONE (hold at last INSERT target Z) ----
        mask = phase_snap == 3
        if mask.any():
            action[mask, 0] = hole_x[mask]
            action[mask, 1] = hole_y[mask]
            action[mask, 2] = self.hold_z[mask]

        # Apply transitions
        changed = next_phase != self.phase
        if changed.any():
            self.phase_step_count[changed] = 0
        self.phase = next_phase

        return action


# =========================================================================
# Rollout + save
# =========================================================================

def rollout_rod_insert(env, expert, horizon, settle_steps=30):
    """Run parallel rollouts for rod insertion."""
    import numpy as np
    import torch

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    rl_env = env.unwrapped

    table_camera = rl_env.scene.sensors["table_cam"]
    wrist_camera = rl_env.scene.sensors["wrist_cam"]

    # Get references
    robot = rl_env.scene["robot"]
    rod = rl_env.scene["object"]
    block = rl_env.scene["block"]
    ee_frame = rl_env.scene["ee_frame"]

    obs_dict, _ = env.reset()
    expert.reset()

    # Recording buffers
    ee_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    wrist_image_lists = [[] for _ in range(num_envs)]
    joint_pos_lists = [[] for _ in range(num_envs)]
    rod_pose_lists = [[] for _ in range(num_envs)]
    block_pose_lists = [[] for _ in range(num_envs)]
    phase_lists = [[] for _ in range(num_envs)]

    # ---- Settle phase: move robot to random HOME position with gripper DOWN ----
    # The rod spawns at (0.5, 0, 0.5) out of the way.  We move the robot
    # to the random home position with the gripper straight down and barely
    # OPEN (0.006), then teleport the rod between the finger pads and close.
    #
    # ROD_GRIP_Z_OFFSET: offset from EE (fingertip) to rod centre in world Z.
    # Gripper points DOWN, so finger pads are BELOW EE in world frame.
    # Negative offset places rod below EE, between the finger pads.
    # This value MUST match 3_collect_rod_extract.py to ensure consistent grip.
    ROD_GRIP_Z_OFFSET = -0.02
    DOWN_QUAT = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).expand(num_envs, -1)

    # Generate random home position for each env
    block_pos = block.data.root_pos_w - rl_env.scene.env_origins
    hole_xy = block_pos[:, :2]  # hole centre = block XY
    expert.set_home_pos(hole_xy)
    settle_target_pos = expert.home_pos.clone()
    print(f"[{_ts()}] Home pos: {settle_target_pos[0].cpu().numpy()}", flush=True)

    print(f"[{_ts()}] Pre-positioning robot ({settle_steps} steps, fingers barely open)...", flush=True)
    for s in range(settle_steps):
        # Gripper +1.0 = OPEN → drives fingers to 0.006 (overridden open_command)
        settle_action = torch.cat([settle_target_pos, DOWN_QUAT,
                                   torch.full((num_envs, 1), 1.0, device=device)], dim=-1)
        obs_dict, _, _, _, _ = env.step(settle_action)
        if (s + 1) % 50 == 0:
            cur_ee_pos = ee_frame.data.target_pos_w[:, 0, :] - rl_env.scene.env_origins
            pos_err = torch.norm(cur_ee_pos - settle_target_pos, dim=-1).max().item()
            print(f"[{_ts()}]   settle step {s+1}/{settle_steps}, pos_err={pos_err:.4f}", flush=True)

    # Teleport rod to grip position (ROD_GRIP_Z_OFFSET below EE)
    cur_ee_pos_w = ee_frame.data.target_pos_w[:, 0, :].clone()  # world frame
    rod_target_pos = cur_ee_pos_w.clone()
    rod_target_pos[:, 2] += ROD_GRIP_Z_OFFSET  # offset down from EE (negative)
    rod_teleport_pose = torch.cat([rod_target_pos,
                                   torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1)],
                                  dim=-1)
    rod.write_root_pose_to_sim(rod_teleport_pose)
    rod.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))
    print(f"[{_ts()}] Rod teleported between finger pads", flush=True)

    # Close fingers to grip the rod (gripper -1.0 = CLOSE)
    grip_steps = 80
    print(f"[{_ts()}] Closing gripper ({grip_steps} steps)...", flush=True)
    for s in range(grip_steps):
        settle_action = torch.cat([settle_target_pos, DOWN_QUAT,
                                   torch.full((num_envs, 1), -1.0, device=device)], dim=-1)
        obs_dict, _, _, _, _ = env.step(settle_action)
        # Re-teleport rod for first few steps to prevent it being pushed out
        # before fingers have converged enough to hold it by friction
        if s < 20:
            rod.write_root_pose_to_sim(rod_teleport_pose)
            rod.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

    # Fully tighten gripper: set open_command to 0.0 so both OPEN and CLOSE
    # target fully-closed fingers, preventing any looseness during recording.
    rl_env.action_manager._terms["gripper_action"]._open_command.fill_(0.0)
    print(f"[{_ts()}] Gripper fully tightened (open_command → 0.0)", flush=True)

    print(f"[{_ts()}] Recording started (horizon={horizon})...", flush=True)

    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"[{_ts()}] Step {t+1}/{horizon}", flush=True)

        # Current state
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

        # Camera images
        table_rgb = table_camera.data.output["rgb"]
        wrist_rgb = wrist_camera.data.output["rgb"]
        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        if wrist_rgb.shape[-1] > 3:
            wrist_rgb = wrist_rgb[..., :3]

        # Expert action
        action = expert.compute_action(ee_pose, rod_full, block_full)

        # Record
        for i in range(num_envs):
            ee_pose_lists[i].append(ee_pose[i].cpu().numpy())
            action_lists[i].append(action[i].cpu().numpy())
            image_lists[i].append(table_rgb[i].cpu().numpy().astype(np.uint8))
            wrist_image_lists[i].append(wrist_rgb[i].cpu().numpy().astype(np.uint8))
            joint_pos_lists[i].append(joint_positions[i].cpu().numpy())
            rod_pose_lists[i].append(rod_full[i].cpu().numpy())
            block_pose_lists[i].append(block_full[i].cpu().numpy())
            phase_lists[i].append(expert.phase[i].item())

        # Step environment
        obs_dict, _, _, _, _ = env.step(action)

        # Early termination: all envs in DONE for enough steps
        if expert.is_done().all():
            print(f"[{_ts()}] All envs done at step {t+1}, stopping early.", flush=True)
            break

    # Build episode dicts
    episodes = []
    block_top_z = 0.040  # nominal
    for i in range(num_envs):
        rod_poses_arr = np.array(rod_pose_lists[i])
        rod_bottom_z_final = rod_poses_arr[-1, 2] - expert.ROD_HALF_HEIGHT
        depth = block_top_z - rod_bottom_z_final
        success = bool(depth > expert.success_depth)

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
            "task_type": "insert",
            "phase_names": RodInsertExpert.PHASE_NAMES,
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

        expert = RodInsertExpert(
            num_envs=args.num_envs,
            device=device,
        )

        episodes = []
        batch_count = 0
        max_batches = (args.num_episodes // args.num_envs + 1) * 3
        start_time = time.time()

        print(f"\n[{_ts()}] {'='*60}")
        print(f"[{_ts()}] Collecting {args.num_episodes} rod insertion episodes")
        print(f"[{_ts()}]   num_envs={args.num_envs}, horizon={args.horizon}")
        print(f"[{_ts()}] {'='*60}\n", flush=True)

        while len(episodes) < args.num_episodes and batch_count < max_batches:
            batch_count += 1
            print(f"[{_ts()}] Batch {batch_count}...", flush=True)

            batch_episodes = rollout_rod_insert(
                env, expert, args.horizon, args.settle_steps,
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
