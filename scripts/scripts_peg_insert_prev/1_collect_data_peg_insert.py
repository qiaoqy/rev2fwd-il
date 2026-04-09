#!/usr/bin/env python3
"""Step 1: Collect peg insertion demonstration data with force sensing.

Uses the FORGE PegInsert environment (Isaac-Forge-PegInsert-Direct-v0).
An FSM expert aligns and inserts a peg into a hole using force feedback.

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
For each episode, the following arrays are saved:
    - obs:              (T, obs_dim)   Policy observation sequence
    - state:            (T, state_dim) Full state observation (privileged)
    - images:           (T, H, W, 3)   RGB images from table camera (uint8)
    - wrist_wrist_cam:  (T, H, W, 3)   Wrist camera RGB (uint8)
    - ee_pose:          (T, 7)    End-effector pose [x, y, z, qw, qx, qy, qz]
    - peg_pose:         (T, 7)    Peg (held asset) pose
    - hole_pose:        (T, 7)    Hole (fixed asset) pose
    - action:           (T, 7)    Action [pos(3), rot(3), gripper(1)]
    - ft_force:         (T, 3)    Force sensor (Fx, Fy, Fz)
    - ft_force_raw:     (T, 6)    Raw force/torque (Fx, Fy, Fz, Tx, Ty, Tz)
    - joint_pos:        (T, 7)    Robot joint positions
    - phase:            (T,)      Expert state machine phase ID
    - episode_length:   int       Total timesteps in episode
    - success:          bool      Whether task was successful

=============================================================================
USAGE
=============================================================================
# Basic: collect 3 episodes (debug)
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python scripts/scripts_peg_insert/1_collect_data_peg_insert.py \
    --headless --num_episodes 3 --horizon 600 \
    --out data/pick_place_isaac_lab_simulation/exp39/task_B_peg_insert_3.npz

# Production: 100 episodes
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python scripts/scripts_peg_insert/1_collect_data_peg_insert.py \
    --headless --num_episodes 100 --num_envs 4 --horizon 600 \
    --out data/pick_place_isaac_lab_simulation/exp39/task_B_peg_insert_100.npz
=============================================================================
"""

from __future__ import annotations

import argparse
import time
import sys

sys.stdout.reconfigure(line_buffering=True)
print("[DEBUG] Script started", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect peg insertion data with force sensing.")

    parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/pick_place_isaac_lab_simulation/exp39/task_B_peg_insert_3.npz")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# =========================================================================
# Camera helpers (reused from nut threading script pattern)
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

    camera_eye = (0.7, 0.2, 0.2)
    camera_lookat = (0.6, 0.0, 0.1)
    camera_quat = compute_camera_quat_from_lookat(camera_eye, camera_lookat)

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
        offset=CameraCfg.OffsetCfg(pos=camera_eye, rot=camera_quat, convention="ros"),
    )
    env_cfg.scene.env_spacing = 5.0
    env_cfg.episode_length_s = 120.0


def add_wrist_camera_post_creation(env, num_envs, image_width, image_height):
    import omni.usd
    from pxr import UsdGeom, Gf
    import omni.replicator.core as rep

    stage = omni.usd.get_context().get_stage()
    camera_configs = [
        ("wrist_cam", Gf.Vec3d(0.05, 0.0, 0.0), Gf.Quatf(0.0, 1.0, 0.0, 0.0)),
    ]
    all_camera_data = {}
    for cam_name, cam_pos, cam_quat in camera_configs:
        render_products, rgb_annotators = [], []
        for env_idx in range(num_envs):
            panda_hand_path = f"/World/envs/env_{env_idx}/Robot/panda_hand"
            prim = stage.GetPrimAtPath(panda_hand_path)
            if not prim.IsValid():
                render_products.append(None); rgb_annotators.append(None); continue
            wrist_cam_path = f"{panda_hand_path}/{cam_name}"
            if not stage.GetPrimAtPath(wrist_cam_path).IsValid():
                cam_prim = UsdGeom.Camera.Define(stage, wrist_cam_path)
                cam_prim.GetFocalLengthAttr().Set(18.0)
                cam_prim.GetHorizontalApertureAttr().Set(20.955)
                cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 2.0))
                xform = UsdGeom.Xformable(cam_prim.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(cam_pos)
                xform.AddOrientOp().Set(cam_quat)
            try:
                rp = rep.create.render_product(wrist_cam_path, (image_width, image_height))
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach([rp])
                render_products.append(rp); rgb_annotators.append(ann)
            except Exception as e:
                print(f"[ERROR] wrist cam env_{env_idx}: {e}", flush=True)
                render_products.append(None); rgb_annotators.append(None)
        all_camera_data[cam_name] = (render_products, rgb_annotators)
    return all_camera_data


def make_forge_env_with_camera(task_id, num_envs, device, use_fabric, image_width, image_height):
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=device, num_envs=int(num_envs), use_fabric=bool(use_fabric))
    add_camera_to_env_cfg(env_cfg, image_width, image_height)
    env = gym.make(task_id, cfg=env_cfg)

    wrist_camera_data = add_wrist_camera_post_creation(env, num_envs, image_width, image_height)
    env.unwrapped._wrist_camera_data = wrist_camera_data
    first_cam = list(wrist_camera_data.keys())[0] if wrist_camera_data else None
    if first_cam:
        env.unwrapped._wrist_render_products = wrist_camera_data[first_cam][0]
        env.unwrapped._wrist_rgb_annotators = wrist_camera_data[first_cam][1]
    return env


# =========================================================================
# FSM Expert for Peg Insertion
# =========================================================================

class PegInsertExpert:
    """Force-feedback scripted expert for peg insertion.

    Phases:
        0  APPROACH   — Move EE directly above the hole, centered.
        1  ALIGN      — Fine XY alignment + slow descent until near hole top.
        2  INSERT     — Push peg downward with force feedback.
        3  DONE       — Hold position.

    Action space (FORGE 7-D):
        action[0:3]  position target relative to hole frame (scaled by pos_action_bounds)
        action[3:5]  roll/pitch (forced to 0 by FORGE)
        action[5]    yaw [-1,1] -> [-180°, +90°]
        action[6]    gripper: -1 = closed, +1 = open
    """

    PHASE_NAMES = ["APPROACH", "ALIGN", "INSERT", "DONE"]

    def __init__(self, num_envs: int, device: str,
                 approach_z: float = 0.04,       # 4 cm above hole
                 insert_speed: float = 0.3,      # z action during insertion
                 max_force: float = 30.0,        # back off if exceeded
                 insert_force_target: float = 8.0,  # target downward force
                 success_z_progress: float = 0.020,  # 20 mm insertion depth → done
                 ):
        import torch
        self.num_envs = num_envs
        self.device = device
        self.approach_z = approach_z
        self.insert_speed = insert_speed
        self.max_force = max_force
        self.insert_force_target = insert_force_target
        self.success_z_progress = success_z_progress

        # State
        self.phase = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.initial_peg_z = torch.zeros(num_envs, device=device)
        self.z_progress = torch.zeros(num_envs, device=device)

        # Timing
        self.approach_timeout = 80
        self.align_timeout = 120
        self.insert_timeout = 300

    def reset(self, env_ids=None):
        import torch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.step_count[env_ids] = 0
        self.phase_step_count[env_ids] = 0
        self.initial_peg_z[env_ids] = 0.0
        self.z_progress[env_ids] = 0.0

    def is_done(self):
        return self.phase == 3

    def get_phase_names(self):
        return self.PHASE_NAMES

    def compute_action(
        self,
        fingertip_pos,   # (N, 3)
        fingertip_quat,  # (N, 4)
        fixed_pos,       # (N, 3)  — hole position (env frame)
        force_sensor,    # (N, 3)
        torque_sensor,   # (N, 3)
        peg_pos,         # (N, 3)  — peg position (env frame)
        dt: float = 1.0 / 90.0,
    ):
        """Compute expert action for peg insertion.

        FORGE action semantics:
            action[0:3] * pos_action_bounds = offset from fixed_pos_obs_frame
            So action = (desired_EE - fixed_pos_obs_frame) / pos_action_bounds
            With bounds = [0.05, 0.05, 0.05], action=1.0 means +5cm offset.
        """
        import torch

        action = torch.zeros(self.num_envs, 7, device=self.device)
        self.step_count += 1
        self.phase_step_count += 1

        # Force info
        fz = force_sensor[:, 2]
        force_mag = torch.norm(force_sensor, dim=1)

        # Relative position: EE above hole
        rel_pos = fingertip_pos - fixed_pos  # (N,3) in env frame

        # Peg Z tracking
        peg_z = peg_pos[:, 2]

        # Snapshot phase for single-transition-per-step
        phase_snap = self.phase.clone()
        next_phase = self.phase.clone()

        # Default: gripper closed, no rotation
        action[:, 6] = -1.0   # closed
        action[:, 3:6] = 0.0  # no rotation (FORGE zeros roll/pitch anyway)
        # Keep yaw at a neutral value (center of range)
        # [-1,1] -> [-180°,+90°], so 0 -> -45°. Use -1/3 for ~ -90° (centered)
        action[:, 5] = -1.0 / 3.0

        # =================================================================
        # PHASE 0: APPROACH — go above hole center
        # =================================================================
        mask = phase_snap == 0
        if mask.any():
            # Target: XY centered on hole, Z = approach_z above hole
            # action = (target - fixed_pos) / bounds
            # target_xy = fixed_pos_xy → action_xy = 0
            # target_z = fixed_pos_z + approach_z → action_z = approach_z / 0.05
            action[mask, 0] = 0.0  # center X
            action[mask, 1] = 0.0  # center Y
            action[mask, 2] = self.approach_z / 0.05  # above hole (normalized)

            # Record initial peg Z
            first = mask & (self.phase_step_count == 1)
            if first.any():
                self.initial_peg_z[first] = peg_z[first]

            # Transition when roughly above hole and close to target Z
            target_z = fixed_pos[:, 2] + self.approach_z
            z_close = mask & ((fingertip_pos[:, 2] - target_z).abs() < 0.005)
            xy_close = mask & (torch.norm(rel_pos[:, :2], dim=1) < 0.008)
            ready = z_close & xy_close
            next_phase = torch.where(ready, torch.full_like(next_phase, 1), next_phase)

            # Timeout
            timeout = mask & (self.phase_step_count > self.approach_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 1), next_phase)

        # =================================================================
        # PHASE 1: ALIGN — fine XY centering + slow descent
        # =================================================================
        mask = phase_snap == 1
        if mask.any():
            # Slowly lower toward hole top (z=0 relative to hole tip)
            # Descend gradually: from approach_z down to ~0.005 (5mm above)
            t_frac = (self.phase_step_count[mask].float() / self.align_timeout).clamp(0, 1)
            target_z_norm = (self.approach_z * (1.0 - t_frac) + 0.005 * t_frac) / 0.05

            action[mask, 0] = 0.0  # center X
            action[mask, 1] = 0.0  # center Y
            action[mask, 2] = target_z_norm

            first = mask & (self.phase_step_count == 1)
            if first.any():
                self.initial_peg_z[first] = peg_z[first]

            # Transition: force detected (contact with hole rim) or z close enough
            contact = mask & (force_mag > 2.0)
            z_low = mask & (rel_pos[:, 2] < 0.008)
            next_phase = torch.where(contact | z_low, torch.full_like(next_phase, 2), next_phase)

            timeout = mask & (self.phase_step_count > self.align_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 2), next_phase)

        # =================================================================
        # PHASE 2: INSERT — push down with force feedback
        # =================================================================
        mask = phase_snap == 2
        if mask.any():
            first = mask & (self.phase_step_count == 1)
            if first.any():
                self.initial_peg_z[first] = peg_z[first]

            # Adaptive Z speed: slow down when force is high
            force_ratio = (force_mag[mask] / self.insert_force_target).clamp(0, 3)
            # Base speed scaled down when force is large
            adaptive_speed = self.insert_speed * (1.0 - 0.5 * (force_ratio - 1.0).clamp(0, 1))

            action[mask, 0] = 0.0  # stay centered
            action[mask, 1] = 0.0
            action[mask, 2] = -adaptive_speed  # push down

            # Back off if force exceeds max
            too_much = mask & (force_mag > self.max_force)
            action[too_much, 2] = 0.1  # slight upward

            # Track insertion progress
            z_drop = self.initial_peg_z - peg_z
            self.z_progress = torch.where(mask, z_drop, self.z_progress)

            # Done when sufficient insertion depth
            inserted = mask & (self.z_progress > self.success_z_progress)
            next_phase = torch.where(inserted, torch.full_like(next_phase, 3), next_phase)

            # Timeout
            timeout = mask & (self.phase_step_count > self.insert_timeout)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 3), next_phase)

        # =================================================================
        # PHASE 3: DONE — hold position
        # =================================================================
        mask = phase_snap == 3
        if mask.any():
            action[mask, 0] = 0.0
            action[mask, 1] = 0.0
            action[mask, 2] = -0.1  # gentle hold
            action[mask, 6] = -1.0

        # Apply phase transitions
        changed = next_phase != self.phase
        if changed.any():
            self.phase_step_count[changed] = 0
        self.phase = next_phase

        return action


# =========================================================================
# Rollout
# =========================================================================

def rollout_peg_insert(env, expert: PegInsertExpert, horizon: int):
    """Run parallel rollouts and collect data."""
    import numpy as np
    import torch

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    physics_dt = env.unwrapped.physics_dt
    forge_env = env.unwrapped

    print(f"[DEBUG] rollout: device={device}, num_envs={num_envs}, dt={physics_dt}")

    # Camera refs
    table_camera = forge_env.scene.sensors["table_cam"]
    wrist_camera_data = getattr(forge_env, '_wrist_camera_data', None)
    wrist_cam_names = list(wrist_camera_data.keys()) if wrist_camera_data else []

    # Reset
    obs_dict, _ = env.reset()
    expert.reset()

    # Override EMA for deterministic scripted collection
    import torch as _torch
    forge_env.ema_factor = _torch.ones_like(forge_env.ema_factor)
    print(f"[DEBUG] rollout: Overrode ema_factor to 1.0")

    # Recording buffers
    obs_lists = [[] for _ in range(num_envs)]
    state_lists = [[] for _ in range(num_envs)]
    image_lists = [[] for _ in range(num_envs)]
    wrist_image_lists_dict = {c: [[] for _ in range(num_envs)] for c in wrist_cam_names}
    ee_pose_lists = [[] for _ in range(num_envs)]
    peg_pose_lists = [[] for _ in range(num_envs)]
    hole_pose_lists = [[] for _ in range(num_envs)]
    action_lists = [[] for _ in range(num_envs)]
    ft_force_lists = [[] for _ in range(num_envs)]
    ft_force_raw_lists = [[] for _ in range(num_envs)]
    joint_pos_lists = [[] for _ in range(num_envs)]
    phase_lists = [[] for _ in range(num_envs)]

    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"[DEBUG] rollout: Step {t+1}/{horizon}")

        # Observations
        if isinstance(obs_dict, dict):
            policy_obs = obs_dict.get("policy", next(iter(obs_dict.values())))
            critic_state = obs_dict.get("critic", policy_obs)
        else:
            policy_obs = critic_state = obs_dict

        # Camera images
        table_rgb = table_camera.data.output["rgb"]
        wrist_images_by_cam = {}
        if wrist_camera_data:
            for cn, (_, annotators) in wrist_camera_data.items():
                imgs = []
                for ann in annotators:
                    if ann is not None:
                        d = ann.get_data()
                        if d is not None:
                            if not isinstance(d, torch.Tensor):
                                d = torch.from_numpy(d).to(device)
                            imgs.append(d)
                        else:
                            imgs.append(torch.zeros(table_rgb.shape[1:], dtype=table_rgb.dtype, device=device))
                    else:
                        imgs.append(torch.zeros(table_rgb.shape[1:], dtype=table_rgb.dtype, device=device))
                wrist_images_by_cam[cn] = torch.stack(imgs, dim=0)

        if table_rgb.shape[-1] > 3:
            table_rgb = table_rgb[..., :3]
        for cn in wrist_images_by_cam:
            if wrist_images_by_cam[cn].shape[-1] > 3:
                wrist_images_by_cam[cn] = wrist_images_by_cam[cn][..., :3]

        if t == 0:
            print(f"[DEBUG] table_rgb shape: {table_rgb.shape}")
            for cn, img in wrist_images_by_cam.items():
                print(f"[DEBUG] {cn} shape: {img.shape}")

        # Poses
        fingertip_pos = forge_env.fingertip_midpoint_pos.clone()
        fingertip_quat = forge_env.fingertip_midpoint_quat.clone()

        peg_pos = forge_env._held_asset.data.root_pos_w - forge_env.scene.env_origins
        peg_quat = forge_env._held_asset.data.root_quat_w
        peg_pose = torch.cat([peg_pos, peg_quat], dim=-1)

        hole_pos = forge_env._fixed_asset.data.root_pos_w - forge_env.scene.env_origins
        hole_quat = forge_env._fixed_asset.data.root_quat_w
        hole_pose = torch.cat([hole_pos, hole_quat], dim=-1)

        # Force/torque
        if hasattr(forge_env, 'force_sensor_smooth'):
            ft_force_raw = forge_env.force_sensor_smooth.clone()
            ft_force = ft_force_raw[:, :3]
        else:
            ft_force_raw = torch.zeros(num_envs, 6, device=device)
            ft_force = torch.zeros(num_envs, 3, device=device)

        joint_pos = forge_env._robot.data.joint_pos[:, :7]
        ee_pose = torch.cat([fingertip_pos, fingertip_quat], dim=-1)

        # Expert action
        ft_torque = ft_force_raw[:, 3:6]
        action = expert.compute_action(
            fingertip_pos=fingertip_pos,
            fingertip_quat=fingertip_quat,
            fixed_pos=hole_pos,
            force_sensor=ft_force,
            torque_sensor=ft_torque,
            peg_pos=peg_pos,
            dt=physics_dt,
        )

        # Record
        pol_np = policy_obs.cpu().numpy()
        crit_np = critic_state.cpu().numpy()
        timg_np = table_rgb.cpu().numpy().astype(np.uint8)
        wimg_np = {cn: wrist_images_by_cam[cn].cpu().numpy().astype(np.uint8) for cn in wrist_images_by_cam}
        phase_np = expert.phase.cpu().numpy().copy()
        ee_np = ee_pose.cpu().numpy()
        peg_np = peg_pose.cpu().numpy()
        hole_np = hole_pose.cpu().numpy()
        act_np = action.cpu().numpy()
        ftf_np = ft_force.cpu().numpy()
        ftr_np = ft_force_raw.cpu().numpy()
        jp_np = joint_pos.cpu().numpy()

        for i in range(num_envs):
            obs_lists[i].append(pol_np[i])
            state_lists[i].append(crit_np[i])
            image_lists[i].append(timg_np[i])
            for cn in wrist_cam_names:
                wrist_image_lists_dict[cn][i].append(wimg_np[cn][i])
            ee_pose_lists[i].append(ee_np[i])
            peg_pose_lists[i].append(peg_np[i])
            hole_pose_lists[i].append(hole_np[i])
            action_lists[i].append(act_np[i])
            ft_force_lists[i].append(ftf_np[i])
            ft_force_raw_lists[i].append(ftr_np[i])
            joint_pos_lists[i].append(jp_np[i])
            phase_lists[i].append(phase_np[i])

        # Step
        obs_dict, reward, terminated, truncated, info = env.step(action)

        # Debug
        if (t + 1) % 10 == 0:
            for i in range(min(num_envs, 2)):
                ph = int(expert.phase[i])
                pn = PegInsertExpert.PHASE_NAMES[ph] if ph < len(PegInsertExpert.PHASE_NAMES) else "?"
                fz_val = ftf_np[i, 2]
                zp = expert.z_progress[i].item() * 1000
                print(f"[DEBUG] t={t+1} env{i}: phase={pn}, Fz={fz_val:.2f}N, z_prog={zp:.2f}mm")

    # Build episode dicts
    results = []
    for i in range(num_envs):
        z_prog = expert.z_progress[i].item()
        ep_success = z_prog > expert.success_z_progress
        ep = {
            "obs": np.array(obs_lists[i], dtype=np.float32),
            "state": np.array(state_lists[i], dtype=np.float32),
            "images": np.array(image_lists[i], dtype=np.uint8),
            **{f"wrist_{cn}": np.array(wrist_image_lists_dict[cn][i], dtype=np.uint8) for cn in wrist_cam_names},
            "ee_pose": np.array(ee_pose_lists[i], dtype=np.float32),
            "peg_pose": np.array(peg_pose_lists[i], dtype=np.float32),
            "hole_pose": np.array(hole_pose_lists[i], dtype=np.float32),
            "action": np.array(action_lists[i], dtype=np.float32),
            "ft_force": np.array(ft_force_lists[i], dtype=np.float32),
            "ft_force_raw": np.array(ft_force_raw_lists[i], dtype=np.float32),
            "joint_pos": np.array(joint_pos_lists[i], dtype=np.float32),
            "phase": np.array(phase_lists[i], dtype=np.int32),
            "phase_names": PegInsertExpert.PHASE_NAMES,
            "episode_length": len(obs_lists[i]),
            "success": ep_success,
            "z_progress": float(z_prog),
            "success_threshold": forge_env.cfg_task.success_threshold,
            "wrist_cam_names": wrist_cam_names,
        }
        results.append(ep)
        print(f"[DEBUG] env {i}: success={ep_success}, z_progress={z_prog*1000:.1f}mm, final_phase={expert.phase[i].item()}")
    return results


def save_episodes(path, episodes):
    from pathlib import Path
    import numpy as np

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, episodes=episodes)
    print(f"Saved {len(episodes)} episodes to {p}")
    if episodes:
        e = episodes[0]
        print(f"  obs={e['obs'].shape}, images={e['images'].shape}, "
              f"action={e['action'].shape}, ft_force={e['ft_force'].shape}, "
              f"length={e['episode_length']}, success={e['success']}")


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
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print(f"[DEBUG] Creating FORGE PegInsert env: {args.task}")
        env = make_forge_env_with_camera(
            task_id=args.task,
            num_envs=args.num_envs,
            device=args.device,
            use_fabric=not bool(args.disable_fabric),
            image_width=args.image_width,
            image_height=args.image_height,
        )
        print("[DEBUG] Environment created")
        device = env.unwrapped.device
        forge_env = env.unwrapped
        print(f"[DEBUG] Task: {forge_env.cfg_task.name}")
        print(f"[DEBUG] Action space: {env.action_space}")
        print(f"[DEBUG] Observation space: {env.observation_space}")

        expert = PegInsertExpert(num_envs=args.num_envs, device=device)

        episodes = []
        start_time = time.time()
        batch_count = 0

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} peg insertion episodes")
        print(f"  num_envs={args.num_envs}, horizon={args.horizon}")
        print(f"  image_size={args.image_width}x{args.image_height}")
        print(f"{'='*60}\n")

        while len(episodes) < args.num_episodes:
            batch_count += 1
            results = rollout_peg_insert(env=env, expert=expert, horizon=args.horizon)

            # For debug phase: save ALL episodes (success or not)
            for ep in results:
                episodes.append(ep)
                if len(episodes) >= args.num_episodes:
                    break

            n_success = sum(1 for e in results if e["success"])
            elapsed = time.time() - start_time
            print(f"Batch {batch_count}: {n_success}/{len(results)} success | "
                  f"Collected {len(episodes)}/{args.num_episodes} | {elapsed:.0f}s")

        episodes = episodes[:args.num_episodes]
        save_episodes(args.out, episodes)
        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
