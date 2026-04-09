#!/usr/bin/env python3
"""Step 3: Collect peg EXTRACTION demonstration data (Task B).

Uses the FORGE PegInsert environment (Isaac-Forge-PegInsert-Direct-v0).
After reset, a setup phase uses GT (ground-truth) peg-hole XY offsets to drive
the peg into the hole.  Then the extraction expert pulls the peg out and
adds small XY wobble to simulate the alignment phase of the forward task (A).

When this trajectory is time-reversed it becomes an insertion demonstration:
    reversed WOBBLE → alignment / search
    reversed EXTRACT → smooth insertion

=============================================================================
OUTPUT DATA FORMAT (NPZ file)
=============================================================================
Same format as 1_collect_data_peg_insert.py.

=============================================================================
USAGE
=============================================================================
# Debug: collect 3 episodes
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python scripts/scripts_peg_insert/3_collect_data_peg_extract.py \
    --headless --num_episodes 3 --horizon 400 \
    --out data/pick_place_isaac_lab_simulation/exp39/task_B_peg_extract_3.npz

# Production: 100 episodes
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 python scripts/scripts_peg_insert/3_collect_data_peg_extract.py \
    --headless --num_episodes 100 --num_envs 4 --horizon 400 \
    --out data/pick_place_isaac_lab_simulation/exp39/task_B_peg_extract_100.npz
=============================================================================
"""

from __future__ import annotations

import argparse
import time
import sys

sys.stdout.reconfigure(line_buffering=True)
print("[DEBUG] Script started", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect peg extraction data (Task B).")

    parser.add_argument("--task", type=str, default="Isaac-Forge-PegInsert-Direct-v0")
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str,
                        default="data/pick_place_isaac_lab_simulation/exp39/task_B_peg_extract_3.npz")
    parser.add_argument("--disable_fabric", type=int, default=0, choices=[0, 1])
    parser.add_argument("--setup_max_steps", type=int, default=300,
                        help="Max steps for GT-guided setup insertion (not recorded).")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


# =========================================================================
# Camera helpers (same as insertion script)
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
# Setup: insert peg via env.step() actions (NOT recorded)
# =========================================================================

def setup_insert_peg(env, max_steps: int = 300, target_depth: float = 0.015):
    """Insert peg into hole using FORGE action space.

    Uses the same approach as PegInsertExpert but more aggressively:
    center on hole (action XY=0) and push down (action Z=-0.5).
    Since success rate is ~33% per attempt, this retries with env.reset()
    until insertion succeeds.

    This is a *setup* phase — not recorded as demonstration data.

    Args:
        env: FORGE gymnasium env (wrapped).
        max_steps: Maximum action steps per attempt.
        target_depth: Required insertion depth (m). 0.015 = 15mm.

    Returns:
        True if insertion succeeded.
    """
    import torch

    forge_env = env.unwrapped
    device = forge_env.device
    num_envs = forge_env.num_envs

    hole_height = forge_env.cfg_task.fixed_asset_cfg.height  # 0.025

    max_retries = 20
    for attempt in range(max_retries):
        if attempt > 0:
            obs_dict, _ = env.reset()
            forge_env.ema_factor = torch.ones_like(forge_env.ema_factor)

        hole_pos = (forge_env._fixed_asset.data.root_pos_w - forge_env.scene.env_origins).clone()
        hole_tip_z = hole_pos[:, 2] + hole_height

        # Phase 1: Approach — hover above hole center
        for t in range(min(60, max_steps)):
            action = torch.zeros(num_envs, 7, device=device)
            action[:, 0] = 0.0   # center X
            action[:, 1] = 0.0   # center Y
            action[:, 2] = 0.6   # ~3cm above hole tip
            action[:, 5] = -1.0 / 3.0  # neutral yaw
            action[:, 6] = -1.0  # gripper closed
            obs_dict, _, _, _, _ = env.step(action)

        # Phase 2: Align — slowly descend to just above hole
        for t in range(min(80, max_steps)):
            frac = t / 80.0
            z_action = 0.6 * (1.0 - frac) + 0.05 * frac  # 3cm → ~2.5mm
            action = torch.zeros(num_envs, 7, device=device)
            action[:, 0] = 0.0
            action[:, 1] = 0.0
            action[:, 2] = z_action
            action[:, 5] = -1.0 / 3.0
            action[:, 6] = -1.0
            obs_dict, _, _, _, _ = env.step(action)

        # Phase 3: Insert — push down
        for t in range(min(max_steps - 140, 200)):
            peg_pos = forge_env._held_asset.data.root_pos_w - forge_env.scene.env_origins
            z_inserted = hole_tip_z - peg_pos[:, 2]

            if (z_inserted > target_depth).all():
                print(f"[SETUP] Insertion success at attempt {attempt+1}, step {140+t}: "
                      f"z_inserted={z_inserted.min().item()*1000:.1f}mm", flush=True)
                return True

            action = torch.zeros(num_envs, 7, device=device)
            action[:, 0] = 0.0
            action[:, 1] = 0.0
            action[:, 2] = -0.4  # push down
            action[:, 5] = -1.0 / 3.0
            action[:, 6] = -1.0
            obs_dict, _, _, _, _ = env.step(action)

        peg_pos = forge_env._held_asset.data.root_pos_w - forge_env.scene.env_origins
        z_inserted = hole_tip_z - peg_pos[:, 2]
        print(f"[SETUP] Attempt {attempt+1}/{max_retries} failed: "
              f"z_inserted={z_inserted.min().item()*1000:.1f}mm", flush=True)

    print(f"[SETUP] WARNING: All {max_retries} attempts failed!", flush=True)
    return False


# =========================================================================
# FSM Expert for Peg Extraction (Task B)
# =========================================================================

class PegExtractExpert:
    """Scripted expert for peg extraction.

    Phases:
        0  EXTRACT  — Pull peg upward out of hole.
        1  WOBBLE   — Small random XY movements above hole (simulates alignment).
        2  DONE     — Hold final position.

    When time-reversed, the trajectory becomes:
        DONE → WOBBLE → EXTRACT
        i.e.  approach → align/search → insert
    """

    PHASE_NAMES = ["EXTRACT", "WOBBLE", "DONE"]

    def __init__(self, num_envs: int, device: str,
                 extract_speed: float = 0.15,     # z action per step (normalized) during extract
                 extract_target_z: float = 0.04,  # final height above hole (4cm)
                 wobble_amplitude: float = 0.15,  # max XY wobble (normalized action)
                 wobble_steps: int = 120,          # steps of wobble
                 done_steps: int = 50,             # steps of hold at end
                 ):
        import torch

        self.num_envs = num_envs
        self.device = device

        self.extract_speed = extract_speed
        self.extract_target_z = extract_target_z
        self.wobble_amplitude = wobble_amplitude
        self.wobble_steps = wobble_steps
        self.done_steps = done_steps

        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.initial_peg_z = torch.zeros(num_envs, device=device)
        self.current_z_action = torch.zeros(num_envs, device=device)

        # Pre-generate wobble pattern (smooth sinusoidal)
        self._wobble_pattern = self._generate_wobble_pattern(wobble_steps)

    def _generate_wobble_pattern(self, n_steps: int):
        """Generate smooth wobble XY offsets as a Lissajous-like pattern."""
        import torch
        import math

        t = torch.linspace(0, 1, n_steps, device=self.device)
        # Use different frequencies for X and Y to create a figure-8-like pattern
        # that covers a range of positions above the hole
        wx = self.wobble_amplitude * torch.sin(2 * math.pi * 3.0 * t)  # 3 cycles
        wy = self.wobble_amplitude * torch.sin(2 * math.pi * 2.0 * t + math.pi / 4)  # 2 cycles, phase shift
        # Taper: start with larger amplitude, gradually reduce (reversal = start small, grow)
        taper = 1.0 - 0.5 * t  # 1.0 → 0.5
        wx = wx * taper
        wy = wy * taper
        return torch.stack([wx, wy], dim=-1)  # (n_steps, 2)

    def reset(self, env_ids=None):
        import torch
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = 0
        self.step_count[env_ids] = 0
        self.phase_step_count[env_ids] = 0
        self.initial_peg_z[env_ids] = 0.0
        self.current_z_action[env_ids] = 0.0

    def is_done(self):
        return self.phase == 2

    def get_phase_names(self):
        return self.PHASE_NAMES

    def compute_action(
        self,
        fingertip_pos,    # (N, 3) env frame
        fingertip_quat,   # (N, 4)
        fixed_pos,        # (N, 3) hole position
        force_sensor,     # (N, 3)
        peg_pos,          # (N, 3) peg position
    ):
        """Compute extraction action.

        FORGE action semantics:
            target_EE = fixed_pos_action_frame + action[0:3] * bounds
            action = (desired_EE - fixed_pos_action_frame) / bounds
        But since we don't know init_fixed_pos_obs_noise here, we use the
        observation-relative formulation:
            action[0:2] = 0  → targets noisy hole centre (good enough for XY)
            action[2] = z_target / 0.05  → targets specific Z offset from hole tip
        """
        import torch

        action = torch.zeros(self.num_envs, 7, device=self.device)
        self.step_count += 1
        self.phase_step_count += 1

        # Phase snapshot (single transition per step)
        phase_snap = self.phase.clone()
        next_phase = self.phase.clone()

        # Default: gripper closed, neutral yaw
        action[:, 6] = -1.0
        action[:, 5] = -1.0 / 3.0

        # Track peg z_progress (how far extracted: positive = peg went up)
        peg_z = peg_pos[:, 2]
        hole_height = 0.025  # known from config

        # =================================================================
        # PHASE 0: EXTRACT — pull peg upward
        # =================================================================
        mask = phase_snap == 0
        if mask.any():
            first = mask & (self.phase_step_count == 1)
            if first.any():
                self.initial_peg_z[first] = peg_z[first]
                # Start z_action from current (inserted) position
                # Approximate: peg is ~20mm inserted, fingertip is ~12mm above hole origin
                # = ~(12mm - 25mm) = -13mm below hole tip
                # action_z = -13mm / 50mm ≈ -0.26
                self.current_z_action[first] = -0.3  # start below hole tip

            # Gradually increase z_action (move upward)
            self.current_z_action[mask] += self.extract_speed / 50.0  # smooth ramp
            # Clamp to target height
            target_z_action = self.extract_target_z / 0.05  # e.g. 0.04/0.05 = 0.8
            self.current_z_action[mask] = self.current_z_action[mask].clamp(-1.0, target_z_action)

            action[mask, 0] = 0.0  # keep XY centred on hole
            action[mask, 1] = 0.0
            action[mask, 2] = self.current_z_action[mask]

            # Transition: reached target height
            z_above = mask & (self.current_z_action >= target_z_action - 0.01)
            # Also check that peg has actually moved up significantly
            z_extracted = peg_z - self.initial_peg_z  # positive = up
            extracted_enough = mask & (z_extracted > 0.010)  # at least 10mm up
            next_phase = torch.where(
                z_above & extracted_enough,
                torch.full_like(next_phase, 1),
                next_phase,
            )

            # Timeout: force transition after 250 steps
            timeout = mask & (self.phase_step_count > 250)
            next_phase = torch.where(timeout, torch.full_like(next_phase, 1), next_phase)

        # =================================================================
        # PHASE 1: WOBBLE — small random XY movements above hole
        # =================================================================
        mask = phase_snap == 1
        if mask.any():
            # Index into wobble pattern (clamp to avoid overflow)
            idx = self.phase_step_count[mask].clamp(0, self.wobble_steps - 1).long()
            wobble_xy = self._wobble_pattern[idx]  # (n_masked, 2)

            action[mask, 0] = wobble_xy[:, 0]
            action[mask, 1] = wobble_xy[:, 1]
            # Keep Z at the extraction height
            target_z_action = self.extract_target_z / 0.05
            action[mask, 2] = target_z_action

            # Transition: wobble complete
            done = mask & (self.phase_step_count >= self.wobble_steps)
            next_phase = torch.where(done, torch.full_like(next_phase, 2), next_phase)

        # =================================================================
        # PHASE 2: DONE — hold final position
        # =================================================================
        mask = phase_snap == 2
        if mask.any():
            target_z_action = self.extract_target_z / 0.05
            action[mask, 0] = 0.0
            action[mask, 1] = 0.0
            action[mask, 2] = target_z_action
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

def rollout_peg_extract(env, expert: PegExtractExpert, horizon: int, setup_max_steps: int):
    """Run parallel rollouts for peg extraction.

    1. Setup phase: GT-guided insertion (NOT recorded).
    2. Recording phase: extraction + wobble + done.
    """
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

    # Reset env
    obs_dict, _ = env.reset()
    expert.reset()

    # Override EMA for deterministic scripted collection
    forge_env.ema_factor = torch.ones_like(forge_env.ema_factor)
    print(f"[DEBUG] rollout: Overrode ema_factor to 1.0")

    # ---- Setup: insert peg via env.step() actions (not recorded) ----
    inserted = setup_insert_peg(env, max_steps=setup_max_steps)
    if not inserted:
        print("[WARNING] Setup insertion incomplete — proceeding anyway for debug.")

    # Re-override EMA (setup resets may have changed it)
    forge_env.ema_factor = torch.ones_like(forge_env.ema_factor)

    # Reset expert after setup
    expert.reset()

    # ---- Recording phase: extraction ----
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

    # Refresh obs after setup stepping
    # (The obs_dict from the last setup step is implicit in env state)
    # Get current observation by reading env state
    forge_env._compute_intermediate_values(dt=physics_dt)

    for t in range(horizon):
        if t == 0 or (t + 1) % 100 == 0:
            print(f"[DEBUG] rollout: Step {t+1}/{horizon}")

        # Observations
        obs_dict_curr = forge_env._get_observations()
        if isinstance(obs_dict_curr, dict):
            policy_obs = obs_dict_curr.get("policy", next(iter(obs_dict_curr.values())))
            critic_state = obs_dict_curr.get("critic", policy_obs)
        else:
            policy_obs = critic_state = obs_dict_curr

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
        action = expert.compute_action(
            fingertip_pos=fingertip_pos,
            fingertip_quat=fingertip_quat,
            fixed_pos=hole_pos,
            force_sensor=ft_force,
            peg_pos=peg_pos,
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
                pn = PegExtractExpert.PHASE_NAMES[ph] if ph < len(PegExtractExpert.PHASE_NAMES) else "?"
                fz_val = ftf_np[i, 2]
                # extraction progress: how far peg moved up from initial
                z_ext = (peg_pos[i, 2] - expert.initial_peg_z[i]).item() * 1000
                print(f"[DEBUG] t={t+1} env{i}: phase={pn}, Fz={fz_val:.2f}N, z_extract={z_ext:.2f}mm")

    # Build episode dicts
    results = []
    for i in range(num_envs):
        # Compute extraction success: peg above hole tip
        final_peg_z = peg_pose_lists[i][-1][2]
        final_hole_z = hole_pose_lists[i][-1][2]
        hole_height = forge_env.cfg_task.fixed_asset_cfg.height
        z_above_tip = final_peg_z - (final_hole_z + hole_height)
        ep_success = z_above_tip > 0.005  # peg at least 5mm above hole tip

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
            "phase_names": PegExtractExpert.PHASE_NAMES,
            "episode_length": len(obs_lists[i]),
            "success": ep_success,
            "z_above_tip": float(z_above_tip),
            "wrist_cam_names": wrist_cam_names,
            "task_type": "extract",  # mark as extraction task
        }
        results.append(ep)
        print(f"[DEBUG] env {i}: success={ep_success}, z_above_tip={z_above_tip*1000:.1f}mm, "
              f"final_phase={expert.phase[i].item()}")
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

        expert = PegExtractExpert(num_envs=args.num_envs, device=device)

        episodes = []
        start_time = time.time()
        batch_count = 0

        print(f"\n{'='*60}")
        print(f"Collecting {args.num_episodes} peg EXTRACTION episodes (Task B)")
        print(f"  num_envs={args.num_envs}, horizon={args.horizon}")
        print(f"  setup_max_steps={args.setup_max_steps}")
        print(f"  image_size={args.image_width}x{args.image_height}")
        print(f"{'='*60}\n")

        while len(episodes) < args.num_episodes:
            batch_count += 1
            results = rollout_peg_extract(
                env=env,
                expert=expert,
                horizon=args.horizon,
                setup_max_steps=args.setup_max_steps,
            )

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

    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
