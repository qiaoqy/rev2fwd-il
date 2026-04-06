"""Pick-and-place expert with symmetric Pick↔Lift / Place↔Departure paths (v3).

Based on v2, with:
1. Removed MID_APPROACH intermediate waypoints.
2. Added multi-level Lift phase (4 levels) symmetric to Pick approach.
3. Restructured Departure phase (4 levels) symmetric to Place descent.
4. Pick approach:  APPROACH → ABOVE → CORRECT → ALIGN → GO_TO (5 levels, descending)
   Lift:           LIFT_ALIGN → LIFT_CORRECT → LIFT_ABOVE → LIFT_APPROACH (4 levels, ascending)
   Place descent:  APPROACH → ABOVE → CORRECT → ALIGN → GO_TO (5 levels, descending)
   Departure:      DEPART_ALIGN → DEPART_CORRECT → DEPART_ABOVE → DEPART_APPROACH (4 levels, ascending)
5. All lift/departure parameters independently sampled per-episode.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import gymnasium as gym


class ExpertState(IntEnum):
    """States for the pick-and-place finite state machine (24 states)."""

    REST = 0
    # Pick approach path (5 waypoints: far → close)
    APPROACH_OBJ = 1           # ±0.20, highest
    GO_ABOVE_OBJ = 2           # ±0.08
    CORRECT_OBJ = 3            # ±0.04
    ALIGN_TO_OBJ = 4           # ±0.025
    GO_TO_OBJ = 5              # ±0.01 grasp noise
    CLOSE = 6
    # Lift path (4 waypoints: low → high, symmetric to pick approach)
    LIFT_ALIGN = 7             # ±0.025, sym. ALIGN height
    LIFT_CORRECT = 8           # ±0.04,  sym. CORRECT height
    LIFT_ABOVE = 9             # ±0.08,  sym. ABOVE height
    LIFT_APPROACH = 10         # ±0.20,  sym. APPROACH height
    # Place descent path (5 waypoints: far → close)
    APPROACH_PLACE = 11        # ±0.20
    GO_ABOVE_PLACE = 12        # ±0.08
    CORRECT_PLACE = 13         # ±0.04
    ALIGN_TO_PLACE = 14        # ±0.025
    GO_TO_PLACE = 15           # ±0.01 place noise
    LOWER_TO_RELEASE = 16
    OPEN = 17
    # Departure path (4 waypoints: low → high, symmetric to place descent)
    DEPART_ALIGN = 18          # ±0.025, sym. ALIGN height
    DEPART_CORRECT = 19        # ±0.04,  sym. CORRECT height
    DEPART_ABOVE = 20          # ±0.08,  sym. ABOVE height
    DEPART_APPROACH = 21       # ±0.20,  sym. APPROACH height
    RETURN_REST = 22
    DONE = 23


GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


def _small_random_quat(
    n: int,
    device: torch.device,
    roll_deg: float = 10.0,
    pitch_deg: float = 10.0,
    yaw_deg: float = 15.0,
) -> torch.Tensor:
    """Sample small-angle quaternions (wxyz) with bounded Euler perturbations.

    Returns:
        Quaternion tensor (n, 4) in [w, x, y, z] order.
    """
    roll = torch.empty(n, device=device).uniform_(-roll_deg, roll_deg) * (math.pi / 180.0)
    pitch = torch.empty(n, device=device).uniform_(-pitch_deg, pitch_deg) * (math.pi / 180.0)
    yaw = torch.empty(n, device=device).uniform_(-yaw_deg, yaw_deg) * (math.pi / 180.0)

    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternion tensors (broadcast-safe)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


class PickPlaceExpertRandWaypoint:
    """FSM expert with symmetric Pick↔Lift / Place↔Departure paths (v3).

    24 states total. Pick approach and Lift share the same Z height levels
    (independent random draws). Place descent and Departure also share levels.

    Action format: [x, y, z, qw, qx, qy, qz, gripper] = 8 dims
    Gripper: OPEN = +1, CLOSE = -1
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        hover_z: float = 0.25,
        grasp_z_offset: float = 0.02,
        release_z_offset: float = -0.03,
        position_threshold: float = 0.01,
        wait_steps: int = 10,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.hover_z = hover_z
        self.grasp_z_offset = grasp_z_offset
        self.release_z_offset = release_z_offset
        self.position_threshold = position_threshold
        self.wait_steps = wait_steps

        # State tracking
        self.state = torch.full((num_envs,), ExpertState.REST, dtype=torch.int32, device=self.device)
        self.wait_counter = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        self.rest_pose = None
        self.place_pose = None

        # Fixed grasp orientation (gripper pointing down): wxyz [0,1,0,0]
        self.grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        # Object XY and Z at grasp time (stored when entering CLOSE)
        self.grasp_obj_xy = None
        self.grasp_obj_z = None

        # ---- randomisation params (initialised in first reset) ----
        self._rand_init = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        ee_pose: torch.Tensor,
        place_xy: tuple[float, float],
        place_z: float = 0.055,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = self.num_envs
        d = self.device

        if not self._rand_init:
            z = lambda: torch.zeros(n, device=d)
            z2 = lambda: torch.zeros(n, 2, device=d)
            q = lambda: torch.zeros(n, 4, device=d)

            self.episode_hover_z = z()

            # --- Pick approach: approach / above / correct / align ---
            self.approach_obj_dxy = z2()
            self.approach_obj_extra_z = z()
            self.above_obj_dxy = z2()
            self.above_obj_extra_z = z()
            self.correct_obj_dxy = z2()
            self.correct_obj_extra_z = z()
            self.align_obj_dxy = z2()
            self.align_obj_extra_z = z()

            # --- Lift (4 levels, symmetric to pick approach, independent) ---
            self.lift_align_dxy = z2()
            self.lift_align_extra_z = z()
            self.lift_correct_dxy = z2()
            self.lift_correct_extra_z = z()
            self.lift_above_dxy = z2()
            self.lift_above_extra_z = z()
            self.lift_approach_dxy = z2()
            self.lift_approach_extra_z = z()

            # --- Place descent: approach / above / correct / align ---
            self.approach_place_dxy = z2()
            self.approach_place_extra_z = z()
            self.above_place_dxy = z2()
            self.above_place_extra_z = z()
            self.correct_place_dxy = z2()
            self.correct_place_extra_z = z()
            self.align_place_dxy = z2()
            self.align_place_extra_z = z()

            # --- Grasp / Place noise (X only, no Y) ---
            self.grasp_dx = z()
            self.grasp_dz = z()
            self.place_dx = z()
            self.place_dz = z()

            # --- Departure (4 levels, symmetric to place descent, independent) ---
            self.depart_align_dxy = z2()
            self.depart_align_extra_z = z()
            self.depart_correct_dxy = z2()
            self.depart_correct_extra_z = z()
            self.depart_above_dxy = z2()
            self.depart_above_extra_z = z()
            self.depart_approach_dxy = z2()
            self.depart_approach_extra_z = z()

            # --- orientation perturbation for mid-level waypoints ---
            self.quat_perturbation = q()
            self.quat_perturbation[:, 0] = 1.0  # identity

            self._rand_init = True

        k = len(env_ids)

        # Fixed XY distance coefficients per waypoint level
        APPROACH_R = 0.10  # 10 cm
        ABOVE_R = 0.08     # 8 cm
        CORRECT_R = 0.04   # 4 cm
        ALIGN_R = 0.02     # 2 cm

        # Global hover height
        self.episode_hover_z[env_ids] = torch.empty(k, device=d).uniform_(0.2, 0.3)

        # ==== PICK SIDE (5 levels) — unified direction ====
        pick_theta = torch.empty(k, device=d).uniform_(0, 2 * math.pi)
        pick_dir = torch.stack([torch.cos(pick_theta), torch.sin(pick_theta)], dim=-1)  # (k, 2)

        self.approach_obj_dxy[env_ids] = pick_dir * APPROACH_R
        self.approach_obj_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.10)

        self.above_obj_dxy[env_ids] = pick_dir * ABOVE_R
        self.above_obj_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.03)

        self.correct_obj_dxy[env_ids] = pick_dir * CORRECT_R
        self.correct_obj_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.08, 0.12)

        self.align_obj_dxy[env_ids] = pick_dir * ALIGN_R
        self.align_obj_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.06)

        # ==== LIFT SIDE (4 levels) — unified direction (independent from pick) ====
        lift_theta = torch.empty(k, device=d).uniform_(0, 2 * math.pi)
        lift_dir = torch.stack([torch.cos(lift_theta), torch.sin(lift_theta)], dim=-1)

        self.lift_align_dxy[env_ids] = lift_dir * ALIGN_R
        self.lift_align_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.06)

        self.lift_correct_dxy[env_ids] = lift_dir * CORRECT_R
        self.lift_correct_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.08, 0.12)

        self.lift_above_dxy[env_ids] = lift_dir * ABOVE_R
        self.lift_above_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.03)

        self.lift_approach_dxy[env_ids] = lift_dir * APPROACH_R
        self.lift_approach_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.10)

        # ==== PLACE SIDE (5 levels) — unified direction ====
        place_theta = torch.empty(k, device=d).uniform_(0, 2 * math.pi)
        place_dir = torch.stack([torch.cos(place_theta), torch.sin(place_theta)], dim=-1)

        self.approach_place_dxy[env_ids] = place_dir * APPROACH_R
        self.approach_place_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.10)

        self.above_place_dxy[env_ids] = place_dir * ABOVE_R
        self.above_place_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.03)

        self.correct_place_dxy[env_ids] = place_dir * CORRECT_R
        self.correct_place_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.08, 0.12)

        self.align_place_dxy[env_ids] = place_dir * ALIGN_R
        self.align_place_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.06)

        # ==== GRASP / PLACE NOISE: X +[0, 1cm], Y=0, Z [0, 2cm] ====
        self.grasp_dx[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.01)
        self.grasp_dz[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.02)
        self.place_dx[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.01)
        self.place_dz[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.02)

        # ==== DEPARTURE (4 levels) — unified direction (independent from place) ====
        depart_theta = torch.empty(k, device=d).uniform_(0, 2 * math.pi)
        depart_dir = torch.stack([torch.cos(depart_theta), torch.sin(depart_theta)], dim=-1)

        self.depart_align_dxy[env_ids] = depart_dir * ALIGN_R
        self.depart_align_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.06)

        self.depart_correct_dxy[env_ids] = depart_dir * CORRECT_R
        self.depart_correct_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.08, 0.12)

        self.depart_above_dxy[env_ids] = depart_dir * ABOVE_R
        self.depart_above_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.03)

        self.depart_approach_dxy[env_ids] = depart_dir * APPROACH_R
        self.depart_approach_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.10)

        # ---- Orientation perturbation (disabled – no rotation randomisation) ----
        self.quat_perturbation[env_ids] = _small_random_quat(k, d, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0)

        # ---- Rest pose (override orientation to gripper-down) ----
        if self.rest_pose is None:
            self.rest_pose = ee_pose.clone()
            self.rest_pose[:, 3:7] = self.grasp_quat
        else:
            self.rest_pose[env_ids] = ee_pose[env_ids].clone()
            self.rest_pose[env_ids, 3:7] = self.grasp_quat

        # ---- Place pose ----
        if self.place_pose is None:
            self.place_pose = torch.zeros(self.num_envs, 7, device=d)

        self.place_pose[env_ids, 0] = place_xy[0]
        self.place_pose[env_ids, 1] = place_xy[1]
        self.place_pose[env_ids, 2] = place_z
        self.place_pose[env_ids, 3:7] = self.grasp_quat

        # ---- State ----
        self.state[env_ids] = ExpertState.APPROACH_OBJ
        self.wait_counter[env_ids] = 0

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------
    def act(
        self,
        ee_pose: torch.Tensor,
        object_pose: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.zeros(self.num_envs, 8, device=self.device)
        action[:, :7] = ee_pose.clone()
        action[:, 7] = GRIPPER_OPEN

        obj_xy = object_pose[:, :2]
        obj_z = object_pose[:, 2]

        # Pre-compute perturbed orientation for mid-level waypoints
        perturbed_quat = _quat_mul(self.grasp_quat.unsqueeze(0), self.quat_perturbation)  # (N,4)

        # ---- pick-side waypoints (5 levels: approach → above → correct → align → grasp) ----
        approach_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        approach_obj_pos[:, :2] = obj_xy + self.approach_obj_dxy
        approach_obj_pos[:, 2] = self.episode_hover_z + self.approach_obj_extra_z

        above_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        above_obj_pos[:, :2] = obj_xy + self.above_obj_dxy
        above_obj_pos[:, 2] = self.episode_hover_z + self.above_obj_extra_z

        correct_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        correct_obj_pos[:, :2] = obj_xy + self.correct_obj_dxy
        correct_obj_pos[:, 2] = obj_z + self.grasp_z_offset + self.correct_obj_extra_z

        align_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        align_obj_pos[:, :2] = obj_xy + self.align_obj_dxy
        align_obj_pos[:, 2] = obj_z + self.grasp_z_offset + self.align_obj_extra_z

        at_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        at_obj_pos[:, 0] = obj_xy[:, 0] + self.grasp_dx
        at_obj_pos[:, 1] = obj_xy[:, 1]
        at_obj_pos[:, 2] = obj_z + self.grasp_z_offset

        # ---- lift-side waypoints (4 levels, ascending from grasp, XY rel grasp_obj) ----
        lift_base_xy = self.grasp_obj_xy if self.grasp_obj_xy is not None else obj_xy
        lift_base_z = self.grasp_obj_z if self.grasp_obj_z is not None else obj_z

        lift_align_pos = torch.zeros(self.num_envs, 3, device=self.device)
        lift_align_pos[:, :2] = lift_base_xy + self.lift_align_dxy
        lift_align_pos[:, 2] = lift_base_z + self.grasp_z_offset + self.lift_align_extra_z

        lift_correct_pos = torch.zeros(self.num_envs, 3, device=self.device)
        lift_correct_pos[:, :2] = lift_base_xy + self.lift_correct_dxy
        lift_correct_pos[:, 2] = lift_base_z + self.grasp_z_offset + self.lift_correct_extra_z

        lift_above_pos = torch.zeros(self.num_envs, 3, device=self.device)
        lift_above_pos[:, :2] = lift_base_xy + self.lift_above_dxy
        lift_above_pos[:, 2] = self.episode_hover_z + self.lift_above_extra_z

        lift_approach_pos = torch.zeros(self.num_envs, 3, device=self.device)
        lift_approach_pos[:, :2] = lift_base_xy + self.lift_approach_dxy
        lift_approach_pos[:, 2] = self.episode_hover_z + self.lift_approach_extra_z

        # ---- place-side waypoints (5 levels, descent) ----
        approach_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        approach_place_pos[:, :2] = self.place_pose[:, :2] + self.approach_place_dxy
        approach_place_pos[:, 2] = self.episode_hover_z + self.approach_place_extra_z

        above_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        above_place_pos[:, :2] = self.place_pose[:, :2] + self.above_place_dxy
        above_place_pos[:, 2] = self.episode_hover_z + self.above_place_extra_z

        correct_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        correct_place_pos[:, :2] = self.place_pose[:, :2] + self.correct_place_dxy
        correct_place_pos[:, 2] = self.place_pose[:, 2] + self.correct_place_extra_z

        align_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        align_place_pos[:, :2] = self.place_pose[:, :2] + self.align_place_dxy
        align_place_pos[:, 2] = self.place_pose[:, 2] + self.align_place_extra_z

        at_place_pos = self.place_pose[:, :3].clone()
        at_place_pos[:, 0] = at_place_pos[:, 0] + self.place_dx

        release_pos = self.place_pose[:, :3].clone()
        release_pos[:, 0] = release_pos[:, 0] + self.place_dx
        release_pos[:, 2] = self.place_pose[:, 2] + self.release_z_offset

        # ---- departure waypoints (4 levels, ascending from release, independent) ----
        depart_align_pos = torch.zeros(self.num_envs, 3, device=self.device)
        depart_align_pos[:, :2] = self.place_pose[:, :2] + self.depart_align_dxy
        depart_align_pos[:, 2] = self.place_pose[:, 2] + self.depart_align_extra_z

        depart_correct_pos = torch.zeros(self.num_envs, 3, device=self.device)
        depart_correct_pos[:, :2] = self.place_pose[:, :2] + self.depart_correct_dxy
        depart_correct_pos[:, 2] = self.place_pose[:, 2] + self.depart_correct_extra_z

        depart_above_pos = torch.zeros(self.num_envs, 3, device=self.device)
        depart_above_pos[:, :2] = self.place_pose[:, :2] + self.depart_above_dxy
        depart_above_pos[:, 2] = self.episode_hover_z + self.depart_above_extra_z

        depart_approach_pos = torch.zeros(self.num_envs, 3, device=self.device)
        depart_approach_pos[:, :2] = self.place_pose[:, :2] + self.depart_approach_dxy
        depart_approach_pos[:, 2] = self.episode_hover_z + self.depart_approach_extra_z

        # ---- FSM logic ----
        for state_val in ExpertState:
            mask = self.state == state_val
            if not mask.any():
                continue

            if state_val == ExpertState.REST:
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

            # ==== PICK APPROACH PATH (5 levels) ====
            elif state_val == ExpertState.APPROACH_OBJ:
                action[mask, :3] = approach_obj_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, approach_obj_pos, ExpertState.GO_ABOVE_OBJ)

            elif state_val == ExpertState.GO_ABOVE_OBJ:
                action[mask, :3] = above_obj_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, above_obj_pos, ExpertState.CORRECT_OBJ)

            elif state_val == ExpertState.CORRECT_OBJ:
                action[mask, :3] = correct_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, correct_obj_pos, ExpertState.ALIGN_TO_OBJ)

            elif state_val == ExpertState.ALIGN_TO_OBJ:
                action[mask, :3] = align_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, align_obj_pos, ExpertState.GO_TO_OBJ)

            elif state_val == ExpertState.GO_TO_OBJ:
                action[mask, :3] = at_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                # Wait before closing gripper to let EE stabilise
                dist = torch.norm(ee_pose[mask, :3] - at_obj_pos[mask], dim=-1)
                reached = dist < self.position_threshold
                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1
                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.CLOSE
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.CLOSE:
                action[mask, :3] = at_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                # Store object table XY and Z before lifting
                if self.grasp_obj_xy is None:
                    self.grasp_obj_xy = obj_xy.clone()
                    self.grasp_obj_z = obj_z.clone()
                self.grasp_obj_xy[mask] = obj_xy[mask]
                self.grasp_obj_z[mask] = obj_z[mask]
                self.wait_counter[mask] += 1
                transition = mask & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.LIFT_ALIGN
                self.wait_counter[transition] = 0

            # ==== LIFT PATH (4 levels, ascending) ====
            elif state_val == ExpertState.LIFT_ALIGN:
                action[mask, :3] = lift_align_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, lift_align_pos, ExpertState.LIFT_CORRECT)

            elif state_val == ExpertState.LIFT_CORRECT:
                action[mask, :3] = lift_correct_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, lift_correct_pos, ExpertState.LIFT_ABOVE)

            elif state_val == ExpertState.LIFT_ABOVE:
                action[mask, :3] = lift_above_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, lift_above_pos, ExpertState.LIFT_APPROACH)

            elif state_val == ExpertState.LIFT_APPROACH:
                action[mask, :3] = lift_approach_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, lift_approach_pos, ExpertState.APPROACH_PLACE)

            # ==== PLACE DESCENT PATH (5 levels) ====
            elif state_val == ExpertState.APPROACH_PLACE:
                action[mask, :3] = approach_place_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, approach_place_pos, ExpertState.GO_ABOVE_PLACE)

            elif state_val == ExpertState.GO_ABOVE_PLACE:
                action[mask, :3] = above_place_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, above_place_pos, ExpertState.CORRECT_PLACE)

            elif state_val == ExpertState.CORRECT_PLACE:
                action[mask, :3] = correct_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, correct_place_pos, ExpertState.ALIGN_TO_PLACE)

            elif state_val == ExpertState.ALIGN_TO_PLACE:
                action[mask, :3] = align_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, align_place_pos, ExpertState.GO_TO_PLACE)

            elif state_val == ExpertState.GO_TO_PLACE:
                action[mask, :3] = at_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition(mask, ee_pose, at_place_pos, ExpertState.LOWER_TO_RELEASE)

            elif state_val == ExpertState.LOWER_TO_RELEASE:
                action[mask, :3] = release_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                # Wait before opening gripper to let object settle
                dist = torch.norm(ee_pose[mask, :3] - release_pos[mask], dim=-1)
                reached = dist < self.position_threshold
                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1
                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.OPEN
                self.wait_counter[transition] = 0

            elif state_val == ExpertState.OPEN:
                action[mask, :3] = release_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self.wait_counter[mask] += 1
                transition = mask & (self.wait_counter >= self.wait_steps)
                self.state[transition] = ExpertState.DEPART_ALIGN
                self.wait_counter[transition] = 0

            # ==== DEPARTURE PATH (4 levels, ascending) ====
            elif state_val == ExpertState.DEPART_ALIGN:
                action[mask, :3] = depart_align_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, depart_align_pos, ExpertState.DEPART_CORRECT)

            elif state_val == ExpertState.DEPART_CORRECT:
                action[mask, :3] = depart_correct_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, depart_correct_pos, ExpertState.DEPART_ABOVE)

            elif state_val == ExpertState.DEPART_ABOVE:
                action[mask, :3] = depart_above_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, depart_above_pos, ExpertState.DEPART_APPROACH)

            elif state_val == ExpertState.DEPART_APPROACH:
                action[mask, :3] = depart_approach_pos[mask]
                action[mask, 3:7] = perturbed_quat[mask]
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, depart_approach_pos, ExpertState.RETURN_REST)

            elif state_val == ExpertState.RETURN_REST:
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN
                self._transition(mask, ee_pose, self.rest_pose[:, :3], ExpertState.DONE)

            elif state_val == ExpertState.DONE:
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

        return action

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _transition(
        self,
        mask: torch.Tensor,
        ee_pose: torch.Tensor,
        target_pos: torch.Tensor,
        next_state: ExpertState,
    ) -> None:
        dist = torch.norm(ee_pose[mask, :3] - target_pos[mask], dim=-1)
        reached = dist < self.position_threshold
        reached_envs = mask.clone()
        reached_envs[mask] = reached
        self.state[reached_envs] = next_state

    def get_state_names(self) -> list[str]:
        return [ExpertState(s.item()).name for s in self.state]

    def is_done(self) -> torch.Tensor:
        return self.state == ExpertState.DONE
