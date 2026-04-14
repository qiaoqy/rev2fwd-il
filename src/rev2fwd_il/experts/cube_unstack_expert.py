"""3-cube unstack FSM expert for Exp38.

Sequentially picks cubes from a vertical stack (top→bottom) and places each
at a pre-sampled position inside the red rectangular region, then returns to
rest.  After all three rounds the expert transitions to DONE.

Action format:  [x, y, z, qw, qx, qy, qz, gripper]  (8-dim, absolute EE pose)
Gripper:  +1 = open,  -1 = close
"""

from __future__ import annotations

from enum import IntEnum

import torch

# ---------------------------------------------------------------------------
# FSM states (reused every round)
# ---------------------------------------------------------------------------

class UnstackState(IntEnum):
    REST = 0
    APPROACH_OBJ = 1
    GO_ABOVE_OBJ = 2
    ALIGN_TO_OBJ = 3
    GO_TO_OBJ = 4
    CLOSE = 5
    LIFT = 6
    APPROACH_PLACE = 7
    GO_ABOVE_PLACE = 8
    ALIGN_TO_PLACE = 9
    GO_TO_PLACE = 10
    LOWER_TO_RELEASE = 11
    OPEN = 12
    LIFT_AFTER_RELEASE = 13
    RETURN_REST = 14
    DONE = 15


GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0


class CubeUnstackExpert:
    """FSM expert that unstacks 3 cubes (top→bottom) and scatters them.

    Parameters
    ----------
    num_envs : int
    device : str | torch.device
    pick_order : list[str]
        Cube scene-key names in pick order (top→bottom).
    cube_params : dict[str, dict]
        Per-cube parameters keyed by scene_key.  Each dict must contain
        ``grasp_z_offset``, ``release_z_offset``, ``place_z``.
    hover_z, position_threshold, wait_steps : float / int
        Shared FSM tuning knobs.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        pick_order: list[str],
        cube_params: dict[str, dict],
        hover_z: float = 0.30,
        position_threshold: float = 0.015,
        wait_steps: int = 10,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.pick_order = pick_order  # ["cube_small", "cube_medium", "cube_large"]
        self.cube_params = cube_params
        self.hover_z = hover_z
        self.position_threshold = position_threshold
        self.wait_steps = wait_steps
        self.num_rounds = len(pick_order)

        # per-env tracking
        self.state = torch.full((num_envs,), UnstackState.REST, dtype=torch.int32, device=self.device)
        self.wait_counter = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.round_idx = torch.zeros(num_envs, dtype=torch.int32, device=self.device)  # 0-based

        self.rest_pose = None  # (N, 7)
        self.grasp_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)

        # place targets: (N, 3, 2) — 3 rounds × (x, y)
        self.place_targets = None

        # frozen EE pose at grasp — set when entering CLOSE, used by CLOSE & LIFT
        self.grasp_pos = torch.zeros(num_envs, 3, device=self.device)

        # per-round randomization buffers
        self._rand_init = False
        self._alloc_randomization()

    # ------------------------------------------------------------------
    # Randomization buffers
    # ------------------------------------------------------------------

    def _alloc_randomization(self):
        n = self.num_envs
        d = self.device
        self.episode_hover_z = torch.zeros(n, device=d)
        self.approach_obj_dxy = torch.zeros(n, 2, device=d)
        self.approach_obj_extra_z = torch.zeros(n, device=d)
        self.above_obj_dxy = torch.zeros(n, 2, device=d)
        self.above_obj_extra_z = torch.zeros(n, device=d)
        self.align_obj_dxy = torch.zeros(n, 2, device=d)
        self.align_obj_dz = torch.zeros(n, device=d)
        self.approach_place_dxy = torch.zeros(n, 2, device=d)
        self.approach_place_extra_z = torch.zeros(n, device=d)
        self.above_place_dxy = torch.zeros(n, 2, device=d)
        self.above_place_extra_z = torch.zeros(n, device=d)
        self.align_place_dxy = torch.zeros(n, 2, device=d)
        self.align_place_dz = torch.zeros(n, device=d)
        self.lift_extra_z = torch.zeros(n, device=d)
        self._rand_init = True

    def _randomize_waypoints(self, env_ids=None):
        """Sample fresh randomisation for one round of pick-place."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        k = len(env_ids)
        d = self.device

        self.episode_hover_z[env_ids] = torch.empty(k, device=d).uniform_(0.25, 0.35)
        self.approach_obj_dxy[env_ids] = torch.empty(k, 2, device=d).uniform_(-0.03, 0.03)
        self.approach_obj_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.08)
        self.above_obj_dxy[env_ids] = torch.empty(k, 2, device=d).uniform_(-0.01, 0.01)
        self.above_obj_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.03)
        self.align_obj_dxy[env_ids] = torch.empty(k, 2, device=d).uniform_(-0.003, 0.003)
        self.align_obj_dz[env_ids] = torch.empty(k, device=d).uniform_(-0.02, 0.0)
        self.approach_place_dxy[env_ids] = torch.empty(k, 2, device=d).uniform_(-0.03, 0.03)
        self.approach_place_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.03, 0.08)
        self.above_place_dxy[env_ids] = torch.empty(k, 2, device=d).uniform_(-0.01, 0.01)
        self.above_place_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.0, 0.03)
        self.align_place_dxy[env_ids] = torch.empty(k, 2, device=d).uniform_(-0.003, 0.003)
        self.align_place_dz[env_ids] = torch.empty(k, device=d).uniform_(-0.02, 0.0)
        self.lift_extra_z[env_ids] = torch.empty(k, device=d).uniform_(0.08, 0.10)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, ee_pose: torch.Tensor, place_targets: torch.Tensor):
        """Reset the expert for a new episode.

        Args:
            ee_pose: (N, 7) current EE pose.
            place_targets: (N, 3, 2) — pre-sampled place XY for each round.
        """
        self.rest_pose = ee_pose.clone()
        self.rest_pose[:, 3:7] = self.grasp_quat
        self.place_targets = place_targets.to(self.device)
        self.round_idx[:] = 0
        self.state[:] = UnstackState.APPROACH_OBJ
        self.wait_counter[:] = 0
        self._randomize_waypoints()

    # ------------------------------------------------------------------
    # Helpers: per-round cube / place retrieval
    # ------------------------------------------------------------------

    def current_cube_name(self, env_idx: int) -> str:
        r = self.round_idx[env_idx].item()
        r = min(r, self.num_rounds - 1)
        return self.pick_order[r]

    def _get_current_cube_pose(self, all_cube_poses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return (N, 7) pose of the cube being picked this round."""
        result = torch.zeros(self.num_envs, 7, device=self.device)
        for r in range(self.num_rounds):
            mask = self.round_idx == r
            if mask.any():
                result[mask] = all_cube_poses[self.pick_order[r]][mask]
        return result

    def _get_current_place_pose(self) -> torch.Tensor:
        """Return (N, 7) place target for this round (z = cube-specific place_z)."""
        pose = torch.zeros(self.num_envs, 7, device=self.device)
        for r in range(self.num_rounds):
            mask = self.round_idx == r
            if mask.any():
                pose[mask, 0] = self.place_targets[mask, r, 0]
                pose[mask, 1] = self.place_targets[mask, r, 1]
                cp = self.cube_params[self.pick_order[r]]
                pose[mask, 2] = cp["place_z"]
                pose[mask, 3:7] = self.grasp_quat
        return pose

    def _get_current_grasp_z_offset(self) -> torch.Tensor:
        """Return (N,) grasp_z_offset for current round's cube."""
        out = torch.zeros(self.num_envs, device=self.device)
        for r in range(self.num_rounds):
            mask = self.round_idx == r
            if mask.any():
                out[mask] = self.cube_params[self.pick_order[r]]["grasp_z_offset"]
        return out

    def _get_current_release_z_offset(self) -> torch.Tensor:
        out = torch.zeros(self.num_envs, device=self.device)
        for r in range(self.num_rounds):
            mask = self.round_idx == r
            if mask.any():
                out[mask] = self.cube_params[self.pick_order[r]]["release_z_offset"]
        return out

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    def act(self, ee_pose: torch.Tensor, all_cube_poses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute (N, 8) action.

        Args:
            ee_pose: (N, 7)
            all_cube_poses: dict mapping scene_key → (N, 7) pose tensor.
        """
        action = torch.zeros(self.num_envs, 8, device=self.device)
        action[:, :7] = ee_pose.clone()
        action[:, 7] = GRIPPER_OPEN

        cube_pose = self._get_current_cube_pose(all_cube_poses)
        place_pose = self._get_current_place_pose()
        grasp_z_off = self._get_current_grasp_z_offset()
        release_z_off = self._get_current_release_z_offset()

        obj_xy = cube_pose[:, :2]
        obj_z = cube_pose[:, 2]

        # Waypoints (same structure as PickPlaceExpertB)
        approach_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        approach_obj_pos[:, :2] = obj_xy + self.approach_obj_dxy
        approach_obj_pos[:, 2] = self.episode_hover_z + self.approach_obj_extra_z

        above_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        above_obj_pos[:, :2] = obj_xy + self.above_obj_dxy
        above_obj_pos[:, 2] = self.episode_hover_z + self.above_obj_extra_z

        align_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        align_obj_pos[:, :2] = obj_xy + self.align_obj_dxy
        align_obj_pos[:, 2] = self.episode_hover_z + self.align_obj_dz

        at_obj_pos = torch.zeros(self.num_envs, 3, device=self.device)
        at_obj_pos[:, :2] = obj_xy
        at_obj_pos[:, 2] = obj_z + grasp_z_off

        approach_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        approach_place_pos[:, :2] = place_pose[:, :2] + self.approach_place_dxy
        approach_place_pos[:, 2] = self.episode_hover_z + self.approach_place_extra_z

        above_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        above_place_pos[:, :2] = place_pose[:, :2] + self.above_place_dxy
        above_place_pos[:, 2] = self.episode_hover_z + self.above_place_extra_z

        align_place_pos = torch.zeros(self.num_envs, 3, device=self.device)
        align_place_pos[:, :2] = place_pose[:, :2] + self.align_place_dxy
        align_place_pos[:, 2] = self.episode_hover_z + self.align_place_dz

        at_place_pos = place_pose[:, :3].clone()

        release_pos = place_pose[:, :3].clone()
        release_pos[:, 2] = place_pose[:, 2] + release_z_off

        lift_pos = torch.zeros(self.num_envs, 3, device=self.device)
        lift_pos[:, :2] = self.grasp_pos[:, :2]
        lift_pos[:, 2] = self.episode_hover_z + self.lift_extra_z

        # State machine transitions
        for sv in UnstackState:
            mask = self.state == sv
            if not mask.any():
                continue

            if sv == UnstackState.REST:
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

            elif sv == UnstackState.APPROACH_OBJ:
                action[mask, :3] = approach_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition_on_reach(mask, ee_pose, approach_obj_pos, UnstackState.GO_ABOVE_OBJ)

            elif sv == UnstackState.GO_ABOVE_OBJ:
                action[mask, :3] = above_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition_on_reach(mask, ee_pose, above_obj_pos, UnstackState.ALIGN_TO_OBJ)

            elif sv == UnstackState.ALIGN_TO_OBJ:
                action[mask, :3] = align_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition_on_reach(mask, ee_pose, align_obj_pos, UnstackState.GO_TO_OBJ)

            elif sv == UnstackState.GO_TO_OBJ:
                action[mask, :3] = at_obj_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                # snapshot EE XYZ right before transitioning to CLOSE
                dist = torch.norm(ee_pose[mask, :3] - at_obj_pos[mask], dim=-1)
                about_to_close = dist < self.position_threshold
                snapshot_envs = mask.clone()
                snapshot_envs[mask] = about_to_close
                if snapshot_envs.any():
                    self.grasp_pos[snapshot_envs] = ee_pose[snapshot_envs, :3]
                self._transition_on_reach(mask, ee_pose, at_obj_pos, UnstackState.CLOSE)

            elif sv == UnstackState.CLOSE:
                # use fully frozen grasp_pos (XYZ) — no live cube tracking
                action[mask, :3] = self.grasp_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self.wait_counter[mask] += 1
                transition = mask & (self.wait_counter >= self.wait_steps)
                self.state[transition] = UnstackState.LIFT
                self.wait_counter[transition] = 0

            elif sv == UnstackState.LIFT:
                action[mask, :3] = lift_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition_on_reach(mask, ee_pose, lift_pos, UnstackState.APPROACH_PLACE)

            elif sv == UnstackState.APPROACH_PLACE:
                action[mask, :3] = approach_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition_on_reach(mask, ee_pose, approach_place_pos, UnstackState.GO_ABOVE_PLACE)

            elif sv == UnstackState.GO_ABOVE_PLACE:
                action[mask, :3] = above_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition_on_reach(mask, ee_pose, above_place_pos, UnstackState.ALIGN_TO_PLACE)

            elif sv == UnstackState.ALIGN_TO_PLACE:
                action[mask, :3] = align_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition_on_reach(mask, ee_pose, align_place_pos, UnstackState.GO_TO_PLACE)

            elif sv == UnstackState.GO_TO_PLACE:
                action[mask, :3] = at_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition_on_reach(mask, ee_pose, at_place_pos, UnstackState.LOWER_TO_RELEASE)

            elif sv == UnstackState.LOWER_TO_RELEASE:
                action[mask, :3] = release_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_CLOSE
                self._transition_on_reach(mask, ee_pose, release_pos, UnstackState.OPEN)

            elif sv == UnstackState.OPEN:
                action[mask, :3] = release_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self.wait_counter[mask] += 1
                transition = mask & (self.wait_counter >= self.wait_steps)
                self.state[transition] = UnstackState.LIFT_AFTER_RELEASE
                self.wait_counter[transition] = 0

            elif sv == UnstackState.LIFT_AFTER_RELEASE:
                action[mask, :3] = above_place_pos[mask]
                action[mask, 3:7] = self.grasp_quat
                action[mask, 7] = GRIPPER_OPEN
                self._transition_on_reach(mask, ee_pose, above_place_pos, UnstackState.RETURN_REST)

            elif sv == UnstackState.RETURN_REST:
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN
                # On reaching rest → advance round or finish
                dist = torch.norm(ee_pose[mask, :3] - self.rest_pose[mask, :3], dim=-1)
                reached = dist < self.position_threshold
                reached_envs = mask.clone()
                reached_envs[mask] = reached
                self.wait_counter[reached_envs] += 1
                transition = reached_envs & (self.wait_counter >= self.wait_steps)
                if transition.any():
                    self.wait_counter[transition] = 0
                    # check if more rounds remain
                    more = transition & (self.round_idx < self.num_rounds - 1)
                    done = transition & (self.round_idx >= self.num_rounds - 1)
                    if more.any():
                        self.round_idx[more] += 1
                        self.state[more] = UnstackState.APPROACH_OBJ
                        self._randomize_waypoints(more.nonzero(as_tuple=False).squeeze(-1))
                    if done.any():
                        self.state[done] = UnstackState.DONE

            elif sv == UnstackState.DONE:
                action[mask, :3] = self.rest_pose[mask, :3]
                action[mask, 3:7] = self.rest_pose[mask, 3:7]
                action[mask, 7] = GRIPPER_OPEN

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition_on_reach(self, mask, ee_pose, target_pos, next_state):
        """Check position-reach and advance state for envs in *mask*."""
        dist = torch.norm(ee_pose[mask, :3] - target_pos[mask], dim=-1)
        reached = dist < self.position_threshold
        reached_envs = mask.clone()
        reached_envs[mask] = reached
        self.wait_counter[reached_envs] += 1
        transition = reached_envs & (self.wait_counter >= self.wait_steps)
        self.state[transition] = next_state
        self.wait_counter[transition] = 0

    def is_done(self) -> torch.Tensor:
        return self.state == UnstackState.DONE

    def get_state_names(self) -> list[str]:
        return [UnstackState(s.item()).name for s in self.state]
