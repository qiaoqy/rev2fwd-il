"""2-round partial-unstack FSM expert for Exp44.

Picks cube_small (top) then cube_medium from a 3-cube stack and places each
at a fixed position on the table.  cube_large remains untouched.

This module re-exports the generic ``CubeUnstackExpert`` from exp38 and
provides a convenience factory ``make_exp44_expert()`` that wires it with
the correct Exp44 parameters.

Action format:  [x, y, z, qw, qx, qy, qz, gripper]  (8-dim, absolute EE pose)
Gripper:  +1 = open,  -1 = close
"""

from __future__ import annotations

import torch

from rev2fwd_il.experts.cube_unstack_expert import CubeUnstackExpert, UnstackState  # noqa: F401
from rev2fwd_il.sim.exp44_registry import (
    CUBE_DEFS_44,
    DEFAULT_EXP44_CONFIG,
    PICK_ORDER_44,
)


def make_exp44_expert(
    num_envs: int,
    device: str | torch.device,
) -> CubeUnstackExpert:
    """Create a ``CubeUnstackExpert`` pre-configured for the Exp44 task.

    Only 2 rounds: cube_small → fixed pos, then cube_medium → fixed pos.
    """
    cfg = DEFAULT_EXP44_CONFIG

    cube_params: dict[str, dict] = {}
    for cd in CUBE_DEFS_44:
        cube_params[cd.name] = dict(
            grasp_z_offset=cd.grasp_z_offset,
            release_z_offset=cd.release_z_offset,
            place_z=cd.place_z,
        )

    return CubeUnstackExpert(
        num_envs=num_envs,
        device=device,
        pick_order=PICK_ORDER_44,
        cube_params=cube_params,
        hover_z=cfg.hover_z,
        position_threshold=cfg.position_threshold,
        wait_steps=cfg.wait_steps,
    )


def make_fixed_place_targets(
    num_envs: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Return (num_envs, 2, 2) fixed place-target tensor for the 2 rounds.

    Round 0 (cube_small):  small_place_xy
    Round 1 (cube_medium): medium_place_xy
    """
    cfg = DEFAULT_EXP44_CONFIG
    targets = torch.zeros(num_envs, 2, 2, device=device)
    targets[:, 0, 0] = cfg.small_place_xy[0]
    targets[:, 0, 1] = cfg.small_place_xy[1]
    targets[:, 1, 0] = cfg.medium_place_xy[0]
    targets[:, 1, 1] = cfg.medium_place_xy[1]
    return targets
