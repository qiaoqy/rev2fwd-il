"""Registry for the 3-cube partial-unstack task (Exp44).

Three cubes of different sizes/colours start vertically stacked at a fixed
position on the table.  The FSM expert removes only cube_small and
cube_medium, placing each at a fixed position.  cube_large stays in place.

Cube sizes:  large=8cm (blue), medium=6cm (yellow), small=4cm (red).
No table markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Cube definition
# ---------------------------------------------------------------------------

@dataclass
class CubeDef44:
    """Single cube definition for Exp44."""

    name: str  # e.g. "cube_large"
    scene_key: str  # Isaac Lab scene key (same as name)
    edge_length: float  # full edge length (m) — used for XY
    color: tuple[float, float, float]  # RGB 0-1
    mass: float  # kg
    init_z: float  # centre z when stacked on table
    height: float | None = None  # z-dimension override (None → same as edge_length)
    grasp_z_offset: float = -0.01  # z tweak when grasping (1cm below centre)
    release_z_offset: float = -0.03  # z tweak when releasing
    place_z: float = 0.055  # EE target z for GO_TO_PLACE
    goal_z: float = 0.025  # object centre z on table (half-edge)


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass
class Exp44Config:
    """Full configuration for the Exp44 partial-unstack task."""

    cubes: list[CubeDef44] = field(default_factory=list)  # [large, medium, small]

    # Fixed positions (XY on table)
    stack_xy: tuple[float, float] = (0.5, 0.0)  # cube_large stays here
    medium_place_xy: tuple[float, float] = (0.5, 0.15)  # Task B target for medium
    small_place_xy: tuple[float, float] = (0.5, -0.15)  # Task B target for small

    # FSM parameters
    hover_z: float = 0.30
    position_threshold: float = 0.015
    wait_steps: int = 10
    success_radius: float = 0.03


# ---------------------------------------------------------------------------
# Cube definitions  (bottom → top in the stack)
# ---------------------------------------------------------------------------

CUBE_LARGE_44 = CubeDef44(
    name="cube_large",
    scene_key="cube_large",
    edge_length=0.080,
    color=(0.2, 0.4, 1.0),  # blue
    mass=1.0,
    init_z=0.010,  # half-height on table (height=0.02)
    height=0.020,  # XY stays 8cm, Z reduced to 2cm
    grasp_z_offset=-0.01,
    release_z_offset=-0.03,
    place_z=0.070,  # not used (large is never placed)
    goal_z=0.010,  # half-height
)

CUBE_MEDIUM_44 = CubeDef44(
    name="cube_medium",
    scene_key="cube_medium",
    edge_length=0.060,
    color=(1.0, 0.85, 0.0),  # yellow
    mass=0.10,
    init_z=0.050,  # bottom=0.020, centre=0.020+0.030
    grasp_z_offset=-0.01,
    release_z_offset=-0.03,
    place_z=0.060,
    goal_z=0.030,
)

CUBE_SMALL_44 = CubeDef44(
    name="cube_small",
    scene_key="cube_small",
    edge_length=0.040,
    color=(1.0, 0.2, 0.2),  # red
    mass=0.06,
    init_z=0.100,  # bottom=0.080, centre=0.080+0.020
    grasp_z_offset=-0.01,
    release_z_offset=-0.03,
    place_z=0.050,
    goal_z=0.020,
)

# Ordered bottom → top
CUBE_DEFS_44: list[CubeDef44] = [CUBE_LARGE_44, CUBE_MEDIUM_44, CUBE_SMALL_44]

# Pick order: top → bottom (only the two that move)
PICK_ORDER_44: list[str] = ["cube_small", "cube_medium"]

DEFAULT_EXP44_CONFIG = Exp44Config(cubes=CUBE_DEFS_44)


# ---------------------------------------------------------------------------
# Spawn helpers (lazy imports to avoid importing Isaac Lab at module level)
# ---------------------------------------------------------------------------

def _rigid_props():
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )


def make_cuboid_spawn_44(cube_def: CubeDef44):
    """Return a CuboidCfg spawn config for *cube_def*."""
    import isaaclab.sim as sim_utils

    e = cube_def.edge_length
    h = cube_def.height if cube_def.height is not None else e
    return sim_utils.CuboidCfg(
        size=(e, e, h),
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=cube_def.mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=cube_def.color,
        ),
    )


def get_cube_def_44(name: str) -> CubeDef44:
    """Look up a CubeDef44 by name."""
    for c in CUBE_DEFS_44:
        if c.name == name:
            return c
    raise KeyError(f"Unknown cube name: {name!r}")
