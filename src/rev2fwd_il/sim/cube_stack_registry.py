"""Registry for the 3-cube unstack task (Exp38).

Defines three cubes of different sizes/colours that start vertically stacked
on the green marker. The FSM expert unstacks them top→bottom and places each
into the red rectangular region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CubeDef:
    """Single cube definition within the stack."""

    name: str  # e.g. "cube_large"
    scene_key: str  # Isaac Lab scene key (same as name)
    edge_length: float  # full edge length (m)
    color: tuple[float, float, float]  # RGB 0-1
    mass: float  # kg
    init_z: float  # centre z when stacked on table
    grasp_z_offset: float = 0.0  # z tweak when grasping
    release_z_offset: float = -0.03  # z tweak when releasing
    place_z: float = 0.055  # EE target z for GO_TO_PLACE
    goal_z: float = 0.025  # object centre z on table (half-edge)


@dataclass
class CubeStackConfig:
    """Full configuration for the 3-cube unstack task."""

    cubes: list[CubeDef]  # [large (bottom), medium, small (top)]

    # Scene layout
    goal_xy: tuple[float, float] = (0.5, -0.2)  # green marker (initial stack)
    red_region_center_xy: tuple[float, float] = (0.5, 0.2)  # red region centre
    red_region_size_xy: tuple[float, float] = (0.30, 0.30)  # red region dims

    # FSM parameters
    hover_z: float = 0.30
    position_threshold: float = 0.015
    wait_steps: int = 10
    success_radius: float = 0.03

    # Safety
    min_place_separation: float = 0.10  # min distance between placed cubes


# ---------------------------------------------------------------------------
# Cube definitions (bottom → top)
# ---------------------------------------------------------------------------

CUBE_LARGE = CubeDef(
    name="cube_large",
    scene_key="cube_large",
    edge_length=0.060,
    color=(1.0, 0.85, 0.0),  # yellow
    mass=0.12,
    init_z=0.030,  # half-edge on table
    grasp_z_offset=-0.01,  # 1cm below centre for more stable grasp
    release_z_offset=-0.03,
    place_z=0.060,
    goal_z=0.030,
)

CUBE_MEDIUM = CubeDef(
    name="cube_medium",
    scene_key="cube_medium",
    edge_length=0.050,
    color=(0.2, 0.4, 1.0),  # blue
    mass=0.10,
    init_z=0.085,  # sits on top of large: 0.060 + 0.025
    grasp_z_offset=-0.01,  # 1cm below centre for more stable grasp
    release_z_offset=-0.03,
    place_z=0.055,
    goal_z=0.025,
)

CUBE_SMALL = CubeDef(
    name="cube_small",
    scene_key="cube_small",
    edge_length=0.040,
    color=(0.6, 0.2, 0.8),  # purple
    mass=0.08,
    init_z=0.130,  # sits on top of medium: 0.110 + 0.020
    grasp_z_offset=-0.01,  # 1cm below centre for more stable grasp
    release_z_offset=-0.03,
    place_z=0.050,
    goal_z=0.020,
)

# Ordered bottom→top
CUBE_DEFS: list[CubeDef] = [CUBE_LARGE, CUBE_MEDIUM, CUBE_SMALL]

# Pick order: top→bottom
PICK_ORDER: list[str] = ["cube_small", "cube_medium", "cube_large"]

DEFAULT_STACK_CONFIG = CubeStackConfig(cubes=CUBE_DEFS)


# ---------------------------------------------------------------------------
# Spawn helpers (lazy imports)
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


def make_cuboid_spawn(cube_def: CubeDef):
    """Return a CuboidCfg spawn config for *cube_def*."""
    import isaaclab.sim as sim_utils

    e = cube_def.edge_length
    return sim_utils.CuboidCfg(
        size=(e, e, e),
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=cube_def.mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=cube_def.color,
        ),
    )


def get_cube_def(name: str) -> CubeDef:
    """Look up a CubeDef by name."""
    for c in CUBE_DEFS:
        if c.name == name:
            return c
    raise KeyError(f"Unknown cube name: {name!r}")
