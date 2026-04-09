"""Object registry for multi-object pick-and-place experiments.

Each entry maps an object name to its spawning configuration, physical properties,
and grasping parameters needed by the FSM expert. This registry is consumed by
both the environment configuration and the data-collection scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObjectConfig:
    """Configuration for a single graspable object type.

    Attributes:
        spawn_cfg_fn: A callable that returns the spawn configuration.
            Called lazily so that ``isaaclab.sim`` imports happen only inside
            Isaac Sim runtime.
        init_pos: Initial object position [x, y, z] in local env frame.
        init_rot: Initial object rotation [qw, qx, qy, qz].
        object_height: Approximate height of the object (used for grasp z calc).
        grasp_z_offset: Z offset from object surface when grasping.
        release_z_offset: Z offset for release (negative = lower).
        hover_z: Hover height above table surface.
        position_threshold: Distance threshold for FSM waypoint convergence.
        wait_steps: Wait steps between FSM waypoints.
        success_radius: XY distance for success check.
        description: Human-readable description.
    """

    spawn_cfg_fn: Any  # callable returning SpawnCfg
    init_pos: tuple[float, float, float] = (0.5, 0.0, 0.055)
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_height: float = 0.05
    grasp_z_offset: float = 0.0
    release_z_offset: float = -0.04
    hover_z: float = 0.25
    position_threshold: float = 0.015
    wait_steps: int = 10
    success_radius: float = 0.03
    place_z: float = 0.055       # EE target z for GO_TO_PLACE state
    goal_z: float = 0.055        # Object centre z when resting on table (teleport z)
    mesh_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Offset from prim root to mesh geometric centre (x, y, z).
    # For USD-based objects whose prim origin ≠ mesh centre.
    # Applied to object_pose before the FSM expert uses it.
    description: str = ""


# ---------------------------------------------------------------------------
# Spawn-config factory functions  (lazy, so sim_utils is only imported at
# runtime inside Isaac Sim)
# ---------------------------------------------------------------------------

def _make_cube_spawn():
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    return UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.8, 0.8, 0.8),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )


def _rigid_props():
    """Shared rigid body properties for primitive shapes."""
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )


def _make_cylinder_spawn():
    import isaaclab.sim as sim_utils

    return sim_utils.CylinderCfg(
        radius=0.02,
        height=0.06,
        axis="Z",
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.6, 1.0),  # light blue
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
            friction_combine_mode="max",
            compliant_contact_stiffness=1e5,
            compliant_contact_damping=1e3,
        ),
    )


def _make_sphere_spawn():
    import isaaclab.sim as sim_utils

    return sim_utils.SphereCfg(
        radius=0.025,
        rigid_props=_rigid_props(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.5, 0.0),  # orange
        ),
    )


def _make_bottle_spawn():
    """Bottle from a local USD asset (body + taper + neck + cap)."""
    import os

    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    import isaaclab.sim as sim_utils

    usd_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "bottle.usda"
    )
    return UsdFileCfg(
        usd_path=usd_path,
        scale=(1.0, 1.0, 1.0),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _make_gear_spawn():
    """Factory gear medium USD at 2x scale (d=6cm, h=6cm).

    The USD prim origin is NOT at the geometric center — it sits ~8mm below
    the mesh bottom at 1x (~13mm at 2x).  The grasp_z_offset in the
    ObjectConfig compensates for this so the gripper reaches the mesh centre.
    """
    from isaaclab.sim.schemas.schemas_cfg import (
        ArticulationRootPropertiesCfg,
        RigidBodyPropertiesCfg,
    )
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    import isaaclab.sim as sim_utils

    return UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_medium.usd",
        scale=(2.0, 2.0, 2.0),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        articulation_props=ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
    )


# ---------------------------------------------------------------------------
# Object registry
# ---------------------------------------------------------------------------

OBJECT_REGISTRY: dict[str, ObjectConfig] = {
    "cube": ObjectConfig(
        spawn_cfg_fn=_make_cube_spawn,
        init_pos=(0.5, 0.0, 0.055),
        object_height=0.05,
        grasp_z_offset=0.0,
        release_z_offset=-0.04,
        hover_z=0.25,
        position_threshold=0.015,
        wait_steps=10,
        success_radius=0.03,
        place_z=0.055,
        goal_z=0.055,
        description="DexCube 0.8x scale (~4cm edge)",
    ),
    "cylinder": ObjectConfig(
        spawn_cfg_fn=_make_cylinder_spawn,
        init_pos=(0.5, 0.0, 0.030),  # half height = 3cm
        object_height=0.06,
        grasp_z_offset=-0.01,  # lowered 2cm (was 0.01) — grasp below centre to prevent slip
        release_z_offset=-0.03,  # release_z = 0.065-0.03 = 0.035 (safe)
        hover_z=0.25,
        position_threshold=0.015,
        wait_steps=10,
        success_radius=0.03,
        place_z=0.065,   # higher than cube to avoid table collision
        goal_z=0.030,    # cylinder centre when resting on table
        description="Cylinder r=2cm h=6cm (light blue)",
    ),
    "sphere": ObjectConfig(
        spawn_cfg_fn=_make_sphere_spawn,
        init_pos=(0.5, 0.0, 0.025),  # radius = 2.5cm
        object_height=0.05,
        grasp_z_offset=-0.005,  # grasp slightly below center
        release_z_offset=-0.025,  # release_z = 0.055-0.025 = 0.030 (safe)
        hover_z=0.25,
        position_threshold=0.020,  # relaxed — sphere may drift slightly
        wait_steps=15,  # extra settle for rolling
        success_radius=0.04,  # relaxed — sphere can roll
        place_z=0.055,
        goal_z=0.025,    # sphere centre = radius when resting on table
        description="Sphere r=2.5cm (orange)",
    ),
    "bottle": ObjectConfig(
        spawn_cfg_fn=_make_bottle_spawn,
        init_pos=(0.5, 0.0, 0.050),  # half height = 5cm
        object_height=0.10,
        grasp_z_offset=0.01,  # lowered 1cm (was 0.02) — grasp lower on body to prevent slip
        release_z_offset=-0.02,  # release_z = 0.10-0.02 = 0.080 (bottle bottom ~0.03, safe)
        hover_z=0.28,  # higher hover for tall object
        position_threshold=0.015,
        wait_steps=10,
        success_radius=0.03,
        place_z=0.10,    # much higher — bottle is 10cm tall
        goal_z=0.050,    # bottle centre when resting on table
        description="Bottle USD: body+taper+neck+cap h=10cm (green glass)",
    ),
    "gear": ObjectConfig(
        spawn_cfg_fn=_make_gear_spawn,
        # Prim origin offset: at 2x scale the prim root rests at z≈-0.013 when
        # the mesh bottom touches the table.  Mesh height ≈ 0.05m (2x),
        # mesh AABB centre is offset (dx=+0.0405, dy=0, dz=+0.035) from prim root.
        init_pos=(0.5, 0.0, 0.030),   # prim slightly above table → falls & settles
        object_height=0.050,           # AABB height at 2x scale
        grasp_z_offset=0.01,           # slightly above corrected mesh centre
        release_z_offset=-0.02,
        hover_z=0.25,
        position_threshold=0.015,
        wait_steps=10,
        success_radius=0.03,
        place_z=0.055,
        goal_z=0.030,
        mesh_origin_offset=(0.0405, 0.0, 0.035),  # measured from AABB at 2x
        description="Factory gear medium 2x (d≈8cm, mesh offset dx=4cm dz=3.5cm)",
    ),
}


def get_object_config(name: str) -> ObjectConfig:
    """Get object config by name. Raises KeyError if not found."""
    if name not in OBJECT_REGISTRY:
        valid = ", ".join(OBJECT_REGISTRY.keys())
        raise KeyError(f"Unknown object type '{name}'. Valid types: {valid}")
    return OBJECT_REGISTRY[name]


def list_object_types() -> list[str]:
    """Return list of registered object type names."""
    return list(OBJECT_REGISTRY.keys())
