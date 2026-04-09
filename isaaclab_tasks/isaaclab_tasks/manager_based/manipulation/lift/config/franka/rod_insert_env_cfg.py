# Rod-Insert environment configuration for Exp39.
#
# Inherits the Franka IK-Abs lift environment and adds:
#   - Rod (object): thin cylinder held by the gripper
#   - Block with hole: fixed on the table, rod must be inserted into its hole
#
# The gripper stays closed throughout. Task A = insert, Task B = extract.

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from . import ik_abs_env_cfg

# Path to custom USD assets
# From: isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/
# Up 7 levels → workspace root (rev2fwd-il/)
# Then: src/rev2fwd_il/sim/assets/
_ASSET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..", "..", "..",  # back to workspace root
    "src", "rev2fwd_il", "sim", "assets",
)
# Resolve to absolute path
_ASSET_DIR = os.path.normpath(_ASSET_DIR)


def _rod_usd_path():
    return os.path.join(_ASSET_DIR, "rod.usda")


def _block_usd_path():
    return os.path.join(_ASSET_DIR, "block_with_hole.usda")


@configclass
class FrankaRodInsertEnvCfg(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
    """Rod-insert environment: Franka holds a rod above a block with a hole.

    The rod is the ``scene.object`` (tracked by observations and rewards).
    The block is an additional scene entity (``scene.block``).
    """

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot finger init: barely OPEN ----
        # Gap = 2 × 0.006 = 12 mm, just wider than the 8 mm rod.
        # The collection script teleports the rod between the pads and
        # then commands close to grip.
        self.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.006

        # ---- Rod (held object) ----
        # Spawns high up (z=0.5) out of the way; the collection script
        # teleports it to the EE position after the settle phase.
        # disable_gravity=True so it stays in place until teleported.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.5],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=_rod_usd_path(),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=100.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )

        # ---- Block with hole (fixed on table, kinematic) ----
        # Block height = 40mm, half = 20mm.  Block centre z = 0.020 (bottom at z=0 = table).
        # kinematic_enabled=True: block never moves regardless of forces.
        # This also enables meshSimplification collision in the USD file.
        self.scene.block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.020],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=_block_usd_path(),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=0.0,
                    max_linear_velocity=0.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )

        # ---- Override object reset to not randomize position ----
        # Rod should always start above the hole centre.
        # No XY randomisation — deterministic init.
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # Wider env spacing to avoid inter-env collisions
        self.scene.env_spacing = 5.0
