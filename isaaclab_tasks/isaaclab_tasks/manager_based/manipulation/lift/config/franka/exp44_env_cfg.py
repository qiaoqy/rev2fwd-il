# Exp44: 3-cube partial-unstack environment configuration.
#
# Inherits from the Franka IK-Abs cube lift env, replaces the single
# ``scene.object`` with three coloured cuboids stacked at a fixed position,
# and disables reward / observation terms that reference the removed "object".
#
# Cube sizes: large=8cm blue, medium=6cm yellow, small=4cm red.
# No table markers.

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka import ik_abs_env_cfg

from rev2fwd_il.sim.exp44_registry import (
    CUBE_DEFS_44,
    DEFAULT_EXP44_CONFIG,
    make_cuboid_spawn_44,
)


@configclass
class FrankaExp44EnvCfg(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
    """Isaac Lab env config with 3 stacked cubes (Exp44 partial unstack)."""

    def __post_init__(self):
        # --- parent sets robot, IK controller, etc. ---
        super().__post_init__()

        cfg = DEFAULT_EXP44_CONFIG
        sx, sy = cfg.stack_xy

        # ---- Replace the single object with a tiny invisible dummy ----
        import isaaclab.sim as sim_utils

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.0, -2.0], rot=[1, 0, 0, 0]
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),
                rigid_props=sim_utils.schemas.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visible=False,
            ),
        )

        # ---- Add 3 coloured cubes (stacked at stack_xy) ----
        for cube_def in CUBE_DEFS_44:
            rigid_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{cube_def.scene_key}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=[sx, sy, cube_def.init_z],
                    rot=[1, 0, 0, 0],
                ),
                spawn=make_cuboid_spawn_44(cube_def),
            )
            setattr(self.scene, cube_def.scene_key, rigid_cfg)

        # ---- Disable components that reference "object" ----
        self.rewards = None
        self.commands = None
        self.curriculum = None

        if hasattr(self.observations, "policy"):
            if hasattr(self.observations.policy, "target_object_position"):
                self.observations.policy.target_object_position = None

        if hasattr(self.terminations, "object_dropping"):
            self.terminations.object_dropping = None

        if hasattr(self.events, "reset_object_position"):
            self.events.reset_object_position = None

        # Long episode for 2 rounds of pick-place
        self.episode_length_s = 200.0
