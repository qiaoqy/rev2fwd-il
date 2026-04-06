# Exp38: 3-cube unstack environment configuration.
#
# Inherits from the Franka IK-Abs cube lift env, replaces the single
# ``scene.object`` with three coloured cuboids stacked at the goal position,
# and disables reward / observation terms that reference the removed "object".

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka import ik_abs_env_cfg

from rev2fwd_il.sim.cube_stack_registry import (
    CUBE_DEFS,
    DEFAULT_STACK_CONFIG,
    make_cuboid_spawn,
)


@configclass
class FrankaCubeStackEnvCfg(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
    """Isaac Lab env config with 3 stacked cubes instead of a single object."""

    def __post_init__(self):
        # --- parent sets robot, IK controller, etc. ---
        super().__post_init__()

        cfg = DEFAULT_STACK_CONFIG
        gx, gy = cfg.goal_xy

        # ---- Replace the single object with 3 cubes ----
        # We keep scene.object as a tiny dummy so that any framework-level
        # code that iterates scene entities doesn't KeyError.  We put it
        # far below the table where it is invisible and harmless.
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

        # ---- Add 3 coloured cubes ----
        for cube_def in CUBE_DEFS:
            rigid_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{cube_def.scene_key}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=[gx, gy, cube_def.init_z],
                    rot=[1, 0, 0, 0],
                ),
                spawn=make_cuboid_spawn(cube_def),
            )
            setattr(self.scene, cube_def.scene_key, rigid_cfg)

        # ---- Disable components that reference "object" ----
        # Rewards, commands, and curriculum are not needed for FSM collection.
        self.rewards = None
        self.commands = None
        self.curriculum = None

        # Observation: remove target_object_position which references "object_pose" command
        if hasattr(self.observations, "policy"):
            if hasattr(self.observations.policy, "target_object_position"):
                self.observations.policy.target_object_position = None

        # Terminations: keep time_out, drop object_dropping (references "object")
        if hasattr(self.terminations, "object_dropping"):
            self.terminations.object_dropping = None

        # Events: keep reset_all, but drop reset_object_position (references "object")
        if hasattr(self.events, "reset_object_position"):
            self.events.reset_object_position = None

        # Long episode for 3 rounds of pick-place
        self.episode_length_s = 200.0
