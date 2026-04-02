# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Multi-object IK-Abs environment configurations for Franka.
#
# These configs replace the DexCube spawn in the base Franka lift env with
# alternative objects from the object registry.  They inherit everything else
# (robot, IK controller, cameras, rewards, etc.) from the existing cube config.

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import ik_abs_env_cfg


# ---------------------------------------------------------------------------
# Helper: build a new env config class for a given object type
# ---------------------------------------------------------------------------

def _make_env_cfg_class(object_type: str, *, usd_object: bool = False):
    """Dynamically create an env config class that uses the given object type.

    Args:
        object_type: Registry key (e.g. "cylinder", "gear").
        usd_object: If True, the object is spawned from a USD file whose
            internal body names may differ from the default "Object".
            We override ``reset_object_position`` to drop the ``body_names``
            filter so the regex matcher doesn't fail.
    """
    from rev2fwd_il.sim.object_registry import get_object_config

    obj_cfg = get_object_config(object_type)

    @configclass
    class _Cfg(ik_abs_env_cfg.FrankaCubeLiftEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=list(obj_cfg.init_pos),
                    rot=list(obj_cfg.init_rot),
                ),
                spawn=obj_cfg.spawn_cfg_fn(),
            )
            if usd_object:
                # USD files may have internal body names that don't match
                # the default "Object" filter → use SceneEntityCfg without
                # body_names so IsaacLab doesn't fail on regex matching.
                self.events.reset_object_position = EventTerm(
                    func=mdp.reset_root_state_uniform,
                    mode="reset",
                    params={
                        "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                        "velocity_range": {},
                        "asset_cfg": SceneEntityCfg("object"),
                    },
                )

    _Cfg.__name__ = f"Franka{object_type.capitalize()}LiftEnvCfg"
    _Cfg.__qualname__ = _Cfg.__name__
    return _Cfg


# ---------------------------------------------------------------------------
# Concrete classes (one per object type, excluding cube which already exists)
# ---------------------------------------------------------------------------

FrankaCylinderLiftEnvCfg = _make_env_cfg_class("cylinder")
FrankaSphereLiftEnvCfg = _make_env_cfg_class("sphere")
FrankaBottleLiftEnvCfg = _make_env_cfg_class("bottle")
FrankaGearLiftEnvCfg = _make_env_cfg_class("gear", usd_object=True)
