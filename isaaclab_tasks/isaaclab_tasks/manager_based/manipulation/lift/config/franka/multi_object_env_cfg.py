# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Multi-object IK-Abs environment configurations for Franka.
#
# These configs replace the DexCube spawn in the base Franka lift env with
# alternative objects from the object registry.  They inherit everything else
# (robot, IK controller, cameras, rewards, etc.) from the existing cube config.

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from . import ik_abs_env_cfg


# ---------------------------------------------------------------------------
# Helper: build a new env config class for a given object type
# ---------------------------------------------------------------------------

def _make_env_cfg_class(object_type: str):
    """Dynamically create an env config class that uses the given object type."""
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

    _Cfg.__name__ = f"Franka{object_type.capitalize()}LiftEnvCfg"
    _Cfg.__qualname__ = _Cfg.__name__
    return _Cfg


# ---------------------------------------------------------------------------
# Concrete classes (one per object type, excluding cube which already exists)
# ---------------------------------------------------------------------------

FrankaCylinderLiftEnvCfg = _make_env_cfg_class("cylinder")
FrankaSphereLiftEnvCfg = _make_env_cfg_class("sphere")
FrankaBottleLiftEnvCfg = _make_env_cfg_class("bottle")
FrankaGearLiftEnvCfg = _make_env_cfg_class("gear")
