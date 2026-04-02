#!/usr/bin/env python3
"""Quick test to measure factory gear USD origin offset.

Spawns each gear size at a known position and measures the reported obj_pose
vs. expected position to determine the prim origin offset from geometric center.
"""
from __future__ import annotations
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args

def main():
    args = _parse_args()
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    import isaaclab.sim as sim_utils

    rigid_props = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    gear_configs = {
        "small": f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_small.usd",
        "medium": f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_medium.usd",
        "large": f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_large.usd",
    }

    for scale_factor in [1.0, 2.0, 3.0]:
        for gear_name, usd_path in gear_configs.items():
            spawn_z = 0.10  # high enough to NOT collide with table
            init_pos = (0.5, 0.0, spawn_z)

            env_cfg = parse_env_cfg(
                "Isaac-Lift-Cube-Franka-IK-Abs-v0",
                device="cuda:0", num_envs=1, use_fabric=True,
            )
            env_cfg.scene.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=list(init_pos), rot=[1.0, 0.0, 0.0, 0.0],
                ),
                spawn=UsdFileCfg(
                    usd_path=usd_path,
                    scale=(scale_factor, scale_factor, scale_factor),
                    rigid_props=rigid_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    articulation_props=ArticulationRootPropertiesCfg(
                        articulation_enabled=False,
                    ),
                ),
            )
            env_cfg.episode_length_s = 100.0
            if hasattr(env_cfg, 'terminations'):
                if hasattr(env_cfg.terminations, 'time_out'):
                    env_cfg.terminations.time_out = None
                if hasattr(env_cfg.terminations, 'object_dropping'):
                    env_cfg.terminations.object_dropping = None

            env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
            env.reset()

            # Step a few times to let physics settle
            device = env.unwrapped.device
            action = torch.zeros(1, env.action_space.shape[-1], device=device)
            ee_pose = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
            action[:, :3] = ee_pose[:, :3] - env.unwrapped.scene.env_origins[:, :3]
            action[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
            action[:, 7] = 1.0
            for _ in range(50):
                env.step(action)

            # Read object pose
            obj = env.unwrapped.scene["object"]
            obj_pos_w = obj.data.root_pos_w[0].cpu().numpy()
            env_origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
            obj_pos_local = obj_pos_w - env_origin

            offset_z = obj_pos_local[2] - spawn_z
            print(f"[{gear_name}] scale={scale_factor:.0f}x | "
                  f"init_z={spawn_z:.3f} | "
                  f"obj_z={obj_pos_local[2]:.4f} | "
                  f"offset_z={offset_z:.4f} | "
                  f"obj_xy=({obj_pos_local[0]:.4f}, {obj_pos_local[1]:.4f})")

            env.close()

    simulation_app.close()

if __name__ == "__main__":
    main()
