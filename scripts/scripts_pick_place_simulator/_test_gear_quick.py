#!/usr/bin/env python3
"""Quick test: spawn factory gear models and print obj_pose to measure origin offset."""
from __future__ import annotations
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gear", type=str, default="all", choices=["small","medium","large","all"])
    parser.add_argument("--scale", type=float, default=1.0)
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

    print(f"ISAACLAB_NUCLEUS_DIR = {ISAACLAB_NUCLEUS_DIR}")

    rigid_props = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16, solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0, max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0, disable_gravity=False,
    )

    gear_usd = {
        "small": f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_small.usd",
        "medium": f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_medium.usd",
        "large": f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_large.usd",
    }
    gears = [args.gear] if args.gear != "all" else ["small", "medium", "large"]
    sf = args.scale

    for gname in gears:
        spawn_z = 0.15  # high spawn so gear doesn't touch table initially
        env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0",
                                device="cuda:0", num_envs=1, use_fabric=True)
        env_cfg.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, spawn_z], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=gear_usd[gname],
                scale=(sf, sf, sf),
                rigid_props=rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
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
        device = env.unwrapped.device

        # Immediately read obj_pose before physics drops it
        obj = env.unwrapped.scene["object"]
        pos0_w = obj.data.root_pos_w[0].cpu().numpy()
        origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
        pos0 = pos0_w - origin
        print(f"[{gname}] scale={sf:.0f}x | init_z={spawn_z:.3f} | "
              f"BEFORE physics: obj_xyz=({pos0[0]:.4f}, {pos0[1]:.4f}, {pos0[2]:.4f}) | "
              f"offset_z={pos0[2]-spawn_z:.4f}", flush=True)

        # Step physics to let it fall and settle on table
        action = torch.zeros(1, env.action_space.shape[-1], device=device)
        ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
        action[:, :3] = ee[:, :3] - env.unwrapped.scene.env_origins[:, :3]
        action[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
        action[:, 7] = 1.0
        for _ in range(200):
            env.step(action)

        pos1_w = obj.data.root_pos_w[0].cpu().numpy()
        pos1 = pos1_w - origin
        print(f"[{gname}] scale={sf:.0f}x | AFTER settling on table: "
              f"obj_xyz=({pos1[0]:.4f}, {pos1[1]:.4f}, {pos1[2]:.4f})", flush=True)
        env.close()

    simulation_app.close()

if __name__ == "__main__":
    main()
