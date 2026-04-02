#!/usr/bin/env python3
"""Measure factory gear mesh bounding-box centre relative to the prim origin.

Spawns the gear, disables the position-randomization event, and reads:
1. The rigid body root_pos (= prim origin as reported by Isaac Lab)
2. The mesh AABB (axis-aligned bounding box) computed from USD prims

The difference = the offset the FSM must compensate for.
"""
from __future__ import annotations
import argparse, sys

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gear", type=str, default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--scale", type=float, default=2.0)
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

    import torch, numpy as np
    import gymnasium as gym
    import isaaclab_tasks  # noqa
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
    from isaaclab.sim.schemas.schemas_cfg import (
        ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg,
    )
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    import isaaclab.sim as sim_utils
    import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

    sf = args.scale
    gear_usd = f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_gear_{args.gear}.usd"
    spawn_pos = [0.5, 0.0, 0.10]  # spawn high

    env_cfg = parse_env_cfg("Isaac-Lift-Cube-Franka-IK-Abs-v0",
                            device="cuda:0", num_envs=1, use_fabric=False)

    env_cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=spawn_pos, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=gear_usd,
            scale=(sf, sf, sf),
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
                articulation_enabled=False),
        ),
    )

    # Disable position randomization — we want to read exact spawn position
    env_cfg.events.reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    env_cfg.episode_length_s = 100.0
    if hasattr(env_cfg, 'terminations'):
        if hasattr(env_cfg.terminations, 'time_out'):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, 'object_dropping'):
            env_cfg.terminations.object_dropping = None

    # use_fabric=False so we can read USD stage
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    # --- 1. Read prim-origin position (rigid body root) ---
    obj = env.unwrapped.scene["object"]
    origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
    root_pos_w = obj.data.root_pos_w[0].cpu().numpy()
    root_pos = root_pos_w - origin
    print(f"\n{'='*60}", flush=True)
    print(f"Gear: {args.gear}  scale: {sf}x", flush=True)
    print(f"Spawn pos:  ({spawn_pos[0]:.4f}, {spawn_pos[1]:.4f}, {spawn_pos[2]:.4f})", flush=True)
    print(f"Root pos:   ({root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f})", flush=True)
    print(f"Root offset from spawn: ({root_pos[0]-spawn_pos[0]:.4f}, "
          f"{root_pos[1]-spawn_pos[1]:.4f}, {root_pos[2]-spawn_pos[2]:.4f})", flush=True)

    # --- 2. Read mesh AABB from USD stage ---
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()

        obj_prim_path = "/World/envs/env_0/Object"
        obj_prim = stage.GetPrimAtPath(obj_prim_path)
        if not obj_prim.IsValid():
            # Try finding it
            for p in stage.Traverse():
                if "Object" in str(p.GetPath()):
                    print(f"  Found prim: {p.GetPath()}", flush=True)

        # Compute world-space bounding box
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(obj_prim)
        bbox_range = bbox.ComputeAlignedRange()
        bb_min = bbox_range.GetMin()
        bb_max = bbox_range.GetMax()
        bb_center = (Gf.Vec3d(bb_min) + Gf.Vec3d(bb_max)) / 2.0
        bb_size = Gf.Vec3d(bb_max) - Gf.Vec3d(bb_min)

        bb_center_local = np.array([bb_center[0], bb_center[1], bb_center[2]]) - origin
        print(f"\nMesh AABB (world, relative to env_origin):", flush=True)
        print(f"  min:    ({bb_min[0]-origin[0]:.4f}, {bb_min[1]-origin[1]:.4f}, {bb_min[2]-origin[2]:.4f})", flush=True)
        print(f"  max:    ({bb_max[0]-origin[0]:.4f}, {bb_max[1]-origin[1]:.4f}, {bb_max[2]-origin[2]:.4f})", flush=True)
        print(f"  center: ({bb_center_local[0]:.4f}, {bb_center_local[1]:.4f}, {bb_center_local[2]:.4f})", flush=True)
        print(f"  size:   ({bb_size[0]:.4f}, {bb_size[1]:.4f}, {bb_size[2]:.4f})", flush=True)

        mesh_offset = bb_center_local - root_pos
        print(f"\nMesh centre OFFSET from prim root:", flush=True)
        print(f"  dx={mesh_offset[0]:.4f}  dy={mesh_offset[1]:.4f}  dz={mesh_offset[2]:.4f}", flush=True)
        print(f"{'='*60}\n", flush=True)

    except Exception as e:
        print(f"USD introspection failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    # --- 3. Let it fall, then re-measure ---
    device = env.unwrapped.device
    action = torch.zeros(1, env.action_space.shape[-1], device=device)
    ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
    action[:, :3] = ee[:, :3] - env.unwrapped.scene.env_origins[:, :3]
    action[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    action[:, 7] = 1.0
    for _ in range(300):
        env.step(action)

    root_pos_after = obj.data.root_pos_w[0].cpu().numpy() - origin
    print(f"After settling on table:", flush=True)
    print(f"  Root pos: ({root_pos_after[0]:.4f}, {root_pos_after[1]:.4f}, {root_pos_after[2]:.4f})", flush=True)

    try:
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
        bbox_cache.Clear()
        bbox = bbox_cache.ComputeWorldBound(obj_prim)
        bbox_range = bbox.ComputeAlignedRange()
        bb_min2 = bbox_range.GetMin()
        bb_max2 = bbox_range.GetMax()
        bb_center2 = (Gf.Vec3d(bb_min2) + Gf.Vec3d(bb_max2)) / 2.0
        bb_center2_local = np.array([bb_center2[0], bb_center2[1], bb_center2[2]]) - origin
        mesh_offset2 = bb_center2_local - root_pos_after
        print(f"  Mesh AABB center: ({bb_center2_local[0]:.4f}, {bb_center2_local[1]:.4f}, {bb_center2_local[2]:.4f})", flush=True)
        print(f"  Mesh offset from root: dx={mesh_offset2[0]:.4f}  dy={mesh_offset2[1]:.4f}  dz={mesh_offset2[2]:.4f}", flush=True)
    except Exception as e:
        print(f"Post-settle USD introspection failed: {e}", flush=True)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
