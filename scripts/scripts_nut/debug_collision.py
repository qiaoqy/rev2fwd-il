#!/usr/bin/env python3
"""Debug collision meshes for nut threading task.

Inspects contact_offset, rest_offset, and mesh collision approximation type
for all collision bodies in the nut threading environment.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/scripts_nut/debug_collision.py --headless --disable_fabric 1
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--disable_fabric", type=int, default=1)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: register envs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Trace where the module files are loaded from
import isaaclab_tasks.direct.factory.factory_tasks_cfg as ftcfg
import isaaclab_tasks.direct.factory.factory_env_cfg as fecfg
print(f"\n=== Module Locations ===", flush=True)
print(f"  factory_tasks_cfg: {ftcfg.__file__}", flush=True)
print(f"  factory_env_cfg: {fecfg.__file__}", flush=True)

# Check the raw class values
from isaaclab_tasks.direct.factory.factory_tasks_cfg import NutThread
nt = NutThread()
print(f"  NutThread.fixed_asset.spawn.collision_props.contact_offset = {nt.fixed_asset.spawn.collision_props.contact_offset}", flush=True)
print(f"  NutThread.held_asset.spawn.collision_props.contact_offset = {nt.held_asset.spawn.collision_props.contact_offset}", flush=True)
try:
    # Try multiple possible attribute names for the robot config
    for attr in ['robot', 'franka', 'panda']:
        if hasattr(fecfg.FactoryEnvCfg, attr):
            cfg = getattr(fecfg.FactoryEnvCfg, attr)
            print(f"  FactoryEnvCfg.{attr}.spawn.collision_props.contact_offset = {cfg.spawn.collision_props.contact_offset}", flush=True)
            break
    else:
        # List available attributes
        attrs = [a for a in dir(fecfg.FactoryEnvCfg) if not a.startswith('_')]
        print(f"  FactoryEnvCfg attributes: {attrs}", flush=True)
except Exception as e:
    print(f"  Robot config check error: {e}", flush=True)

import gymnasium as gym

env_cfg = parse_env_cfg("Isaac-Forge-NutThread-Direct-v0", device="cuda:0", num_envs=1, use_fabric=not bool(args.disable_fabric))
env = gym.make("Isaac-Forge-NutThread-Direct-v0", cfg=env_cfg)
forge_env = env.unwrapped

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

stage = sim_utils.SimulationContext.instance().stage

print("\n=== Inspecting Collision Properties ===", flush=True)

for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "env_0" not in path and "envs" in path:
        continue

    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        continue

    contact_offset = None
    rest_offset = None
    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        physx_col = PhysxSchema.PhysxCollisionAPI(prim)
        co = physx_col.GetContactOffsetAttr()
        ro = physx_col.GetRestOffsetAttr()
        if co:
            contact_offset = co.Get()
        if ro:
            rest_offset = ro.Get()

    mesh_type = "none"
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        approx = mesh_api.GetApproximationAttr()
        if approx:
            mesh_type = approx.Get()

    if prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
        mesh_type = "SDF"
    elif prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
        mesh_type = "convexDecomposition"
    elif prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI):
        mesh_type = "convexHull"
    elif prim.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI):
        mesh_type = "triangleMesh"

    if any(name in path for name in ["Robot", "HeldAsset", "FixedAsset"]):
        print(f"[COLLISION] {path}", flush=True)
        print(f"  contact_offset={contact_offset}, rest_offset={rest_offset}, mesh_type={mesh_type}", flush=True)

print("\n=== Checking Geometry Extents ===", flush=True)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "env_0" not in path and "envs" in path:
        continue

    if not any(name in path for name in ["FixedAsset", "HeldAsset"]):
        continue

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        extent = mesh.GetExtentAttr().Get()
        points = mesh.GetPointsAttr().Get()
        if extent:
            print(f"[MESH] {path}", flush=True)
            print(f"  extent={extent}", flush=True)
            if points:
                print(f"  num_points={len(points)}", flush=True)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                print(f"  HAS COLLISION API", flush=True)

print("\n=== Config Values (from Python config) ===", flush=True)
print(f"  Robot collision_props: {forge_env.cfg.robot.spawn.collision_props}", flush=True)
print(f"  FixedAsset collision_props: {forge_env.cfg_task.fixed_asset.spawn.collision_props}", flush=True)
print(f"  HeldAsset collision_props: {forge_env.cfg_task.held_asset.spawn.collision_props}", flush=True)

print("\n=== Nut (HeldAsset) dimensions ===", flush=True)
print(f"  NutM16 diameter={forge_env.cfg_task.held_asset_cfg.diameter}m = {forge_env.cfg_task.held_asset_cfg.diameter*1000}mm", flush=True)
print(f"  NutM16 height={forge_env.cfg_task.held_asset_cfg.height}m = {forge_env.cfg_task.held_asset_cfg.height*1000}mm", flush=True)

print("\n=== Bolt (FixedAsset) dimensions ===", flush=True)
print(f"  BoltM16 diameter={forge_env.cfg_task.fixed_asset_cfg.diameter}m = {forge_env.cfg_task.fixed_asset_cfg.diameter*1000}mm", flush=True)
print(f"  BoltM16 height={forge_env.cfg_task.fixed_asset_cfg.height}m = {forge_env.cfg_task.fixed_asset_cfg.height*1000}mm", flush=True)
print(f"  BoltM16 base_height={forge_env.cfg_task.fixed_asset_cfg.base_height}m = {forge_env.cfg_task.fixed_asset_cfg.base_height*1000}mm", flush=True)
print(f"  BoltM16 thread_pitch={forge_env.cfg_task.fixed_asset_cfg.thread_pitch}m = {forge_env.cfg_task.fixed_asset_cfg.thread_pitch*1000}mm", flush=True)

robot_co = forge_env.cfg.robot.spawn.collision_props.contact_offset
fixed_co = forge_env.cfg_task.fixed_asset.spawn.collision_props.contact_offset
held_co = forge_env.cfg_task.held_asset.spawn.collision_props.contact_offset
print("\n=== Summary ===", flush=True)
print(f"Robot contact_offset={robot_co}m ({robot_co*1000}mm)", flush=True)
print(f"Bolt  contact_offset={fixed_co}m ({fixed_co*1000}mm)", flush=True)
print(f"Nut   contact_offset={held_co}m ({held_co*1000}mm)", flush=True)
print(f"Gripper<->Nut gap threshold: {(robot_co+held_co)*1000}mm", flush=True)
print(f"Nut<->Bolt gap threshold: {(held_co+fixed_co)*1000}mm", flush=True)

env.close()
simulation_app.close()
