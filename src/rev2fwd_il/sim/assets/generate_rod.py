#!/usr/bin/env python3
"""Generate a simple rod (cylinder) USD asset for peg-insertion experiments.

Rod: Ø8mm × 50mm cylinder, orange colour, origin at geometric centre.

Usage:
    python generate_rod.py              # writes rod.usda next to this script
    python generate_rod.py /tmp/rod.usda
"""
from __future__ import annotations

import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _create_material(stage, path, color, metallic=0.0, roughness=0.3):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def create_rod(out_path: str):
    radius = 0.004   # 4 mm
    height = 0.060   # 60 mm

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Rod")
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.GetMassAttr().Set(0.05)

    # Material — orange
    mat = _create_material(
        stage, "/Rod/Materials/RodMaterial",
        color=(1.0, 0.5, 0.0), metallic=0.1, roughness=0.4,
    )

    # Cylinder body — origin at geometric centre (z=0)
    cyl = UsdGeom.Cylinder.Define(stage, "/Rod/Body")
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    cyl.GetAxisAttr().Set("Z")
    cyl.GetExtentAttr().Set([
        Gf.Vec3f(-radius, -radius, -height / 2),
        Gf.Vec3f(radius, radius, height / 2),
    ])

    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(cyl.GetPrim())
    cyl.GetPrim().GetAttribute("physics:approximation").Set("convexHull")

    UsdShade.MaterialBindingAPI(cyl.GetPrim()).Bind(mat)

    # Physics material — high friction so gripper can hold tight
    phys_mat_shade = UsdShade.Material.Define(stage, "/Rod/Materials/RodPhysicsMaterial")
    phys_mat = UsdPhysics.MaterialAPI.Apply(phys_mat_shade.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(1.0)
    phys_mat.CreateDynamicFrictionAttr().Set(1.0)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    # Bind physics material to collision geometry
    UsdPhysics.MaterialAPI.Apply(cyl.GetPrim())
    binding_api = UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim())
    binding_api.Bind(
        phys_mat_shade,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    stage.GetRootLayer().Save()
    print(f"Rod USD saved to: {out_path}")


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rod.usda")
    out = sys.argv[1] if len(sys.argv) > 1 else default_path
    create_rod(out)
