#!/usr/bin/env python3
"""Generate a realistic bottle USD asset using OpenUSD (pxr) APIs.

The bottle consists of:
  - Body:  cylinder  (radius=2cm, height=6cm)  — lower portion
  - Neck:  cone-like taper from body to neck
  - Neck:  cylinder  (radius=0.8cm, height=2.5cm) — upper narrow part
  - Cap:   cylinder  (radius=1.0cm, height=0.8cm)  — cap on top

Total height ≈ 10cm. Origin at geometric centre of the body (z=0 at body
bottom).  This matches the existing CylinderCfg bottle dimensions.

The bottle has:
  - A glossy green opaque body material
  - A white cap material
  - Proper collision approximation (convexDecomposition)

Usage:
    python generate_bottle.py            # writes bottle.usda next to this script
    python generate_bottle.py /tmp/b.usda  # writes to custom path
"""
from __future__ import annotations

import math
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _add_cylinder(stage, path: str, radius: float, height: float, z_offset: float):
    """Add a cylinder prim centered at z_offset (bottom at z_offset - h/2)."""
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)
    cyl.GetAxisAttr().Set("Z")
    xform = UsdGeom.Xformable(cyl.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, z_offset))
    cyl.GetExtentAttr().Set(
        [Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)]
    )
    # Apply collision API to each geometry prim
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(cyl.GetPrim())
    cyl.GetPrim().GetAttribute("physics:approximation").Set("convexHull")
    return cyl


def _add_cone(stage, path: str, radius: float, height: float, z_offset: float):
    """Add a cone prim. Tip points +Z, base at -height/2."""
    cone = UsdGeom.Cone.Define(stage, path)
    cone.GetRadiusAttr().Set(radius)
    cone.GetHeightAttr().Set(height)
    cone.GetAxisAttr().Set("Z")
    xform = UsdGeom.Xformable(cone.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, z_offset))
    cone.GetExtentAttr().Set(
        [Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)]
    )
    return cone


def _ring_mesh(
    stage,
    path: str,
    bottom_r: float,
    top_r: float,
    height: float,
    z_offset: float,
    n_sides: int = 32,
):
    """Create a tapered ring (truncated-cone) mesh from bottom_r to top_r."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = []
    face_vertex_counts = []
    face_vertex_indices = []

    # Bottom ring + top ring
    for ring_idx, (r, z) in enumerate(
        [(bottom_r, -height / 2 + z_offset), (top_r, height / 2 + z_offset)]
    ):
        for i in range(n_sides):
            angle = 2 * math.pi * i / n_sides
            points.append(Gf.Vec3f(r * math.cos(angle), r * math.sin(angle), z))

    # Quads connecting bottom ring [0..n-1] to top ring [n..2n-1]
    for i in range(n_sides):
        i0 = i
        i1 = (i + 1) % n_sides
        i2 = n_sides + (i + 1) % n_sides
        i3 = n_sides + i
        face_vertex_counts.append(4)
        face_vertex_indices.extend([i0, i1, i2, i3])

    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    # Apply collision API to mesh prims
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().GetAttribute("physics:approximation").Set("convexHull")
    return mesh


def _create_material(stage, path: str, color: tuple, opacity: float = 1.0, metallic: float = 0.0, roughness: float = 0.3):
    """Create a simple UsdPreviewSurface material."""
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def create_bottle(out_path: str):
    """Create the bottle USD file."""
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # metres

    root = UsdGeom.Xform.Define(stage, "/Bottle")
    stage.SetDefaultPrim(root.GetPrim())

    # Apply RigidBody and Mass APIs to the root prim
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.GetMassAttr().Set(0.15)

    # ---- Dimensions (metres) ----
    body_r = 0.020       # 2 cm
    body_h = 0.060       # 6 cm
    taper_h = 0.015      # 1.5 cm taper from body to neck
    neck_r = 0.008       # 0.8 cm
    neck_h = 0.017       # 1.7 cm
    cap_r = 0.010        # 1.0 cm
    cap_h = 0.008        # 0.8 cm
    # Total: 6 + 1.5 + 1.7 + 0.8 = 10.0 cm  ✓

    # Shift all geometry so origin is at the geometric center (z=0.05),
    # matching the convention of CylinderCfg (origin at center).
    total_h = body_h + taper_h + neck_h + cap_h   # 0.10
    z_shift = -total_h / 2                         # -0.05

    body_z = body_h / 2 + z_shift                            # -0.020
    taper_z = body_h + taper_h / 2 + z_shift                 # 0.0175
    neck_z = body_h + taper_h + neck_h / 2 + z_shift         # 0.0335
    cap_z = body_h + taper_h + neck_h + cap_h / 2 + z_shift  # 0.046

    # ---- Materials ----
    body_mat = _create_material(
        stage, "/Bottle/Materials/BodyMaterial",
        color=(0.0, 0.45, 0.15),  # dark green glossy
        opacity=1.0,
        roughness=0.12,
        metallic=0.35,
    )
    cap_mat = _create_material(
        stage, "/Bottle/Materials/CapMaterial",
        color=(0.9, 0.9, 0.9),  # white cap
        opacity=1.0,
        roughness=0.5,
    )
    label_mat = _create_material(
        stage, "/Bottle/Materials/LabelMaterial",
        color=(0.85, 0.82, 0.65),  # beige label
        opacity=1.0,
        roughness=0.7,
    )

    # ---- Geometry ----
    # 1. Body cylinder
    body = _add_cylinder(stage, "/Bottle/Body", body_r, body_h, body_z)
    UsdShade.MaterialBindingAPI(body.GetPrim()).Bind(body_mat)

    # 2. Label ring (slightly larger radius to sit on body surface)
    label = _ring_mesh(
        stage, "/Bottle/Label",
        bottom_r=body_r + 0.0003,
        top_r=body_r + 0.0003,
        height=body_h * 0.45,
        z_offset=body_z * 0.95,
        n_sides=32,
    )
    UsdShade.MaterialBindingAPI(label.GetPrim()).Bind(label_mat)

    # 3. Taper (truncated cone) from body_r → neck_r
    taper = _ring_mesh(
        stage, "/Bottle/Taper",
        bottom_r=body_r,
        top_r=neck_r,
        height=taper_h,
        z_offset=taper_z,
        n_sides=32,
    )
    UsdShade.MaterialBindingAPI(taper.GetPrim()).Bind(body_mat)

    # 4. Neck cylinder
    neck = _add_cylinder(stage, "/Bottle/Neck", neck_r, neck_h, neck_z)
    UsdShade.MaterialBindingAPI(neck.GetPrim()).Bind(body_mat)

    # 5. Cap cylinder
    cap = _add_cylinder(stage, "/Bottle/Cap", cap_r, cap_h, cap_z)
    UsdShade.MaterialBindingAPI(cap.GetPrim()).Bind(cap_mat)

    stage.GetRootLayer().Save()
    print(f"Bottle USD saved to: {out_path}")


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bottle.usda")
    out = sys.argv[1] if len(sys.argv) > 1 else default_path
    create_bottle(out)
