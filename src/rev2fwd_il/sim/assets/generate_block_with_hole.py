#!/usr/bin/env python3
"""Generate a block-with-hole USD asset for peg-insertion experiments.

Square block 60×60×40 mm with a circular Ø24 mm vertical through-hole at centre.
Origin at geometric centre of the block (z=0 ↔ block centre height).

Collision uses ``meshSimplification`` on the visual mesh directly (requires
kinematic body in env_cfg).  Previous versions used 4 invisible box colliders
forming a square channel, which diverged from the circular visual hole.

Usage:
    python generate_block_with_hole.py
    python generate_block_with_hole.py /tmp/block.usda
"""
from __future__ import annotations

import math
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _create_material(stage, path, color, metallic=0.3, roughness=0.5):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _box_with_round_hole_mesh(half_w, half_d, half_h, hole_r, n_sides=64):
    """Square box with a cylindrical through-hole along Z.

    Uses a projective approach: for each of ``n_sides`` evenly-spaced angles,
    a ray from the origin is intersected with both the circle (radius
    ``hole_r``) and the square boundary.  This produces matched vertex pairs
    on the inner circle and outer square, connected by quads for the top
    annulus, bottom annulus, inner cylinder wall, and outer box wall.

    ``n_sides`` **must** be divisible by 4 so that box corners fall exactly
    on vertex positions (at 45°, 135°, 225°, 315°).
    """
    assert n_sides % 4 == 0, "n_sides must be divisible by 4"

    pts = []
    fvc = []
    fvi = []

    angles = [2.0 * math.pi * i / n_sides for i in range(n_sides)]

    # Per-angle vertex indices: bottom-circle, top-circle, bottom-square, top-square
    bc, tc, bs, ts = [], [], [], []

    for a in angles:
        ca, sa = math.cos(a), math.sin(a)

        # Circle point
        cx, cy = hole_r * ca, hole_r * sa

        # Square boundary point (ray–box intersection)
        cands = []
        if abs(ca) > 1e-12:
            cands.append(half_w / abs(ca))
        if abs(sa) > 1e-12:
            cands.append(half_d / abs(sa))
        t = min(cands)
        sx, sy = t * ca, t * sa

        bc.append(len(pts)); pts.append(Gf.Vec3f(cx, cy, -half_h))
        tc.append(len(pts)); pts.append(Gf.Vec3f(cx, cy, +half_h))
        bs.append(len(pts)); pts.append(Gf.Vec3f(sx, sy, -half_h))
        ts.append(len(pts)); pts.append(Gf.Vec3f(sx, sy, +half_h))

    for i in range(n_sides):
        j = (i + 1) % n_sides

        # Inner hole wall (visible from inside the hole)
        fvc.append(4); fvi.extend([bc[i], tc[i], tc[j], bc[j]])

        # Outer box wall (visible from outside)
        fvc.append(4); fvi.extend([bs[j], ts[j], ts[i], bs[i]])

        # Top annulus (visible from above, normal +Z)
        fvc.append(4); fvi.extend([tc[i], ts[i], ts[j], tc[j]])

        # Bottom annulus (visible from below, normal −Z)
        fvc.append(4); fvi.extend([bc[j], bs[j], bs[i], bc[i]])

    return pts, fvc, fvi


def create_block_with_hole(out_path: str):
    half_w = 0.030   # 60 mm total width
    half_d = 0.030   # 60 mm total depth
    half_h = 0.020   # 40 mm total height
    hole_r = 0.012   # Ø24 mm hole
    n_sides = 64

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Block")
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.GetMassAttr().Set(0.5)

    mat = _create_material(
        stage, "/Block/Materials/BlockMaterial",
        color=(0.3, 0.3, 0.35), metallic=0.4, roughness=0.45,
    )

    # ---- Visual mesh with meshSimplification collision ----
    points, fvc, fvi = _box_with_round_hole_mesh(half_w, half_d, half_h, hole_r, n_sides)

    mesh = UsdGeom.Mesh.Define(stage, "/Block/Visual")
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(fvc)
    mesh.GetFaceVertexIndicesAttr().Set(fvi)
    mesh.GetDoubleSidedAttr().Set(True)

    extent = [Gf.Vec3f(-half_w, -half_d, -half_h),
              Gf.Vec3f(half_w, half_d, half_h)]
    mesh.GetExtentAttr().Set(extent)
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

    # ---- Collision: meshSimplification on the visual mesh ----
    # Requires kinematic body (set in rod_insert_env_cfg.py).
    # This gives exact circular-hole collision matching the visual mesh.
    visual_prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(visual_prim)
    mesh_coll_api = UsdPhysics.MeshCollisionAPI.Apply(visual_prim)
    mesh_coll_api.GetApproximationAttr().Set("meshSimplification")

    stage.GetRootLayer().Save()
    print(f"Block-with-hole USD saved to: {out_path}")


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_with_hole.usda")
    out = sys.argv[1] if len(sys.argv) > 1 else default_path
    create_block_with_hole(out)
