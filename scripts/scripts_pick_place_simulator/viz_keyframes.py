#!/usr/bin/env python3
"""Generate key-frame PNG snapshots from eval NPZ for quick visual inspection."""

import numpy as np
import cv2
from pathlib import Path
import sys

BASE = Path("data/pick_place_isaac_lab_simulation/exp38")
OUT = BASE / "eval_debug_videos"
OUT.mkdir(exist_ok=True)

GOAL_XY = np.array([0.5, -0.2])
RED_CENTER = np.array([0.5, 0.2])
RED_SIZE = np.array([0.30, 0.30])

SCALE = 4
SRC = 128


def annotate(img_rgb, ee, cubes, step, total, label):
    """Annotate a single image and return BGR canvas."""
    h = SRC * SCALE
    w = SRC * SCALE
    img = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def wp(x, y):
        u = 1.0 - (x - 0.15) / 0.70
        v = 1.0 - (y - (-0.45)) / 0.90
        return int(np.clip(v * w, 0, w-1)), int(np.clip(u * h, 0, h-1))

    # Goal (green dot)
    gc, gr = wp(GOAL_XY[0], GOAL_XY[1])
    cv2.circle(img, (gc, gr), 6, (0, 255, 128), 2)
    cv2.putText(img, "Goal", (gc+8, gr-4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 128), 1)

    # Red region
    x1, y1 = RED_CENTER - RED_SIZE / 2
    x2, y2 = RED_CENTER + RED_SIZE / 2
    c1, r1 = wp(x1, y1)
    c2, r2 = wp(x2, y2)
    cv2.rectangle(img, (min(c1,c2), min(r1,r2)), (max(c1,c2), max(r1,r2)),
                  (0, 0, 200), 2)
    cv2.putText(img, "Red", (min(c1,c2)+4, min(r1,r2)+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    # EE
    ec, er = wp(ee[0], ee[1])
    cv2.circle(img, (ec, er), 5, (0, 255, 0), -1)
    cv2.putText(img, "EE", (ec+6, er-4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1)

    # Cubes
    colors_names = [
        ("L", (0, 215, 255)),
        ("M", (255, 100, 50)),
        ("S", (200, 50, 200)),
    ]
    for (name, color), cpose in zip(colors_names, cubes):
        if cpose is not None:
            cc, cr = wp(cpose[0], cpose[1])
            cv2.circle(img, (cc, cr), 7, color, -1)
            cv2.putText(img, name, (cc+8, cr-4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1)

    # Label
    cv2.putText(img, f"{label}  step {step}/{total}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def process_npz(npz_path, task_label):
    data = np.load(npz_path, allow_pickle=True)
    episodes = data["episodes"]
    print(f"\n{task_label}: {len(episodes)} episodes from {npz_path}")

    for i, ep in enumerate(episodes):
        ep = ep if isinstance(ep, dict) else ep.item()
        imgs = ep["images"]
        ee = ep["ee_pose"]
        cl = ep.get("cube_large_pose")
        cm = ep.get("cube_medium_pose")
        cs = ep.get("cube_small_pose")
        T = len(imgs)
        success = ep.get("success", False)
        tag = "succ" if success else "fail"

        # Key frames: start, 25%, 50%, 75%, end
        keyframes = [0, T//4, T//2, 3*T//4, T-1]
        row_imgs = []
        for t in keyframes:
            cubes = [
                cl[t] if cl is not None else None,
                cm[t] if cm is not None else None,
                cs[t] if cs is not None else None,
            ]
            ann = annotate(imgs[t], ee[t], cubes, t, T,
                           f"{task_label} ep{i} [{tag}]")
            row_imgs.append(ann)

        # Stitch horizontally
        strip = np.concatenate(row_imgs, axis=1)
        out_path = OUT / f"{task_label}_ep{i}_{tag}_keyframes.png"
        cv2.imwrite(str(out_path), strip)
        print(f"  {out_path.name}  ({strip.shape[1]}x{strip.shape[0]})")


process_npz(BASE / "eval_cyclic_A.npz", "Task_A")
process_npz(BASE / "eval_cyclic_B.npz", "Task_B")
print(f"\nDone. See: {OUT}")
