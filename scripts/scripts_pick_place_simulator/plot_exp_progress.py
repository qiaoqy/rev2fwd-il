#!/usr/bin/env python3
"""
plot_exp_progress.py — Rev2Fwd iterative training progress visualization.

Two panels:
  1. Fair test success rate over iterations (Task A & B)
  2. Cyclic collection chain: colored success/failure grid per cycle,
     brackets showing consecutive A+B autonomous runs,
     red markers where a reset was triggered (either task failed).

Read-only — does NOT write to any running pipeline files.

Usage:
    python scripts/scripts_pick_place_simulator/plot_exp_progress.py \
        --exp_dir data/pick_place_isaac_lab_simulation/exp17 \
        --out     data/pick_place_isaac_lab_simulation/exp17/progress_plot.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_record(exp_dir):
    with open(exp_dir / "record.json") as f:
        return json.load(f)


def load_cycle_stats(exp_dir, iter_idx):
    """Return (a_list, b_list) of 0/1 per cycle, or None if file missing."""
    path = exp_dir / f"iter{iter_idx}_collect_A.stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    results = data.get("per_cycle_results", [])
    a = [1 if r["task_A_success"] else 0 for r in results]
    b = [1 if r["task_B_success"] else 0 for r in results]
    return a, b


def get_continuous_runs(a_list, b_list):
    """
    Return list of (start_0based, length) for maximal streaks where
    BOTH Task A and Task B succeeded consecutively (no reset needed).
    """
    runs = []
    current_start = None
    current_len = 0
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if a == 1 and b == 1:
            if current_start is None:
                current_start = i
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None:
                runs.append((current_start, current_len))
            current_start = None
            current_len = 0
    if current_start is not None:
        runs.append((current_start, current_len))
    return runs


# ---------------------------------------------------------------------------
# Cyclic-chain panel
# ---------------------------------------------------------------------------

def draw_cyclic_panel(ax, a_list, b_list, iter_idx):
    n = len(a_list)

    # ── colored cell grid (imshow) ──────────────────────────────────────────
    COLOR_SUCCESS = np.array([100, 175, 130, 255]) / 255.0   # muted sage green
    COLOR_FAIL    = np.array([205, 110, 105, 255]) / 255.0   # muted terracotta

    grid = np.zeros((2, n, 4))
    for i in range(n):
        grid[0, i] = COLOR_SUCCESS if a_list[i] else COLOR_FAIL   # Task A (top row)
        grid[1, i] = COLOR_SUCCESS if b_list[i] else COLOR_FAIL   # Task B (bottom row)

    # extent: [x_left, x_right, y_bottom, y_top]
    # image row 0 (A) → top of image (near y=1.5); row 1 (B) → bottom (near y=-0.5)
    ax.imshow(grid, aspect="auto", extent=[-0.5, n - 0.5, -0.5, 1.5],
              interpolation="nearest", zorder=2)

    # white separator between A and B rows
    ax.axhline(y=0.5, color="white", lw=2.0, zorder=3)

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Task B", "Task A"], fontsize=9)

    # x ticks every 5 cycles (1-indexed labels)
    tick_pos = list(range(4, n, 5))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(i + 1) for i in tick_pos], fontsize=8)
    ax.set_xlabel("Cycle #", fontsize=10)

    # ── continuous-run brackets above the grid ──────────────────────────────
    runs = get_continuous_runs(a_list, b_list)
    max_run   = max((r[1] for r in runs), default=0)
    num_resets = sum(1 for a, b in zip(a_list, b_list) if not (a and b))

    brace_y = 1.72
    cap_h   = 0.08

    for start, length in runs:
        end = start + length - 1
        mid = (start + end) / 2.0
        x0, x1 = start - 0.45, end + 0.45

        if length >= 3:
            col, lw, fs, fw = "#3d7d5c", 2.2, 9, "bold"
        elif length == 2:
            col, lw, fs, fw = "#5a9e78", 1.8, 8, "normal"
        else:
            col, lw, fs, fw = "#87b8a0", 1.2, 7, "normal"

        ax.plot([x0, x1], [brace_y, brace_y], color=col, lw=lw, zorder=4,
                solid_capstyle="round")
        ax.plot([x0, x0], [brace_y - cap_h, brace_y], color=col, lw=lw, zorder=4)
        ax.plot([x1, x1], [brace_y - cap_h, brace_y], color=col, lw=lw, zorder=4)
        ax.text(mid, brace_y + 0.05, str(length), ha="center", va="bottom",
                fontsize=fs, color=col, fontweight=fw, zorder=5)

    # ── reset markers below the grid ────────────────────────────────────────
    reset_y = -0.78
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if not (a and b):
            ax.plot(i, reset_y, "v", color="#b85252", markersize=4.5,
                    alpha=0.75, zorder=3)

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(reset_y - 0.15, brace_y + 0.4)

    ax.set_title(
        f"Iter {iter_idx} — Cyclic A→B Collection\n"
        f"max continuous run = {max_run} cycles   |   resets triggered = {num_resets}/{n}",
        fontsize=10, pad=6,
    )

    legend_elems = [
        mpatches.Patch(facecolor="#64af82", label="Success"),
        mpatches.Patch(facecolor="#cd6e69", label="Failure → reset triggered  ▼"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right", framealpha=0.9)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot iterative training progress.")
    parser.add_argument("--exp_dir", default="data/pick_place_isaac_lab_simulation/exp17")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    record  = load_record(exp_dir)
    iterations = record["iterations"]
    config   = record.get("config", {})
    exp_name = config.get("exp_name", exp_dir.name)
    planned  = config.get("iter_rounds", len(iterations))

    iter_nums  = [it["iteration"] for it in iterations]
    fair_A     = [it["fair_test_metrics"]["fair_A_success_rate"] * 100 for it in iterations]
    fair_B     = [it["fair_test_metrics"]["fair_B_success_rate"] * 100 for it in iterations]
    fair_A_str = [
        f"{it['fair_test_metrics']['fair_A_num_success']}/{it['fair_test_metrics']['fair_A_num_total']}"
        for it in iterations
    ]
    fair_B_str = [
        f"{it['fair_test_metrics']['fair_B_num_success']}/{it['fair_test_metrics']['fair_B_num_total']}"
        for it in iterations
    ]

    COLOR_A = "#5b8db8"   # muted steel blue — Task A
    COLOR_B = "#c06060"   # muted terracotta  — Task B

    n_cyclic = sum(
        1 for i in iter_nums
        if (exp_dir / f"iter{i}_collect_A.stats.json").exists()
    )
    n_cols = max(n_cyclic, 2)

    # ── Figure layout: row 0 = fair test (full width), row 1 = cyclic panels ─
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        hspace=0.62, wspace=0.28,
        height_ratios=[1.1, 1.4],
    )

    # ── Panel 1: Fair Test ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(
        iter_nums, fair_A, "o-", color=COLOR_A, lw=2.5, ms=10,
        label="Task A  (random start → green goal)",
        markerfacecolor="white", markeredgewidth=2.5, zorder=3,
    )
    ax1.plot(
        iter_nums, fair_B, "s-", color=COLOR_B, lw=2.5, ms=10,
        label="Task B  (green goal → red zone)",
        markerfacecolor="white", markeredgewidth=2.5, zorder=3,
    )

    for n_, a, b, as_, bs_ in zip(iter_nums, fair_A, fair_B, fair_A_str, fair_B_str):
        ax1.annotate(
            f"{a:.0f}%  ({as_})", (n_, a),
            xytext=(10, -16), textcoords="offset points",
            fontsize=10, color=COLOR_A, fontweight="bold", zorder=4,
        )
        ax1.annotate(
            f"{b:.0f}%  ({bs_})", (n_, b),
            xytext=(10, 5), textcoords="offset points",
            fontsize=10, color=COLOR_B, fontweight="bold", zorder=4,
        )

    ax1.axhline(100, color="gray", ls="--", alpha=0.25, lw=1)
    x_pad = 0.5
    ax1.set_xlim(iter_nums[0] - x_pad, iter_nums[-1] + x_pad)
    ax1.set_xticks(iter_nums)
    ax1.set_xticklabels([str(i) for i in iter_nums], fontsize=10)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Success Rate (%)", fontsize=11)
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_title(
        f"{exp_name.upper()} — Fair Test Success Rate"
        f"  [{iter_nums[-1]}/{planned} iterations done  •  50 independent episodes each]",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Cyclic chains ───────────────────────────────────────────────
    col = 0
    for iter_n in iter_nums:
        cd = load_cycle_stats(exp_dir, iter_n)
        if cd is None:
            continue
        ax_c = fig.add_subplot(gs[1, col])
        draw_cyclic_panel(ax_c, cd[0], cd[1], iter_n)
        col += 1

    fig.suptitle(
        "Rev2Fwd Pick-and-Place — Iterative Training Progress (Exp17)\n"
        "Numbers above grid = consecutive cycles with both A+B success (no reset needed)   "
        "▼ = reset triggered (either task failed)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out_path = args.out or str(exp_dir / "progress_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
