#!/usr/bin/env python3
"""Plot success rate curves from iterative training pipeline results.

Usage:
    python scripts/scripts_pick_place/plot_success_rate.py \
        --record data/success_rate_record.json \
        --out data/success_rate_curve.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot success rate curves.")
    parser.add_argument("--record", type=str, required=True,
                        help="Path to the JSON record file from the pipeline.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output image path. Default: <record>_curve.png")
    args = parser.parse_args()

    record_path = Path(args.record)
    if not record_path.exists():
        print(f"ERROR: Record file not found: {record_path}")
        sys.exit(1)

    with open(record_path, "r") as f:
        record = json.load(f)

    iterations = record.get("iterations", [])
    if not iterations:
        print("No iteration data found in record.")
        sys.exit(1)

    iter_nums = []
    a_rates = []
    b_rates = []
    a_counts = []
    b_counts = []
    total_a = []
    total_b = []

    for it in iterations:
        metrics = it.get("performance_metrics", {})
        iter_nums.append(it["iteration"])
        a_rates.append(metrics.get("task_A_success_rate", 0) * 100)
        b_rates.append(metrics.get("task_B_success_rate", 0) * 100)
        a_counts.append(metrics.get("task_A_success_count", 0))
        b_counts.append(metrics.get("task_B_success_count", 0))
        total_a.append(metrics.get("total_task_A_episodes", 0))
        total_b.append(metrics.get("total_task_B_episodes", 0))

    iter_nums = np.array(iter_nums)
    a_rates = np.array(a_rates)
    b_rates = np.array(b_rates)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Left: success rate curves
    ax = axes[0]
    ax.plot(iter_nums, a_rates, "o-", color="#2196F3", linewidth=2, markersize=7, label="Task A (pick → goal)")
    ax.plot(iter_nums, b_rates, "s-", color="#F44336", linewidth=2, markersize=7, label="Task B (goal → table)")
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title("Success Rate Over Iterations", fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.set_xticks(iter_nums)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, (x, ya, yb) in enumerate(zip(iter_nums, a_rates, b_rates)):
        ax.annotate(f"{ya:.0f}%", (x, ya), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#2196F3")
        ax.annotate(f"{yb:.0f}%", (x, yb), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#F44336")

    # Right: success counts (bar chart)
    ax2 = axes[1]
    width = 0.35
    ax2.bar(iter_nums - width / 2, a_counts, width, color="#2196F3", alpha=0.7, label="Task A successes")
    ax2.bar(iter_nums + width / 2, b_counts, width, color="#F44336", alpha=0.7, label="Task B successes")
    # Add total attempt count as text
    for i, (x, ca, cb, ta, tb) in enumerate(zip(iter_nums, a_counts, b_counts, total_a, total_b)):
        ax2.text(x - width / 2, ca + 0.5, f"{ca}/{ta}", ha="center", fontsize=7, color="#1565C0")
        ax2.text(x + width / 2, cb + 0.5, f"{cb}/{tb}", ha="center", fontsize=7, color="#C62828")
    ax2.set_xlabel("Iteration", fontsize=13)
    ax2.set_ylabel("Success Count", fontsize=13)
    ax2.set_title("Successes per Iteration", fontsize=14, fontweight="bold")
    ax2.set_xticks(iter_nums)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save
    if args.out is None:
        out_path = record_path.with_name(record_path.stem + "_curve.png")
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved success rate plot to: {out_path}")

    # Also print a text summary
    print(f"\n{'='*50}")
    print("Success Rate Summary")
    print(f"{'='*50}")
    print(f"{'Iter':>4}  {'A Rate':>8}  {'B Rate':>8}  {'A (ok/tot)':>12}  {'B (ok/tot)':>12}")
    print(f"{'-'*50}")
    for i in range(len(iter_nums)):
        print(f"{iter_nums[i]:4d}  {a_rates[i]:7.1f}%  {b_rates[i]:7.1f}%  "
              f"{a_counts[i]:4d}/{total_a[i]:<4d}     {b_counts[i]:4d}/{total_b[i]:<4d}")


if __name__ == "__main__":
    main()
