#!/usr/bin/env python3
"""Plot success rate curves with ablation experiment results overlaid.

Combines:
  - Original iterative pipeline curves (10 iterations)
  - Experiment A: Independent evaluation (iter10 weights, always reset)
  - Experiment B: A-iter10 + B-original (alternating, no B finetuning)

Usage:
    python scripts/scripts_pick_place/plot_success_rate_with_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path("data")

    # ---- Load pipeline data ----
    with open(base / "success_rate_record.json", "r") as f:
        record = json.load(f)

    iterations = record["iterations"]
    iter_nums = np.array([it["iteration"] for it in iterations])
    a_rates = np.array([it["performance_metrics"]["task_A_success_rate"] * 100 for it in iterations])
    b_rates = np.array([it["performance_metrics"]["task_B_success_rate"] * 100 for it in iterations])

    # ---- Load Experiment A (independent eval) ----
    with open(base / "ablation_exp_a_independent.stats.json", "r") as f:
        exp_a = json.load(f)
    exp_a_task_a = exp_a["summary"]["task_A_success_rate"] * 100
    exp_a_task_b = exp_a["summary"]["task_B_success_rate"] * 100

    # ---- Load Experiment B (A-iter10 + B-original, alternating) ----
    with open(base / "ablation_exp_b_eval_A.stats.json", "r") as f:
        exp_b = json.load(f)
    exp_b_task_a = exp_b["summary"]["task_A_success_rate"] * 100
    exp_b_task_b = exp_b["summary"]["task_B_success_rate"] * 100

    # ---- Print summary ----
    print("=" * 65)
    print("Results Summary")
    print("=" * 65)
    print()
    print("Pipeline (alternating A→B, both finetuned):")
    print(f"  Iter  1: A={a_rates[0]:.0f}%   B={b_rates[0]:.0f}%")
    print(f"  Iter 10: A={a_rates[-1]:.0f}%   B={b_rates[-1]:.0f}%")
    print()
    print(f"Exp A (independent eval, iter10 weights, always reset):")
    print(f"  Task A: {exp_a_task_a:.0f}%   Task B: {exp_a_task_b:.0f}%")
    print()
    print(f"Exp B (A=iter10 + B=original, alternating):")
    print(f"  Task A: {exp_b_task_a:.0f}%   Task B: {exp_b_task_b:.0f}%")
    print()
    print("-" * 65)
    print("Analysis:")
    print(f"  Pipeline iter10 B: {b_rates[-1]:.0f}%")
    print(f"  Exp A B (independent, same weights): {exp_a_task_b:.0f}%")
    delta_a = exp_a_task_b - b_rates[-1]
    print(f"    Δ = {delta_a:+.0f}%  →  {'Reset mechanism IS a factor' if delta_a > 5 else 'Reset mechanism is NOT a major factor'}")
    print()
    print(f"  Pipeline iter1 B (original): {b_rates[0]:.0f}%")
    print(f"  Exp B B (original weights, alternating): {exp_b_task_b:.0f}%")
    delta_b = exp_b_task_b - b_rates[-1]
    print(f"    Δ vs iter10 = {delta_b:+.0f}%  →  {'Finetuning hurt B (insufficient convergence)' if delta_b > 5 else 'Finetuning did NOT hurt B much'}")
    print()

    # =====================================================================
    # Plot
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Pipeline curves
    ax.plot(iter_nums, a_rates, "o-", color="#2196F3", linewidth=2.2, markersize=7,
            label="Task A — pipeline (alternating)", zorder=3)
    ax.plot(iter_nums, b_rates, "s-", color="#F44336", linewidth=2.2, markersize=7,
            label="Task B — pipeline (alternating)", zorder=3)

    # Annotate pipeline points
    for x, ya, yb in zip(iter_nums, a_rates, b_rates):
        ax.annotate(f"{ya:.0f}", (x, ya), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color="#1565C0")
        ax.annotate(f"{yb:.0f}", (x, yb), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7.5, color="#C62828")

    # ---- Experiment A markers (at x=10, offset slightly for visibility) ----
    exp_x = 10  # iteration 10

    ax.scatter([exp_x + 0.25], [exp_a_task_a], marker="D", s=120, color="#2196F3",
               edgecolors="black", linewidths=1.2, zorder=5)
    ax.scatter([exp_x + 0.25], [exp_a_task_b], marker="D", s=120, color="#F44336",
               edgecolors="black", linewidths=1.2, zorder=5,
               label=f"Exp A: independent eval (always reset)")

    ax.annotate(f"A={exp_a_task_a:.0f}%", (exp_x + 0.25, exp_a_task_a),
                textcoords="offset points", xytext=(28, -4), ha="left", fontsize=9,
                color="#1565C0", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1))
    ax.annotate(f"B={exp_a_task_b:.0f}%", (exp_x + 0.25, exp_a_task_b),
                textcoords="offset points", xytext=(28, -4), ha="left", fontsize=9,
                color="#C62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1))

    # ---- Experiment B markers (at x=10, offset slightly the other way) ----
    ax.scatter([exp_x - 0.25], [exp_b_task_a], marker="^", s=120, color="#2196F3",
               edgecolors="green", linewidths=1.5, zorder=5)
    ax.scatter([exp_x - 0.25], [exp_b_task_b], marker="^", s=120, color="#F44336",
               edgecolors="green", linewidths=1.5, zorder=5,
               label=f"Exp B: A-iter10 + B-original (alternating)")

    ax.annotate(f"A={exp_b_task_a:.0f}%", (exp_x - 0.25, exp_b_task_a),
                textcoords="offset points", xytext=(-70, 10), ha="left", fontsize=9,
                color="#1565C0", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=1))
    ax.annotate(f"B={exp_b_task_b:.0f}%", (exp_x - 0.25, exp_b_task_b),
                textcoords="offset points", xytext=(-70, 10), ha="left", fontsize=9,
                color="#C62828", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=1))

    # ---- Reference lines ----
    # Original B performance (iter 1)
    ax.axhline(y=b_rates[0], color="#F44336", linestyle=":", alpha=0.35, linewidth=1)
    ax.text(0.6, b_rates[0] + 1.2, f"B baseline (iter1): {b_rates[0]:.0f}%",
            fontsize=8, color="#C62828", alpha=0.6)

    # Styling
    ax.set_xlabel("Training Iteration", fontsize=13)
    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title("Success Rate Curves with Ablation Experiments", fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.set_xlim(0.3, 11.5)
    ax.set_xticks(iter_nums)
    ax.legend(fontsize=9.5, loc="center left", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # ---- Text box with analysis ----
    analysis_text = (
        f"Pipeline iter10:  A={a_rates[-1]:.0f}%  B={b_rates[-1]:.0f}%\n"
        f"Exp A (indep.):   A={exp_a_task_a:.0f}%  B={exp_a_task_b:.0f}%  (reset → Δ={delta_a:+.0f}%)\n"
        f"Exp B (B-orig.):  A={exp_b_task_a:.0f}%  B={exp_b_task_b:.0f}%  (no finetune → Δ={delta_b:+.0f}%)"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.85, edgecolor="gray")
    ax.text(0.98, 0.03, analysis_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props, family="monospace")

    plt.tight_layout()
    out_path = base / "success_rate_with_ablation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
