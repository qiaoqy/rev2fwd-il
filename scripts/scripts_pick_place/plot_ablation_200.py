#!/usr/bin/env python3
"""Plot ablation bar chart comparing 3 experiments (200 cycles each).

Experiments:
  1. Pipeline iter10: A-iter10 + B-iter10, alternating A→B with recovery
  2. Exp A (Independent): A-iter10 + B-iter10, full reset before every episode
  3. Exp B (B-original):  A-iter10 + B-original, alternating A→B with recovery

Usage:
    python scripts/scripts_pick_place/plot_ablation_200.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_alternating_stats(path: str) -> dict:
    """Load stats from 9_eval_with_recovery.py output."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stats not found: {p}")
    with open(p, "r") as f:
        data = json.load(f)
    s = data["summary"]
    return {
        "task_A_rate": s["task_A_success_rate"] * 100,
        "task_B_rate": s["task_B_success_rate"] * 100,
        "task_A_count": s["task_A_success_count"],
        "task_B_count": s["task_B_success_count"],
        "task_A_total": s["total_task_A_episodes"],
        "task_B_total": s["total_task_B_episodes"],
    }


def load_independent_stats(path: str) -> dict:
    """Load stats from 10_eval_independent.py output."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stats not found: {p}")
    with open(p, "r") as f:
        data = json.load(f)
    s = data["summary"]
    return {
        "task_A_rate": s["task_A_success_rate"] * 100,
        "task_B_rate": s["task_B_success_rate"] * 100,
        "task_A_count": s["task_A_success_count"],
        "task_B_count": s.get("task_B_success_count", s.get("task_B_total_episodes", 0)),
        "task_A_total": s.get("task_A_total_episodes", 200),
        "task_B_total": s.get("task_B_total_episodes", 200),
    }


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0, 0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z / denom * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))
    lo = max(0, center - margin) * 100
    hi = min(1, center + margin) * 100
    return lo, hi


def main():
    base = Path("data")

    # ---- Load all three experiments ----
    pipeline = load_alternating_stats(base / "ablation200_pipeline_A.stats.json")
    independent = load_independent_stats(base / "ablation200_independent.stats.json")
    expb = load_alternating_stats(base / "ablation200_expb_A.stats.json")

    experiments = {
        "Pipeline iter10\n(A+B finetuned,\nalternating)": pipeline,
        "Exp A: Independent\n(A+B finetuned,\nalways reset)": independent,
        "Exp B: B-original\n(A finetuned + B orig,\nalternating)": expb,
    }

    # ---- Print summary ----
    print("=" * 70)
    print("200-Cycle Ablation Results")
    print("=" * 70)
    for name, d in experiments.items():
        label = name.replace("\n", " ")
        print(f"\n  {label}:")
        print(f"    Task A: {d['task_A_count']}/{d['task_A_total']} = {d['task_A_rate']:.1f}%")
        print(f"    Task B: {d['task_B_count']}/{d['task_B_total']} = {d['task_B_rate']:.1f}%")

    # ---- Compute CIs ----
    def get_ci(d, task):
        lo, hi = wilson_ci(d[f"task_{task}_count"], d[f"task_{task}_total"])
        rate = d[f"task_{task}_rate"]
        return rate, rate - lo, hi - rate

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(experiments.keys())
    x = np.arange(len(labels))
    width = 0.32

    # Task A bars
    a_rates = [d["task_A_rate"] for d in experiments.values()]
    a_errs_lo = [get_ci(d, "A")[1] for d in experiments.values()]
    a_errs_hi = [get_ci(d, "A")[2] for d in experiments.values()]

    # Task B bars
    b_rates = [d["task_B_rate"] for d in experiments.values()]
    b_errs_lo = [get_ci(d, "B")[1] for d in experiments.values()]
    b_errs_hi = [get_ci(d, "B")[2] for d in experiments.values()]

    bars_a = ax.bar(x - width / 2, a_rates, width,
                    color="#2196F3", alpha=0.85, label="Task A (pick → goal)",
                    yerr=[a_errs_lo, a_errs_hi], capsize=4, error_kw=dict(lw=1.2))
    bars_b = ax.bar(x + width / 2, b_rates, width,
                    color="#F44336", alpha=0.85, label="Task B (goal → table)",
                    yerr=[b_errs_lo, b_errs_hi], capsize=4, error_kw=dict(lw=1.2))

    # Value labels on bars
    for i, (bar_a, bar_b) in enumerate(zip(bars_a, bars_b)):
        d = list(experiments.values())[i]
        ax.text(bar_a.get_x() + bar_a.get_width() / 2, bar_a.get_height() + a_errs_hi[i] + 1.5,
                f"{d['task_A_rate']:.1f}%\n({d['task_A_count']}/{d['task_A_total']})",
                ha="center", va="bottom", fontsize=9, color="#1565C0", fontweight="bold")
        ax.text(bar_b.get_x() + bar_b.get_width() / 2, bar_b.get_height() + b_errs_hi[i] + 1.5,
                f"{d['task_B_rate']:.1f}%\n({d['task_B_count']}/{d['task_B_total']})",
                ha="center", va="bottom", fontsize=9, color="#C62828", fontweight="bold")

    # Reference lines from original 50-cycle results
    ax.axhline(y=90, color="#F44336", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(2.55, 91, "B baseline\n(iter1: 90%)", fontsize=7.5, color="#C62828", alpha=0.5)

    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title("Ablation: Task B Decline Diagnosis (200 cycles each, 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")

    # ---- Analysis text box ----
    p_b = pipeline["task_B_rate"]
    ind_b = independent["task_B_rate"]
    orig_b = expb["task_B_rate"]
    delta_reset = ind_b - p_b
    delta_finetune = orig_b - p_b

    analysis = (
        f"Task B comparison (vs Pipeline {p_b:.1f}%):\n"
        f"  Independent (always reset):  {ind_b:.1f}%  (Δ={delta_reset:+.1f}%)  ← reset effect\n"
        f"  B-original (no finetune):    {orig_b:.1f}%  (Δ={delta_finetune:+.1f}%)  ← finetune effect"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="gray")
    ax.text(0.02, 0.02, analysis, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=props, family="monospace")

    plt.tight_layout()
    out_path = base / "ablation_200_comparison.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
