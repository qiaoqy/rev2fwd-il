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


def wilson_ci(successes, total, z=1.96):
    """Wilson score 95% confidence interval for a proportion.
    
    Returns (lower_bound, upper_bound) as percentages (0-100).
    """
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z / denom * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))
    lo = max(0.0, center - margin) * 100
    hi = min(1.0, center + margin) * 100
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description="Plot success rate curves.")
    parser.add_argument("--record", type=str, required=True,
                        help="Path to the JSON record file from the pipeline.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output image path. Default: <record>_curve.png")
    parser.add_argument("--metrics_key", type=str, default="performance_metrics",
                        help="Key in each iteration entry to read metrics from. "
                             "Default: 'performance_metrics'. Use 'fair_test_metrics' "
                             "for pipeline_fair experiments.")
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
        metrics = it.get(args.metrics_key, it.get("performance_metrics", {}))
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

    # ---- Compute Wilson 95% confidence intervals ----
    a_ci_lo = np.array([wilson_ci(k, n)[0] for k, n in zip(a_counts, total_a)])
    a_ci_hi = np.array([wilson_ci(k, n)[1] for k, n in zip(a_counts, total_a)])
    b_ci_lo = np.array([wilson_ci(k, n)[0] for k, n in zip(b_counts, total_b)])
    b_ci_hi = np.array([wilson_ci(k, n)[1] for k, n in zip(b_counts, total_b)])

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Left: success rate curves with CI bands
    ax = axes[0]

    # CI shaded bands
    ax.fill_between(iter_nums, a_ci_lo, a_ci_hi, color="#2196F3", alpha=0.15, label="_nolegend_")
    ax.fill_between(iter_nums, b_ci_lo, b_ci_hi, color="#F44336", alpha=0.15, label="_nolegend_")

    # Lines with error bars
    ax.errorbar(iter_nums, a_rates,
                yerr=[a_rates - a_ci_lo, a_ci_hi - a_rates],
                fmt="o-", color="#2196F3", linewidth=2, markersize=7, capsize=4, capthick=1.2,
                label="Task A (pick → goal)")
    ax.errorbar(iter_nums, b_rates,
                yerr=[b_rates - b_ci_lo, b_ci_hi - b_rates],
                fmt="s-", color="#F44336", linewidth=2, markersize=7, capsize=4, capthick=1.2,
                label="Task B (goal → table)")

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title("Success Rate Over Iterations (95% Wilson CI)", fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.set_xticks(iter_nums)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Annotate points with rate and CI
    for i, (x, ya, yb) in enumerate(zip(iter_nums, a_rates, b_rates)):
        ax.annotate(f"{ya:.0f}%", (x, ya), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=7.5, color="#1565C0", fontweight="bold")
        ax.annotate(f"{yb:.0f}%", (x, yb), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=7.5, color="#C62828", fontweight="bold")

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

    # Also print a text summary with CIs
    print(f"\n{'='*80}")
    print("Success Rate Summary (with 95% Wilson CI)")
    print(f"{'='*80}")
    print(f"{'Iter':>4}  {'A Rate':>8}  {'A 95% CI':>16}  {'B Rate':>8}  {'B 95% CI':>16}  {'A (ok/tot)':>10}  {'B (ok/tot)':>10}")
    print(f"{'-'*80}")
    for i in range(len(iter_nums)):
        print(f"{iter_nums[i]:4d}  {a_rates[i]:7.1f}%  [{a_ci_lo[i]:5.1f}%, {a_ci_hi[i]:5.1f}%]"
              f"  {b_rates[i]:7.1f}%  [{b_ci_lo[i]:5.1f}%, {b_ci_hi[i]:5.1f}%]"
              f"  {a_counts[i]:4d}/{total_a[i]:<4d}   {b_counts[i]:4d}/{total_b[i]:<4d}")

    # ---- Analysis: check if B decline is significant ----
    print(f"\n{'='*80}")
    print("Confidence Interval Analysis")
    print(f"{'='*80}")
    # Compare first few iters (1-3) vs last few iters (8-10) for Task B
    early_b_counts = sum(b_counts[:3])
    early_b_total = sum(total_b[:3])
    late_b_counts = sum(b_counts[-3:])
    late_b_total = sum(total_b[-3:])
    early_lo, early_hi = wilson_ci(early_b_counts, early_b_total)
    late_lo, late_hi = wilson_ci(late_b_counts, late_b_total)
    early_rate = early_b_counts / early_b_total * 100
    late_rate = late_b_counts / late_b_total * 100
    print(f"  Task B early (iter 1-3):  {early_b_counts}/{early_b_total} = {early_rate:.1f}%  CI [{early_lo:.1f}%, {early_hi:.1f}%]")
    print(f"  Task B late  (iter 8-10): {late_b_counts}/{late_b_total} = {late_rate:.1f}%  CI [{late_lo:.1f}%, {late_hi:.1f}%]")
    if late_hi < early_lo:
        print(f"  → CIs do NOT overlap → Task B decline is statistically significant (p<0.05)")
    elif late_lo > early_hi:
        print(f"  → CIs do NOT overlap → Task B actually improved significantly")
    else:
        print(f"  → CIs overlap → Task B decline is NOT statistically significant at 95% level")
        print(f"    (overlap region: [{max(late_lo, early_lo):.1f}%, {min(late_hi, early_hi):.1f}%])")
    
    # Same for Task A
    early_a_counts = sum(a_counts[:3])
    early_a_total = sum(total_a[:3])
    late_a_counts = sum(a_counts[-3:])
    late_a_total = sum(total_a[-3:])
    early_a_lo, early_a_hi = wilson_ci(early_a_counts, early_a_total)
    late_a_lo, late_a_hi = wilson_ci(late_a_counts, late_a_total)
    early_a_rate = early_a_counts / early_a_total * 100
    late_a_rate = late_a_counts / late_a_total * 100
    print(f"\n  Task A early (iter 1-3):  {early_a_counts}/{early_a_total} = {early_a_rate:.1f}%  CI [{early_a_lo:.1f}%, {early_a_hi:.1f}%]")
    print(f"  Task A late  (iter 8-10): {late_a_counts}/{late_a_total} = {late_a_rate:.1f}%  CI [{late_a_lo:.1f}%, {late_a_hi:.1f}%]")
    if late_a_lo > early_a_hi:
        print(f"  → CIs do NOT overlap → Task A improvement is statistically significant (p<0.05)")
    elif late_a_hi < early_a_lo:
        print(f"  → CIs do NOT overlap → Task A actually declined significantly")
    else:
        print(f"  → CIs overlap → Task A improvement is NOT statistically significant at 95% level")


if __name__ == "__main__":
    main()
