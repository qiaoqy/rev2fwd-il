#!/usr/bin/env python3
"""Merge multiple fair test stats JSON files from parallel GPU runs into one.

Usage:
    python merge_fair_test_stats.py \
        --inputs shard0.stats.json shard1.stats.json ... \
        --out merged.stats.json
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge parallel fair test stats")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard stats JSON files")
    parser.add_argument("--out", required=True, help="Output merged stats JSON path")
    args = parser.parse_args()

    all_episodes_A = []
    all_episodes_B = []
    total_elapsed_A = 0.0
    total_elapsed_B = 0.0
    config = None

    for path in sorted(args.inputs):
        with open(path) as f:
            data = json.load(f)
        if config is None:
            config = data.get("config", {})
        all_episodes_A.extend(data.get("episodes_A", []))
        all_episodes_B.extend(data.get("episodes_B", []))
        s = data.get("summary", {})
        total_elapsed_A += s.get("task_A_elapsed_seconds", 0)
        total_elapsed_B += s.get("task_B_elapsed_seconds", 0)

    # Re-index episodes
    for i, ep in enumerate(all_episodes_A):
        ep["episode_index"] = i
    for i, ep in enumerate(all_episodes_B):
        ep["episode_index"] = i

    n_success_A = sum(1 for ep in all_episodes_A if ep.get("success"))
    n_success_B = sum(1 for ep in all_episodes_B if ep.get("success"))
    n_total_A = len(all_episodes_A)
    n_total_B = len(all_episodes_B)
    rate_A = n_success_A / n_total_A if n_total_A > 0 else 0.0
    rate_B = n_success_B / n_total_B if n_total_B > 0 else 0.0

    steps_A = [ep["success_step"] for ep in all_episodes_A
               if ep.get("success") and ep.get("success_step")]
    steps_B = [ep["success_step"] for ep in all_episodes_B
               if ep.get("success") and ep.get("success_step")]

    merged = {
        "experiment": "independent_evaluation",
        "description": f"Merged from {len(args.inputs)} parallel shards.",
        "config": config,
        "summary": {
            "task_A_success_count": n_success_A,
            "task_A_total_episodes": n_total_A,
            "task_A_success_rate": rate_A,
            "avg_success_step_A": (sum(steps_A) / len(steps_A)) if steps_A else None,
            "task_A_elapsed_seconds": total_elapsed_A,
            "task_B_success_count": n_success_B,
            "task_B_total_episodes": n_total_B,
            "task_B_success_rate": rate_B,
            "avg_success_step_B": (sum(steps_B) / len(steps_B)) if steps_B else None,
            "task_B_elapsed_seconds": total_elapsed_B,
            "total_elapsed_seconds": total_elapsed_A + total_elapsed_B,
        },
        "episodes_A": all_episodes_A,
        "episodes_B": all_episodes_B,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(args.inputs)} shards → {out_path}")
    print(f"  Task A: {n_success_A}/{n_total_A} = {rate_A:.1%}")
    print(f"  Task B: {n_success_B}/{n_total_B} = {rate_B:.1%}")


if __name__ == "__main__":
    main()
