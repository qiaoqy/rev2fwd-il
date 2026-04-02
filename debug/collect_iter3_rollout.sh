#!/bin/bash
# =============================================================================
# Collect 100 iter3 Policy A rollout episodes (keep both success & failure)
# from exp27, using GPUs 0-9, 10 cycles per GPU.
# Output: debug/data/iter3_rollout_A.npz (all episodes merged)
# =============================================================================

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

cd /mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il

# =============================================================================
# Configuration
# =============================================================================

EXP_DIR="data/pick_place_isaac_lab_simulation/exp27"
OUT_DIR="debug/data"
SCRIPT_DIR="scripts/scripts_pick_place_simulator"

CKPT_A="${EXP_DIR}/iter3_ckpt_A"
CKPT_B="${EXP_DIR}/weights/PP_B/checkpoints/checkpoints/last/pretrained_model"

NUM_GPUS=10
NUM_CYCLES_PER_GPU=10
HORIZON=3000
DISTANCE_THRESHOLD=0.03
N_ACTION_STEPS=16
GOAL_X=0.5
GOAL_Y=-0.2
RED_REGION_CENTER_X=0.5
RED_REGION_CENTER_Y=0.2
RED_REGION_SIZE_X=0.3
RED_REGION_SIZE_Y=0.3
SEED_BASE=9000   # different from training seeds

mkdir -p "${OUT_DIR}"

# =============================================================================
# Launch 10 parallel collection instances
# =============================================================================

echo "Collecting iter3 Policy A rollout: ${NUM_GPUS} GPUs × ${NUM_CYCLES_PER_GPU} cycles = $((NUM_GPUS * NUM_CYCLES_PER_GPU)) total"
echo "  Policy A: ${CKPT_A}"
echo "  Policy B: ${CKPT_B}"
echo "  Output dir: ${OUT_DIR}"

PIDS=()
for j in $(seq 0 $((NUM_GPUS - 1))); do
    GPU=$j
    OUT_A="${OUT_DIR}/iter3_rollout_A_p${j}.npz"
    OUT_B="${OUT_DIR}/iter3_rollout_B_p${j}.npz"
    SEED_J=$((SEED_BASE + j))

    echo "  Instance $j: GPU=$GPU, seed=$SEED_J"

    (
        CUDA_VISIBLE_DEVICES=$GPU python -u "${SCRIPT_DIR}/6_eval_cyclic.py" \
            --policy_A "$CKPT_A" \
            --policy_B "$CKPT_B" \
            --out_A "$OUT_A" \
            --out_B "$OUT_B" \
            --num_cycles $NUM_CYCLES_PER_GPU \
            --horizon $HORIZON \
            --distance_threshold $DISTANCE_THRESHOLD \
            --n_action_steps $N_ACTION_STEPS \
            --goal_xy $GOAL_X $GOAL_Y \
            --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
            --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
            --seed $SEED_J \
            --headless \
            --save_all \
            2>&1 | tee "${OUT_DIR}/iter3_collect_p${j}.log"
    ) &
    PIDS+=($!)
    sleep 15
done

echo ""
echo "All ${NUM_GPUS} instances launched. Waiting..."

FAILED=0
for j in $(seq 0 $((NUM_GPUS - 1))); do
    if ! wait ${PIDS[$j]}; then
        echo "ERROR: Instance $j (GPU $j) failed!"
        FAILED=1
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "FATAL: One or more collection instances failed"
    exit 1
fi

echo ""
echo "All instances finished. Merging..."

# =============================================================================
# Merge all parts — keep both success and failure episodes
# =============================================================================

python -u - <<'PYEOF'
import numpy as np
import json
from pathlib import Path

out_dir = Path("debug/data")
num_parts = 10

all_episodes_A = []
all_stats = []

for j in range(num_parts):
    npz_path = out_dir / f"iter3_rollout_A_p{j}.npz"
    stats_path = out_dir / f"iter3_rollout_A_p{j}.stats.json"

    if not npz_path.exists():
        print(f"WARNING: {npz_path} not found, skipping")
        continue

    data = np.load(npz_path, allow_pickle=True)
    episodes = list(data["episodes"])
    n_succ = sum(1 for ep in episodes if ep["success"])
    n_fail = len(episodes) - n_succ
    print(f"  Part {j}: {len(episodes)} episodes ({n_succ} success, {n_fail} fail)")
    all_episodes_A.extend(episodes)

    if stats_path.exists():
        with open(stats_path) as f:
            all_stats.append(json.load(f))

# Overall stats
total_cycles = sum(s["summary"]["total_cycles"] for s in all_stats)
total_success_A = sum(s["summary"]["task_A_success_count"] for s in all_stats)
total_success_B = sum(s["summary"]["task_B_success_count"] for s in all_stats)
n_succ_all = sum(1 for ep in all_episodes_A if ep["success"])
n_fail_all = len(all_episodes_A) - n_succ_all

print(f"\nOverall:")
print(f"  Total cycles: {total_cycles}")
print(f"  Task A success: {total_success_A}/{total_cycles} = {total_success_A/total_cycles:.1%}")
print(f"  Task B success: {total_success_B}/{total_cycles} = {total_success_B/total_cycles:.1%}")
print(f"  Merged episodes: {len(all_episodes_A)} ({n_succ_all} success, {n_fail_all} fail)")

# Save merged (all episodes)
merged_path = out_dir / "iter3_rollout_A_all.npz"
np.savez_compressed(merged_path, episodes=np.array(all_episodes_A, dtype=object))
print(f"  Saved to: {merged_path}")

# Save stats
merged_stats = {
    "total_cycles": total_cycles,
    "task_A_success_count": total_success_A,
    "task_A_success_rate": total_success_A / total_cycles if total_cycles > 0 else 0,
    "task_B_success_count": total_success_B,
    "task_B_success_rate": total_success_B / total_cycles if total_cycles > 0 else 0,
    "num_episodes_total": len(all_episodes_A),
    "num_success": n_succ_all,
    "num_fail": n_fail_all,
    "per_part": [s["summary"] for s in all_stats],
}
stats_path = out_dir / "iter3_rollout_A_all.stats.json"
with open(stats_path, "w") as f:
    json.dump(merged_stats, f, indent=2)
print(f"  Saved stats to: {stats_path}")

PYEOF

echo ""
echo "Done! Results in ${OUT_DIR}/"
