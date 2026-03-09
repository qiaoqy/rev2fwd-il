#!/bin/bash
# =============================================================================
# Fair Test All Checkpoints with Initial Position Noise
# =============================================================================
#
# Tests all saved checkpoints from an experiment directory.
# Task A checkpoints run on GPU_A, Task B checkpoints run on GPU_B, in parallel.
#
# Each checkpoint is tested for NUM_EPISODES episodes with INIT_NOISE (5cm)
# random perturbation to the initial object position, matching the noise
# distribution introduced by time-reversing trajectories collected with the
# same distance threshold.
#
# Usage:
#   bash scripts/scripts_pick_place/run_fair_test_checkpoints.sh
#
# Output (saved under EXP_DIR):
#   noisy_test_task_A.json   — Per-checkpoint Task A results
#   noisy_test_task_B.json   — Per-checkpoint Task B results
#   noisy_test_record.json   — Combined record for plotting
#   noisy_test_curve.png     — Success rate plot
#
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
EXP_DIR="data/pick_place_isaac_lab_simulation/exp4"
NUM_EPISODES=50
INIT_NOISE=0.05              # 5cm noise radius (matches distance_threshold)
HORIZON=400
DISTANCE_THRESHOLD=0.05
N_ACTION_STEPS=16
GOAL_X=0.5
GOAL_Y=0.0

GPU_A=0                      # GPU for Task A testing
GPU_B=2                      # GPU for Task B testing

HEADLESS="--headless"

# =============================================================================
# Helpers
# =============================================================================

add_timestamps() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
    done
}

print_header() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
}

# =============================================================================
# Discover checkpoints
# =============================================================================
print_header "Discovering Checkpoints in $EXP_DIR"

CKPTS_A=()
LABELS_A=()
CKPTS_B=()
LABELS_B=()

for iter_dir in $(ls -d "$EXP_DIR"/iter*_ckpt_A 2>/dev/null | sort -V); do
    iter_name=$(basename "$iter_dir")
    # Extract iteration number: iter3_ckpt_A → 3
    iter_num=${iter_name#iter}
    iter_num=${iter_num%_ckpt_A}
    CKPTS_A+=("$iter_dir")
    LABELS_A+=("iter${iter_num}")
done

for iter_dir in $(ls -d "$EXP_DIR"/iter*_ckpt_B 2>/dev/null | sort -V); do
    iter_name=$(basename "$iter_dir")
    iter_num=${iter_name#iter}
    iter_num=${iter_num%_ckpt_B}
    CKPTS_B+=("$iter_dir")
    LABELS_B+=("iter${iter_num}")
done

echo "  Task A checkpoints: ${#CKPTS_A[@]}"
for i in "${!CKPTS_A[@]}"; do
    echo "    ${LABELS_A[$i]}: ${CKPTS_A[$i]}"
done

echo "  Task B checkpoints: ${#CKPTS_B[@]}"
for i in "${!CKPTS_B[@]}"; do
    echo "    ${LABELS_B[$i]}: ${CKPTS_B[$i]}"
done

if [ ${#CKPTS_A[@]} -eq 0 ] && [ ${#CKPTS_B[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in $EXP_DIR"
    exit 1
fi

# =============================================================================
# Output paths
# =============================================================================
OUT_A="$EXP_DIR/noisy_test_task_A.json"
OUT_B="$EXP_DIR/noisy_test_task_B.json"
RECORD="$EXP_DIR/noisy_test_record.json"
PLOT="$EXP_DIR/noisy_test_curve.png"
LOG_A="$EXP_DIR/noisy_test_task_A.log"
LOG_B="$EXP_DIR/noisy_test_task_B.log"

# =============================================================================
# Launch parallel tests
# =============================================================================
print_header "Launching Parallel Tests"
echo "  GPU $GPU_A: Task A (${#CKPTS_A[@]} checkpoints × $NUM_EPISODES episodes)"
echo "  GPU $GPU_B: Task B (${#CKPTS_B[@]} checkpoints × $NUM_EPISODES episodes)"
echo "  Init noise: ${INIT_NOISE}m"
echo ""

PID_A=""
PID_B=""

# ---- Task A on GPU_A ----
if [ ${#CKPTS_A[@]} -gt 0 ]; then
    echo "  [GPU $GPU_A] Starting Task A evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU_A python scripts/scripts_pick_place/11_eval_single_task.py \
        --task_type A \
        --checkpoints "${CKPTS_A[@]}" \
        --labels "${LABELS_A[@]}" \
        --num_episodes $NUM_EPISODES \
        --init_noise $INIT_NOISE \
        --horizon $HORIZON \
        --distance_threshold $DISTANCE_THRESHOLD \
        --n_action_steps $N_ACTION_STEPS \
        --goal_xy $GOAL_X $GOAL_Y \
        --out "$OUT_A" \
        $HEADLESS \
        > >(add_timestamps | tee "$LOG_A") 2>&1 &
    PID_A=$!
    echo "  [GPU $GPU_A] Task A PID: $PID_A"
fi

# ---- Task B on GPU_B ----
if [ ${#CKPTS_B[@]} -gt 0 ]; then
    echo "  [GPU $GPU_B] Starting Task B evaluation..."
    CUDA_VISIBLE_DEVICES=$GPU_B python scripts/scripts_pick_place/11_eval_single_task.py \
        --task_type B \
        --checkpoints "${CKPTS_B[@]}" \
        --labels "${LABELS_B[@]}" \
        --num_episodes $NUM_EPISODES \
        --init_noise $INIT_NOISE \
        --horizon $HORIZON \
        --distance_threshold $DISTANCE_THRESHOLD \
        --n_action_steps $N_ACTION_STEPS \
        --goal_xy $GOAL_X $GOAL_Y \
        --out "$OUT_B" \
        $HEADLESS \
        > >(add_timestamps | tee "$LOG_B") 2>&1 &
    PID_B=$!
    echo "  [GPU $GPU_B] Task B PID: $PID_B"
fi

# ---- Wait for both ----
echo ""
echo "  Waiting for both tasks to complete..."
echo "  Monitor logs:"
echo "    tail -f $LOG_A"
echo "    tail -f $LOG_B"

set +e
RC_A=0
RC_B=0

if [ -n "$PID_A" ]; then
    wait $PID_A
    RC_A=$?
    if [ $RC_A -eq 0 ]; then
        echo "  ✓ Task A evaluation finished successfully"
    else
        echo "  ✗ Task A evaluation FAILED (exit code $RC_A)"
        echo "    See: $LOG_A"
    fi
fi

if [ -n "$PID_B" ]; then
    wait $PID_B
    RC_B=$?
    if [ $RC_B -eq 0 ]; then
        echo "  ✓ Task B evaluation finished successfully"
    else
        echo "  ✗ Task B evaluation FAILED (exit code $RC_B)"
        echo "    See: $LOG_B"
    fi
fi
set -e

if [ $RC_A -ne 0 ] || [ $RC_B -ne 0 ]; then
    echo "ERROR: One or more evaluations failed."
    exit 1
fi

# =============================================================================
# Merge results into a unified record for plotting
# =============================================================================
print_header "Merging Results"

python3 << PYEOF
import json
from pathlib import Path

exp_dir = Path("$EXP_DIR")
out_a_path = exp_dir / "noisy_test_task_A.json"
out_b_path = exp_dir / "noisy_test_task_B.json"
record_path = exp_dir / "noisy_test_record.json"

results_a = {}
data_a = {}
if out_a_path.exists():
    with open(out_a_path) as f:
        data_a = json.load(f)
    for r in data_a["results"]:
        results_a[r["label"]] = r

results_b = {}
data_b = {}
if out_b_path.exists():
    with open(out_b_path) as f:
        data_b = json.load(f)
    for r in data_b["results"]:
        results_b[r["label"]] = r

all_labels = sorted(set(list(results_a.keys()) + list(results_b.keys())),
                    key=lambda s: int(s.replace("iter", "")))

iterations = []
for label in all_labels:
    iter_num = int(label.replace("iter", ""))
    ra = results_a.get(label, {})
    rb = results_b.get(label, {})
    entry = {
        "iteration": iter_num,
        "checkpoint_info": {
            "policy_A_checkpoint": ra.get("checkpoint", ""),
            "policy_B_checkpoint": rb.get("checkpoint", ""),
        },
        "noisy_test_metrics": {
            "task_A_success_rate": ra.get("success_rate", 0),
            "task_B_success_rate": rb.get("success_rate", 0),
            "task_A_success_count": ra.get("success_count", 0),
            "task_B_success_count": rb.get("success_count", 0),
            "total_task_A_episodes": ra.get("total_episodes", 0),
            "total_task_B_episodes": rb.get("total_episodes", 0),
            "avg_success_step_A": ra.get("avg_success_step"),
            "avg_success_step_B": rb.get("avg_success_step"),
            "total_elapsed_seconds": ra.get("elapsed_seconds", 0) + rb.get("elapsed_seconds", 0),
        },
    }
    iterations.append(entry)

cfg = data_a.get("config", data_b.get("config", {}))
noise_val = cfg.get("init_noise", 0.05)
record = {
    "description": f"Noisy test evaluation (init_noise={noise_val}m)",
    "config": cfg,
    "iterations": iterations,
}

with open(record_path, "w") as f:
    json.dump(record, f, indent=2)

print(f"  Merged {len(iterations)} iterations into {record_path}")
print()
hdr = f"  {'Iter':>4}  {'A Rate':>8}  {'A (ok/tot)':>12}  {'B Rate':>8}  {'B (ok/tot)':>12}"
print(hdr)
print(f"  {'-'*52}")
for it in iterations:
    m = it["noisy_test_metrics"]
    print(f"  {it['iteration']:>4}  {m['task_A_success_rate']*100:>7.1f}%  "
          f"{m['task_A_success_count']:>4}/{m['total_task_A_episodes']:<4}    "
          f"{m['task_B_success_rate']*100:>7.1f}%  "
          f"{m['task_B_success_count']:>4}/{m['total_task_B_episodes']:<4}")
PYEOF

# =============================================================================
# Generate plot
# =============================================================================
print_header "Generating Plot"

python scripts/scripts_pick_place/plot_success_rate.py \
    --record "$RECORD" \
    --out "$PLOT" \
    --metrics_key noisy_test_metrics

echo "  Plot saved to: $PLOT"

# =============================================================================
# Done
# =============================================================================
print_header "Fair Test Complete!"
echo "  Results:"
echo "    Task A: $OUT_A"
echo "    Task B: $OUT_B"
echo "    Record: $RECORD"
echo "    Plot:   $PLOT"
echo "    Logs:   $LOG_A / $LOG_B"
echo ""
