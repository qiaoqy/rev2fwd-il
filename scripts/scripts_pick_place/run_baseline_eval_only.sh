#!/bin/bash
# =============================================================================
# Baseline Eval ONLY — polls for checkpoints and evaluates on GPU 0
# =============================================================================
#
# Runs in parallel with run_baseline_train_only.sh.
# Polls every 10 minutes for staged checkpoints, then evaluates.
# Iteration 1 is already done (skipped).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/scripts_pick_place/run_baseline_eval_only.sh 2>&1 | tee data/baseline_eval.log
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
START_EVAL_ITER=2           # Iteration 1 already done
END_EVAL_ITER=10
NUM_CYCLES=50
HORIZON=400
DISTANCE_THRESHOLD=0.05
N_ACTION_STEPS=16
POLL_INTERVAL=600           # 10 minutes in seconds

GOAL_X=0.5
GOAL_Y=0.0

# Staging directory (written by train script)
STAGING_DIR="data/baseline_checkpoints"

# Eval GPU
EVAL_GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "Eval GPU: $EVAL_GPU"
echo "Poll interval: ${POLL_INTERVAL}s ($(( POLL_INTERVAL / 60 ))min)"

HEADLESS="--headless"
SAVE_VIDEO=""

RECORD_FILE="data/success_rate_baseline_record.json"

# =============================================================================
# Helper Functions
# =============================================================================
print_header() { echo ""; echo "=============================================="; echo "$1"; echo "=============================================="; }
print_section() { echo ""; echo "----------------------------------------------"; echo "$1"; echo "----------------------------------------------"; }

append_record() {
    local iteration=$1 stats_file=$2 ckpt_step_A=$3 ckpt_step_B=$4
    python3 << PYEOF
import json
from pathlib import Path
with open("$RECORD_FILE", 'r') as f:
    record = json.load(f)
rollout_stats = None
if Path("$stats_file").exists():
    with open("$stats_file", 'r') as f:
        rollout_stats = json.load(f)
entry = {"iteration": $iteration, "checkpoint_info": {"policy_A_training_step": $ckpt_step_A, "policy_B_training_step": $ckpt_step_B}, "performance_metrics": {}}
if rollout_stats:
    summary = rollout_stats.get("summary", {})
    entry["performance_metrics"] = {
        "task_A_success_rate": summary.get("task_A_success_rate", 0),
        "task_B_success_rate": summary.get("task_B_success_rate", 0),
        "task_A_success_count": summary.get("task_A_success_count", 0),
        "task_B_success_count": summary.get("task_B_success_count", 0),
        "total_task_A_episodes": summary.get("total_task_A_episodes", 0),
        "total_task_B_episodes": summary.get("total_task_B_episodes", 0),
        "total_elapsed_seconds": summary.get("total_elapsed_seconds", 0),
    }
    episodes_A = rollout_stats.get("episodes_A", [])
    episodes_B = rollout_stats.get("episodes_B", [])
    steps_A = [e["success_step"] for e in episodes_A if e.get("success") and e.get("success_step")]
    steps_B = [e["success_step"] for e in episodes_B if e.get("success") and e.get("success_step")]
    entry["performance_metrics"]["avg_success_step_A"] = sum(steps_A)/len(steps_A) if steps_A else None
    entry["performance_metrics"]["avg_success_step_B"] = sum(steps_B)/len(steps_B) if steps_B else None
else:
    entry["performance_metrics"] = {"task_A_success_rate": 0, "task_B_success_rate": 0, "task_A_success_count": 0, "task_B_success_count": 0, "total_task_A_episodes": 0, "total_task_B_episodes": 0}
record["iterations"].append(entry)
record["total_iterations_completed"] = len(record["iterations"])
with open("$RECORD_FILE", 'w') as f:
    json.dump(record, f, indent=2)
a_rate = entry["performance_metrics"]["task_A_success_rate"] * 100
b_rate = entry["performance_metrics"]["task_B_success_rate"] * 100
print(f"  ✓ Recorded iter $iteration: A={a_rate:.1f}% B={b_rate:.1f}%")
PYEOF
}

# =============================================================================
# Pre-flight
# =============================================================================
print_header "Baseline Eval Polling Script"
echo "  Watching: $STAGING_DIR/iter_N/ready"
echo "  Record:   $RECORD_FILE"
echo "  Iterations: $START_EVAL_ITER to $END_EVAL_ITER"

if [ ! -f "$RECORD_FILE" ]; then
    echo "ERROR: Record file not found: $RECORD_FILE"
    echo "  Run the original baseline script first to create iteration 1 results."
    exit 1
fi

# =============================================================================
# Main Eval Loop
# =============================================================================
for eval_iter in $(seq $START_EVAL_ITER $END_EVAL_ITER); do
    print_header "Waiting for Eval Iteration $eval_iter checkpoint..."

    READY_MARKER="$STAGING_DIR/iter_${eval_iter}/ready"
    ITER_DIR="$STAGING_DIR/iter_${eval_iter}"

    # ---- Poll for checkpoint readiness ----
    while [ ! -f "$READY_MARKER" ]; do
        echo "  [$(date '+%H:%M:%S')] Checkpoint not ready yet. Sleeping ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    done

    echo "  [$(date '+%H:%M:%S')] ✓ Checkpoint ready!"

    # Read step info
    STEP_A=$(python3 -c "import json; d=json.load(open('$ITER_DIR/info.json')); print(d['step_A'])")
    STEP_B=$(python3 -c "import json; d=json.load(open('$ITER_DIR/info.json')); print(d['step_B'])")
    echo "  Steps: A=$STEP_A, B=$STEP_B"

    CHECKPOINT_A="$ITER_DIR/policy_A/pretrained_model"
    CHECKPOINT_B="$ITER_DIR/policy_B/pretrained_model"

    # ---- Evaluate ----
    print_section "[Eval Iter $eval_iter] Running $NUM_CYCLES A-B cycles"

    ROLLOUT_A="data/sr_baseline_eval_A_iter${eval_iter}.npz"
    ROLLOUT_B="data/sr_baseline_eval_B_iter${eval_iter}.npz"
    STATS_FILE="${ROLLOUT_A%.npz}.stats.json"

    rm -f "$ROLLOUT_A" "$ROLLOUT_B" "$STATS_FILE"

    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/scripts_pick_place/9_eval_with_recovery.py \
        --policy_A "$CHECKPOINT_A" \
        --policy_B "$CHECKPOINT_B" \
        --out_A "$ROLLOUT_A" \
        --out_B "$ROLLOUT_B" \
        --num_cycles $NUM_CYCLES \
        --horizon $HORIZON \
        --distance_threshold $DISTANCE_THRESHOLD \
        --n_action_steps $N_ACTION_STEPS \
        --goal_xy $GOAL_X $GOAL_Y \
        $SAVE_VIDEO \
        $HEADLESS

    # Record results
    append_record $eval_iter "$STATS_FILE" $STEP_A $STEP_B

    # Print success rates
    if [ -f "$STATS_FILE" ]; then
        python3 -c "
import json
with open('$STATS_FILE','r') as f:
    s = json.load(f)['summary']
print(f\"  Task A: {s['task_A_success_count']}/{s['total_task_A_episodes']} = {s['task_A_success_rate']*100:.1f}%\")
print(f\"  Task B: {s['task_B_success_count']}/{s['total_task_B_episodes']} = {s['task_B_success_rate']*100:.1f}%\")
"
    fi

    print_section "Eval Iteration $eval_iter complete!"
done

# =============================================================================
# Generate Plot
# =============================================================================
print_header "Generating Baseline Success Rate Plot"

python scripts/scripts_pick_place/plot_success_rate.py \
    --record "$RECORD_FILE" \
    --out "data/success_rate_baseline_curve.png"

# =============================================================================
# Summary
# =============================================================================
print_header "All Baseline Evaluations Complete!"
echo "  Record: $RECORD_FILE"
echo "  Plot:   data/success_rate_baseline_curve.png"

python3 << 'PYSUMMARY'
import json
with open("data/success_rate_baseline_record.json", 'r') as f:
    record = json.load(f)
iters = record.get("iterations", [])
if iters:
    print("  BASELINE Results (no new data):")
    print("  Iter  |  Task A  |  Task B")
    print("  ------|----------|--------")
    for it in iters:
        m = it["performance_metrics"]
        print(f"  {it['iteration']:4d}  |  {m['task_A_success_rate']*100:5.1f}%  |  {m['task_B_success_rate']*100:5.1f}%")
PYSUMMARY
