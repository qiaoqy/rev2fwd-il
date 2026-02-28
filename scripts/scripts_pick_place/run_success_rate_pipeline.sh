#!/bin/bash
# =============================================================================
# Success Rate Pipeline: Iterative Evaluate → Train → Evaluate → ...
# =============================================================================
#
# This script orchestrates a loop of:
#   1. Evaluate: Run 50 A-B cycles with failure recovery, record success rates
#   2. Train: Finetune Policy A & B using only the SUCCESSFUL rollout data
#   3. Repeat for 10 iterations
#   4. Plot success rate curves
#
# Key difference from run_iterative_training.sh:
#   - Uses 9_eval_with_recovery.py which does NOT break on failure
#   - Tracks success rate (%) instead of consecutive successes / task cost
#   - Always runs the full num_cycles regardless of failures
#   - Generates a success rate curve plot at the end
#
# Usage:
#   bash scripts/scripts_pick_place/run_success_rate_pipeline.sh
#
# Configuration:
#   Edit the variables in the "Configuration" section below.
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Conda Environment Activation
# =============================================================================
eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
MAX_ITERATIONS=10           # Number of evaluate-train cycles
NUM_CYCLES=50               # A-B cycles per evaluation round
STEPS_PER_ITER=5000         # Training steps per finetuning iteration
HORIZON=400                 # Maximum steps per task attempt
BATCH_SIZE=32               # Training batch size
DISTANCE_THRESHOLD=0.05     # Distance for success detection
N_ACTION_STEPS=16           # Action steps per inference

# Policy directories (MODIFY THESE FOR YOUR SETUP)
# Source checkpoints (read-only, will not be modified)
POLICY_A_DIR="runs/PP_A_circle"
POLICY_B_DIR="runs/PP_B_circle"

# Working directories (created/managed by this script)
POLICY_A_DIR_TEMP="runs/PP_A_success_rate_temp"
POLICY_B_DIR_TEMP="runs/PP_B_success_rate_temp"
POLICY_A_DIR_LAST="runs/PP_A_success_rate_last"
POLICY_B_DIR_LAST="runs/PP_B_success_rate_last"

# Original LeRobot datasets
LEROBOT_A="${POLICY_A_DIR}/lerobot_dataset"
LEROBOT_B="${POLICY_B_DIR}/lerobot_dataset"

# Goal position (plate center)
GOAL_X=0.5
GOAL_Y=0.0

# GPU configuration
DATA_COLLECTION_GPU=0                                      # Single GPU for eval rollout
TRAINING_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"               # Multi-GPU for training
NUM_TRAINING_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

echo "Data collection GPU: $DATA_COLLECTION_GPU"
echo "Training GPUs: $TRAINING_GPUS (total: $NUM_TRAINING_GPUS)"

# Flags
HEADLESS="--headless"
SAVE_VIDEO=""  # Set to "--save_video" to save videos of each eval

# Output record
RECORD_FILE="data/success_rate_record.json"

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

print_section() {
    echo ""
    echo "----------------------------------------------"
    echo "$1"
    echo "----------------------------------------------"
}

get_checkpoint_path() {
    local policy_dir=$1
    echo "${policy_dir}/checkpoints/checkpoints/last/pretrained_model"
}

get_current_step() {
    local policy_dir=$1
    local training_step_file="${policy_dir}/checkpoints/checkpoints/last/training_state/training_step.json"
    if [ -f "$training_step_file" ]; then
        grep -o '"step": *[0-9]*' "$training_step_file" | sed 's/"step": *//'
    else
        echo "0"
    fi
}

init_record() {
    # Back up old record if exists
    if [ -f "$RECORD_FILE" ]; then
        local backup_ts=$(date +%Y%m%d_%H%M%S)
        local backup="${RECORD_FILE%.json}_backup_${backup_ts}.json"
        echo "Backing up old record → $backup"
        mv "$RECORD_FILE" "$backup"
    fi

    local timestamp=$(date -Iseconds)
    python3 -c "
import json
data = {
    'description': 'Success rate tracking for iterative evaluate-train pipeline',
    'created_at': '$timestamp',
    'config': {
        'max_iterations': $MAX_ITERATIONS,
        'num_cycles': $NUM_CYCLES,
        'steps_per_iter': $STEPS_PER_ITER,
        'horizon': $HORIZON,
        'batch_size': $BATCH_SIZE,
        'distance_threshold': $DISTANCE_THRESHOLD,
        'n_action_steps': $N_ACTION_STEPS,
        'policy_A_source': '$POLICY_A_DIR',
        'policy_B_source': '$POLICY_B_DIR',
        'goal_xy': [$GOAL_X, $GOAL_Y],
    },
    'iterations': []
}
with open('$RECORD_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
    echo "✓ Record file initialized: $RECORD_FILE"
}

append_record() {
    # Args: iteration, stats_json, checkpoint_A_step, checkpoint_B_step
    local iteration=$1
    local stats_file=$2
    local ckpt_step_A=$3
    local ckpt_step_B=$4

    python3 << PYEOF
import json
from pathlib import Path

record_file = "$RECORD_FILE"
stats_file = "$stats_file"
iteration = $iteration
ckpt_step_A = $ckpt_step_A
ckpt_step_B = $ckpt_step_B

with open(record_file, 'r') as f:
    record = json.load(f)

rollout_stats = None
if Path(stats_file).exists():
    with open(stats_file, 'r') as f:
        rollout_stats = json.load(f)

entry = {
    "iteration": iteration,
    "checkpoint_info": {
        "policy_A_training_step": ckpt_step_A,
        "policy_B_training_step": ckpt_step_B,
    },
    "performance_metrics": {},
}

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
    # Average success steps
    episodes_A = rollout_stats.get("episodes_A", [])
    episodes_B = rollout_stats.get("episodes_B", [])
    steps_A = [e["success_step"] for e in episodes_A if e.get("success") and e.get("success_step")]
    steps_B = [e["success_step"] for e in episodes_B if e.get("success") and e.get("success_step")]
    entry["performance_metrics"]["avg_success_step_A"] = sum(steps_A)/len(steps_A) if steps_A else None
    entry["performance_metrics"]["avg_success_step_B"] = sum(steps_B)/len(steps_B) if steps_B else None
else:
    entry["performance_metrics"] = {
        "task_A_success_rate": 0,
        "task_B_success_rate": 0,
        "task_A_success_count": 0,
        "task_B_success_count": 0,
        "total_task_A_episodes": 0,
        "total_task_B_episodes": 0,
    }

record["iterations"].append(entry)
record["total_iterations_completed"] = len(record["iterations"])

with open(record_file, 'w') as f:
    json.dump(record, f, indent=2)

a_rate = entry["performance_metrics"]["task_A_success_rate"] * 100
b_rate = entry["performance_metrics"]["task_B_success_rate"] * 100
print(f"  ✓ Recorded iter {iteration}: A={a_rate:.1f}% B={b_rate:.1f}%")
PYEOF
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks"

for dir in "$LEROBOT_A" "$LEROBOT_B"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Dataset not found: $dir"
        exit 1
    fi
done

CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR")
CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR")

for ckpt in "$CHECKPOINT_A" "$CHECKPOINT_B"; do
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        exit 1
    fi
done

echo "✓ Datasets and checkpoints found"

# =============================================================================
# Configuration Summary
# =============================================================================
print_header "Configuration Summary"
echo "  MAX_ITERATIONS:     $MAX_ITERATIONS"
echo "  NUM_CYCLES:         $NUM_CYCLES (per evaluation)"
echo "  STEPS_PER_ITER:     $STEPS_PER_ITER"
echo "  HORIZON:            $HORIZON"
echo "  BATCH_SIZE:         $BATCH_SIZE"
echo "  DISTANCE_THRESHOLD: $DISTANCE_THRESHOLD"
echo "  N_ACTION_STEPS:     $N_ACTION_STEPS"
echo ""
echo "  POLICY_A_DIR:       $POLICY_A_DIR"
echo "  POLICY_B_DIR:       $POLICY_B_DIR"
echo "  GOAL:               ($GOAL_X, $GOAL_Y)"

# =============================================================================
# Initialize Record
# =============================================================================
print_header "Initializing Record"
init_record

# =============================================================================
# Backup Old Working Directories
# =============================================================================
print_header "Backing Up Old Working Directories"
BACKUP_TS=$(date +%Y%m%d_%H%M%S)
for dir in "$POLICY_A_DIR_TEMP" "$POLICY_A_DIR_LAST" "$POLICY_B_DIR_TEMP" "$POLICY_B_DIR_LAST"; do
    if [ -d "$dir" ]; then
        backup="${dir}_backup_${BACKUP_TS}"
        echo "  $dir → $backup"
        mv "$dir" "$backup"
    fi
done
echo "✓ Done"

# =============================================================================
# Initialize: Copy Origin → Temp
# =============================================================================
print_header "Initializing Working Directories"

echo "  Policy A: origin → temp"
mkdir -p "$POLICY_A_DIR_TEMP"
cp -r "$POLICY_A_DIR/checkpoints" "$POLICY_A_DIR_TEMP/checkpoints"
cp -r "$LEROBOT_A" "$POLICY_A_DIR_TEMP/lerobot_dataset"

echo "  Policy B: origin → temp"
mkdir -p "$POLICY_B_DIR_TEMP"
cp -r "$POLICY_B_DIR/checkpoints" "$POLICY_B_DIR_TEMP/checkpoints"
cp -r "$LEROBOT_B" "$POLICY_B_DIR_TEMP/lerobot_dataset"

echo "✓ Initialized"

# =============================================================================
# Main Loop
# =============================================================================
for iter in $(seq 1 $MAX_ITERATIONS); do
    print_header "========== Iteration $iter / $MAX_ITERATIONS =========="

    # Output paths for this iteration
    ROLLOUT_A="data/sr_eval_A_iter${iter}.npz"
    ROLLOUT_B="data/sr_eval_B_iter${iter}.npz"
    STATS_FILE="${ROLLOUT_A%.npz}.stats.json"

    # Current checkpoints (from temp = previous iteration's output)
    CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")
    STEP_A=$(get_current_step "$POLICY_A_DIR_TEMP")
    STEP_B=$(get_current_step "$POLICY_B_DIR_TEMP")

    # =====================================================================
    # Step 1: EVALUATE — run 50 A-B cycles with failure recovery
    # =====================================================================
    print_section "[Step 1] Evaluating (${NUM_CYCLES} A-B cycles with recovery)"
    echo "  Checkpoint A (step $STEP_A): $CHECKPOINT_A"
    echo "  Checkpoint B (step $STEP_B): $CHECKPOINT_B"

    rm -f "$ROLLOUT_A" "$ROLLOUT_B" "$STATS_FILE"

    CUDA_VISIBLE_DEVICES=$DATA_COLLECTION_GPU python scripts/scripts_pick_place/9_eval_with_recovery.py \
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
    append_record $iter "$STATS_FILE" $STEP_A $STEP_B

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

    # =====================================================================
    # Step 2: TRAIN Policy A (finetune with successful rollout data)
    # =====================================================================
    print_section "[Step 2] Finetuning Policy A ($STEPS_PER_ITER steps)"

    rm -rf "$POLICY_A_DIR_LAST"
    mkdir -p "$POLICY_A_DIR_LAST"

    # ---- 2a: Prepare dataset (merge rollout data if available) and copy checkpoint ----
    if [ -f "$ROLLOUT_A" ]; then
        echo "  [2a] Merging rollout data into dataset + copying checkpoint..."
        python scripts/scripts_pick_place/7_finetune_with_rollout.py \
            --original_lerobot "$POLICY_A_DIR_TEMP/lerobot_dataset" \
            --rollout_data "$ROLLOUT_A" \
            --checkpoint "$CHECKPOINT_A" \
            --out "$POLICY_A_DIR_LAST" \
            --prepare_only \
            --include_obj_pose \
            --include_gripper
        echo "  ✓ Rollout data merged and checkpoint copied for Policy A"
    else
        echo "  [2a] No Task A rollout data — copying original dataset and checkpoint"
        python scripts/scripts_pick_place/7_finetune_with_rollout.py \
            --original_lerobot "$POLICY_A_DIR_TEMP/lerobot_dataset" \
            --checkpoint "$CHECKPOINT_A" \
            --out "$POLICY_A_DIR_LAST" \
            --prepare_only \
            --include_obj_pose \
            --include_gripper
    fi

    # ---- 2b: Copy wandb directory for resumed logging ----
    if [ -d "$POLICY_A_DIR_TEMP/checkpoints/wandb" ]; then
        echo "  [2b] Copying wandb directory..."
        cp -r "$POLICY_A_DIR_TEMP/checkpoints/wandb" "$POLICY_A_DIR_LAST/checkpoints/wandb"
        WANDB_LATEST="$POLICY_A_DIR_LAST/checkpoints/wandb/latest-run"
        if [ -d "$WANDB_LATEST" ]; then
            for mf in "$WANDB_LATEST"/files/wandb-metadata.json; do
                [ -f "$mf" ] && sed -i "s|$POLICY_A_DIR_TEMP|$POLICY_A_DIR_LAST|g" "$mf"
            done
        fi
    fi

    # ---- 2b2: Convert last dir to symlink format ----
    CKPT_DIR="$POLICY_A_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        CUR_STEP=$(get_current_step "$POLICY_A_DIR_LAST")
        STEP_NAME=$(printf "%06d" $CUR_STEP)
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  ✓ Converted: last → $STEP_NAME (symlink)"
    fi

    # ---- 2c: Train ----
    CUR_STEP_A=$(get_current_step "$POLICY_A_DIR_LAST")
    TARGET_A=$((CUR_STEP_A + STEPS_PER_ITER))
    echo "  [2c] Training: step $CUR_STEP_A → $TARGET_A ($NUM_TRAINING_GPUS GPUs)"

    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS \
        scripts/scripts_pick_place/4_train_diffusion.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$POLICY_A_DIR_LAST/lerobot_dataset" \
        --out "$POLICY_A_DIR_LAST" \
        --steps $TARGET_A \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose \
        --include_gripper \
        --wandb

    echo "  ✓ Policy A finetuned"

    # ---- 2d: Rotate directories ----
    rm -rf "$POLICY_A_DIR_TEMP"
    mv "$POLICY_A_DIR_LAST" "$POLICY_A_DIR_TEMP"
    echo "  ✓ last → temp"

    # =====================================================================
    # Step 3: TRAIN Policy B (finetune with successful rollout data)
    # =====================================================================
    print_section "[Step 3] Finetuning Policy B ($STEPS_PER_ITER steps)"

    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")

    rm -rf "$POLICY_B_DIR_LAST"
    mkdir -p "$POLICY_B_DIR_LAST"

    # ---- 3a: Prepare dataset (merge rollout data if available) and copy checkpoint ----
    if [ -f "$ROLLOUT_B" ]; then
        echo "  [3a] Merging rollout data into dataset + copying checkpoint..."
        python scripts/scripts_pick_place/7_finetune_with_rollout.py \
            --original_lerobot "$POLICY_B_DIR_TEMP/lerobot_dataset" \
            --rollout_data "$ROLLOUT_B" \
            --checkpoint "$CHECKPOINT_B" \
            --out "$POLICY_B_DIR_LAST" \
            --prepare_only \
            --include_obj_pose \
            --include_gripper
        echo "  ✓ Rollout data merged and checkpoint copied for Policy B"
    else
        echo "  [3a] No Task B rollout data — copying original dataset and checkpoint"
        python scripts/scripts_pick_place/7_finetune_with_rollout.py \
            --original_lerobot "$POLICY_B_DIR_TEMP/lerobot_dataset" \
            --checkpoint "$CHECKPOINT_B" \
            --out "$POLICY_B_DIR_LAST" \
            --prepare_only \
            --include_obj_pose \
            --include_gripper
    fi

    # ---- 3b: Copy wandb directory for resumed logging ----
    if [ -d "$POLICY_B_DIR_TEMP/checkpoints/wandb" ]; then
        echo "  [3b] Copying wandb directory..."
        cp -r "$POLICY_B_DIR_TEMP/checkpoints/wandb" "$POLICY_B_DIR_LAST/checkpoints/wandb"
        WANDB_LATEST="$POLICY_B_DIR_LAST/checkpoints/wandb/latest-run"
        if [ -d "$WANDB_LATEST" ]; then
            for mf in "$WANDB_LATEST"/files/wandb-metadata.json; do
                [ -f "$mf" ] && sed -i "s|$POLICY_B_DIR_TEMP|$POLICY_B_DIR_LAST|g" "$mf"
            done
        fi
    fi

    # ---- 3b2: Convert last dir to symlink format ----
    CKPT_DIR="$POLICY_B_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        CUR_STEP=$(get_current_step "$POLICY_B_DIR_LAST")
        STEP_NAME=$(printf "%06d" $CUR_STEP)
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  ✓ Converted: last → $STEP_NAME (symlink)"
    fi

    # ---- 3c: Train ----
    CUR_STEP_B=$(get_current_step "$POLICY_B_DIR_LAST")
    TARGET_B=$((CUR_STEP_B + STEPS_PER_ITER))
    echo "  [3c] Training: step $CUR_STEP_B → $TARGET_B ($NUM_TRAINING_GPUS GPUs)"

    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS \
        scripts/scripts_pick_place/4_train_diffusion.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$POLICY_B_DIR_LAST/lerobot_dataset" \
        --out "$POLICY_B_DIR_LAST" \
        --steps $TARGET_B \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose \
        --include_gripper \
        --wandb

    echo "  ✓ Policy B finetuned"

    # ---- 3d: Rotate directories ----
    rm -rf "$POLICY_B_DIR_TEMP"
    mv "$POLICY_B_DIR_LAST" "$POLICY_B_DIR_TEMP"
    echo "  ✓ last → temp"

    print_section "Iteration $iter complete!"
done

# =============================================================================
# Generate Success Rate Plot
# =============================================================================
print_header "Generating Success Rate Plot"

python scripts/scripts_pick_place/plot_success_rate.py \
    --record "$RECORD_FILE" \
    --out "data/success_rate_curve.png"

# =============================================================================
# Summary
# =============================================================================
print_header "Pipeline Complete!"
echo ""
echo "  Record file:  $RECORD_FILE"
echo "  Plot:         data/success_rate_curve.png"
echo ""
echo "  Final checkpoints (in temp dirs):"
echo "    Policy A: $(get_checkpoint_path "$POLICY_A_DIR_TEMP")"
echo "    Policy B: $(get_checkpoint_path "$POLICY_B_DIR_TEMP")"
echo ""

# Print final summary from record
python3 << 'PYSUMMARY'
import json
with open("data/success_rate_record.json", 'r') as f:
    record = json.load(f)
iters = record.get("iterations", [])
if iters:
    print("  Iter  |  Task A  |  Task B")
    print("  ------|----------|--------")
    for it in iters:
        m = it["performance_metrics"]
        print(f"  {it['iteration']:4d}  |  {m['task_A_success_rate']*100:5.1f}%  |  {m['task_B_success_rate']*100:5.1f}%")
    first_a = iters[0]["performance_metrics"]["task_A_success_rate"]*100
    last_a = iters[-1]["performance_metrics"]["task_A_success_rate"]*100
    first_b = iters[0]["performance_metrics"]["task_B_success_rate"]*100
    last_b = iters[-1]["performance_metrics"]["task_B_success_rate"]*100
    print(f"\n  Task A: {first_a:.1f}% → {last_a:.1f}%")
    print(f"  Task B: {first_b:.1f}% → {last_b:.1f}%")
PYSUMMARY
