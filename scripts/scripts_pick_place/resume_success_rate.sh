#!/bin/bash
# Resume success rate pipeline from iteration 8 (after NCCL crash recovery)
set -e

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration (same as original)
# =============================================================================
MAX_ITERATIONS=10
START_ITERATION=8           # Resume from here
NUM_CYCLES=50
STEPS_PER_ITER=5000
HORIZON=400
BATCH_SIZE=32
DISTANCE_THRESHOLD=0.05
N_ACTION_STEPS=16

POLICY_A_DIR="runs/PP_A_circle"
POLICY_B_DIR="runs/PP_B_circle"
POLICY_A_DIR_TEMP="runs/PP_A_success_rate_temp"
POLICY_B_DIR_TEMP="runs/PP_B_success_rate_temp"
POLICY_A_DIR_LAST="runs/PP_A_success_rate_last"
POLICY_B_DIR_LAST="runs/PP_B_success_rate_last"
LEROBOT_A="${POLICY_A_DIR}/lerobot_dataset"
LEROBOT_B="${POLICY_B_DIR}/lerobot_dataset"
GOAL_X=0.5
GOAL_Y=0.0

# GPU configuration - use GPU 0 for data collection, 4,5,6 for training
DATA_COLLECTION_GPU=0
TRAINING_GPUS="4,5,6"
NUM_TRAINING_GPUS=3

# NCCL robustness settings
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

echo "Data collection GPU: $DATA_COLLECTION_GPU"
echo "Training GPUs: $TRAINING_GPUS (total: $NUM_TRAINING_GPUS)"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

HEADLESS="--headless"
SAVE_VIDEO=""
RECORD_FILE="data/success_rate_record.json"

# =============================================================================
# Helper Functions
# =============================================================================
print_header() { echo ""; echo "=============================================="; echo "$1"; echo "=============================================="; }
print_section() { echo ""; echo "----------------------------------------------"; echo "$1"; echo "----------------------------------------------"; }

get_checkpoint_path() { echo "${1}/checkpoints/checkpoints/last/pretrained_model"; }

get_current_step() {
    local f="${1}/checkpoints/checkpoints/last/training_state/training_step.json"
    if [ -f "$f" ]; then grep -o '"step": *[0-9]*' "$f" | sed 's/"step": *//'; else echo "0"; fi
}

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
# Pre-flight: verify temp dirs exist
# =============================================================================
print_header "Resume Pre-flight Checks"
CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")
echo "  Checkpoint A: $CHECKPOINT_A"
echo "  Checkpoint B: $CHECKPOINT_B"
ls "$CHECKPOINT_A/config.json" > /dev/null
ls "$CHECKPOINT_B/config.json" > /dev/null
echo "✓ Temp directories and checkpoints intact"
echo "  Step A: $(get_current_step "$POLICY_A_DIR_TEMP")"
echo "  Step B: $(get_current_step "$POLICY_B_DIR_TEMP")"

# =============================================================================
# Main Loop (resume)
# =============================================================================
for iter in $(seq $START_ITERATION $MAX_ITERATIONS); do
    print_header "========== Iteration $iter / $MAX_ITERATIONS (RESUMED) =========="

    ROLLOUT_A="data/sr_eval_A_iter${iter}.npz"
    ROLLOUT_B="data/sr_eval_B_iter${iter}.npz"
    STATS_FILE="${ROLLOUT_A%.npz}.stats.json"

    CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")
    STEP_A=$(get_current_step "$POLICY_A_DIR_TEMP")
    STEP_B=$(get_current_step "$POLICY_B_DIR_TEMP")

    # =====================================================================
    # Step 1: EVALUATE
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

    append_record $iter "$STATS_FILE" $STEP_A $STEP_B

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
    # Step 2: TRAIN Policy A
    # =====================================================================
    print_section "[Step 2] Finetuning Policy A ($STEPS_PER_ITER steps)"

    rm -rf "$POLICY_A_DIR_LAST"
    mkdir -p "$POLICY_A_DIR_LAST"

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

    CKPT_DIR="$POLICY_A_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        CUR_STEP=$(get_current_step "$POLICY_A_DIR_LAST")
        STEP_NAME=$(printf "%06d" $CUR_STEP)
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  ✓ Converted: last → $STEP_NAME (symlink)"
    fi

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
    rm -rf "$POLICY_A_DIR_TEMP"
    mv "$POLICY_A_DIR_LAST" "$POLICY_A_DIR_TEMP"
    echo "  ✓ last → temp"

    # =====================================================================
    # Step 3: TRAIN Policy B
    # =====================================================================
    print_section "[Step 3] Finetuning Policy B ($STEPS_PER_ITER steps)"
    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")

    rm -rf "$POLICY_B_DIR_LAST"
    mkdir -p "$POLICY_B_DIR_LAST"

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

    CKPT_DIR="$POLICY_B_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        CUR_STEP=$(get_current_step "$POLICY_B_DIR_LAST")
        STEP_NAME=$(printf "%06d" $CUR_STEP)
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  ✓ Converted: last → $STEP_NAME (symlink)"
    fi

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
echo "  Record file:  $RECORD_FILE"
echo "  Plot:         data/success_rate_curve.png"

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
PYSUMMARY
