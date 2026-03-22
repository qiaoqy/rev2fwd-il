#!/bin/bash
# =============================================================================
# Rev2Fwd Pick-and-Place Simulator Pipeline (Auto-Resumable)
# =============================================================================
#
# 概述:
#   端到端实验编排脚本. 流程:
#     Phase 1: 数据准备 (收集 Task B → 时间反转 → 格式转换)
#     Phase 2: 初始训练 Policy A/B (并行多GPU)
#     Phase 3: 迭代 DAgger (cyclic 评估 → 数据聚合 → finetune)
#     Phase 4: 公平测试
#     Phase 5: 失效分析 (按需手动)
#
# 自动恢复:
#   通过 .done_{phase} 标记文件跟踪进度。崩溃后直接重新运行即可恢复。
#
# 使用示例:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/scripts_pick_place_simulator/run_pipeline.sh
#
# =============================================================================

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================

# --- Experiment ---
EXP_NAME="${EXP_NAME:-exp_new}"
BASE_DIR="data/pick_place_isaac_lab_simulation/${EXP_NAME}"

# --- Region parameters (Mode 3, fixed) ---
GOAL_X="0.5"
GOAL_Y="-0.2"
DISTANCE_THRESHOLD="0.03"
RED_REGION_CENTER_X="0.5"
RED_REGION_CENTER_Y="0.2"
RED_REGION_SIZE_X="0.3"
RED_REGION_SIZE_Y="0.3"

# --- Data collection ---
NUM_COLLECT_EPISODES=100

# --- Training ---
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-4}
TRAIN_N_ACTION_STEPS=16
EVAL_N_ACTION_STEPS=16
TARGET_EPOCHS=500
STEPS_PER_ITER=5000

# --- DDIM ---
NOISE_SCHEDULER_TYPE="DDIM"
NUM_INFERENCE_STEPS=10

# --- Iteration ---
ITER_ROUNDS=10
NUM_CYCLES=50
HORIZON=1200

# --- GPU ---
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
COLLECT_GPU=${COLLECT_GPU:-0}

# Parse individual GPUs for parallel evaluation (use GPUs [1],[2],[3] from the list)
IFS=',' read -ra _GPU_LIST <<< "$TRAIN_GPUS"
EVAL_GPU_1="${_GPU_LIST[1]:-$COLLECT_GPU}"
EVAL_GPU_2="${_GPU_LIST[2]:-$COLLECT_GPU}"
EVAL_GPU_3="${_GPU_LIST[3]:-$COLLECT_GPU}"

# --- NCCL robustness ---
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

# --- Misc ---
HEADLESS="--headless"
WANDB_PROJECT="${WANDB_PROJECT:-rev2fwd-${EXP_NAME}}"
SEED=42

# --- Script paths ---
SCRIPT_DIR="scripts/scripts_pick_place_simulator"

# =============================================================================
# Derived paths
# =============================================================================
LOG_DIR="${BASE_DIR}/logs"
WEIGHTS_DIR="${BASE_DIR}/weights"
LEROBOT_DIR="${BASE_DIR}/lerobot"
WORK_DIR="${BASE_DIR}/work"

TASK_B_NPZ="${BASE_DIR}/task_B_${NUM_COLLECT_EPISODES}.npz"
TASK_A_NPZ="${BASE_DIR}/task_A_reversed_${NUM_COLLECT_EPISODES}.npz"

PA_WEIGHTS="${WEIGHTS_DIR}/PP_A"
PB_WEIGHTS="${WEIGHTS_DIR}/PP_B"

PA_TEMP="${WORK_DIR}/PP_A_temp"
PA_LAST="${WORK_DIR}/PP_A_last"
PB_TEMP="${WORK_DIR}/PP_B_temp"
PB_LAST="${WORK_DIR}/PP_B_last"

RECORD_FILE="${BASE_DIR}/record.json"

# =============================================================================
# Helper Functions
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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_ckpt() {
    echo "${1}/checkpoints/checkpoints/last/pretrained_model"
}

get_step() {
    local f="${1}/checkpoints/checkpoints/last/training_state/training_step.json"
    if [ -f "$f" ]; then
        grep -o '"step": *[0-9]*' "$f" | sed 's/"step": *//'
    else
        echo "0"
    fi
}

phase_done() { [ -f "$BASE_DIR/.done_${1}" ]; }
mark_done()  { touch "$BASE_DIR/.done_${1}"; }

random_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

calc_steps_from_npz() {
    # Usage: calc_steps_from_npz <npz_path> <epochs> <batch_size> <num_gpus>
    python3 -c "
import numpy as np, math, sys
data = np.load('$1', allow_pickle=True)
eps = data['episodes']
total_frames = sum(len(e['images']) for e in eps)
steps = math.ceil(total_frames * $2 / ($3 * $4))
print(steps)
"
}

# Append iteration metrics to record.json
append_record() {
    local iteration=$1
    local stats_file=$2
    local step_A=$3
    local step_B=$4
    local fair_A_file=${5:-}
    local fair_B_file=${6:-}

    python3 << PYEOF
import json
from pathlib import Path

with open("$RECORD_FILE") as f:
    rec = json.load(f)

entry = {
    "iteration": $iteration,
    "checkpoint_info": {"policy_A_step": $step_A, "policy_B_step": $step_B},
    "performance_metrics": {},
    "fair_test_metrics": {},
}

sf = Path("$stats_file")
if sf.exists():
    with open(sf) as f:
        st = json.load(f)
    s = st.get("summary", {})
    pm = entry["performance_metrics"]
    for k in ("task_A_success_rate", "task_B_success_rate",
              "task_A_success_count", "task_B_success_count",
              "total_task_A_episodes", "total_task_B_episodes",
              "total_elapsed_seconds"):
        pm[k] = s.get(k, 0)
else:
    entry["performance_metrics"] = dict(
        task_A_success_rate=0, task_B_success_rate=0,
        task_A_success_count=0, task_B_success_count=0,
        total_task_A_episodes=0, total_task_B_episodes=0,
    )

# Fair test metrics
fm = entry["fair_test_metrics"]
fa = Path("$fair_A_file") if "$fair_A_file" else None
fb = Path("$fair_B_file") if "$fair_B_file" else None
if fa and fa.exists():
    with open(fa) as f:
        d = json.load(f)
    fm["fair_A_success_rate"] = d.get("success_rate", 0)
    fm["fair_A_num_success"] = d.get("num_success", 0)
    fm["fair_A_num_total"] = d.get("num_total", 0)
if fb and fb.exists():
    with open(fb) as f:
        d = json.load(f)
    fm["fair_B_success_rate"] = d.get("success_rate", 0)
    fm["fair_B_num_success"] = d.get("num_success", 0)
    fm["fair_B_num_total"] = d.get("num_total", 0)

# Replace existing entry for this iteration (idempotent on re-run)
rec["iterations"] = [i for i in rec["iterations"] if i["iteration"] != $iteration]
rec["iterations"].append(entry)
rec["iterations"].sort(key=lambda x: x["iteration"])

with open("$RECORD_FILE", "w") as f:
    json.dump(rec, f, indent=2)

pm = entry["performance_metrics"]
fm = entry.get("fair_test_metrics", {})
cyclic_str = f"cyclic A={pm['task_A_success_rate']*100:.1f}% B={pm['task_B_success_rate']*100:.1f}%"
fair_str = ""
if fm:
    fair_str = f"  fair A={fm.get('fair_A_success_rate',0)*100:.1f}% B={fm.get('fair_B_success_rate',0)*100:.1f}%"
print(f"  Recorded iter $iteration: {cyclic_str}{fair_str}")
PYEOF
}

# Resume-safe training wrapper
run_resume_train() {
    # Usage: run_resume_train <name> <temp_dir> <last_dir> <rollout_npz_or_empty> <extra_steps>
    local NAME=$1
    local TEMP=$2
    local LAST=$3
    local ROLLOUT=$4
    local EXTRA_STEPS=$5

    # Crash recovery: if only LAST exists, restore to TEMP
    if [ -d "$LAST" ] && [ ! -d "$TEMP" ]; then
        log "[Recovery] Restoring temp from last for Policy $NAME"
        mv "$LAST" "$TEMP"
    fi

    local CKPT
    CKPT=$(get_ckpt "$TEMP")

    rm -rf "$LAST"
    mkdir -p "$LAST"

    # Prepare dataset (merge rollout if provided)
    local ROLLOUT_ARG=""
    if [ -n "$ROLLOUT" ] && [ -f "$ROLLOUT" ]; then
        ROLLOUT_ARG="--rollout_data $ROLLOUT"
        log "Merging rollout data into dataset for Policy $NAME"
    fi

    python "${SCRIPT_DIR}/5_finetune.py" \
        --original_lerobot "$TEMP/lerobot_dataset" \
        $ROLLOUT_ARG \
        --checkpoint "$CKPT" \
        --out "$LAST" \
        --prepare_only --include_obj_pose --include_gripper

    # Copy wandb state for resumed logging
    if [ -d "$TEMP/checkpoints/wandb" ]; then
        cp -r "$TEMP/checkpoints/wandb" "$LAST/checkpoints/wandb"
    fi

    # Convert 'last' checkpoint dir → numbered symlink
    local CKPT_DIR="$LAST/checkpoints/checkpoints"
    local LAST_CKPT="$CKPT_DIR/last"
    if [ -d "$LAST_CKPT" ] && [ ! -L "$LAST_CKPT" ]; then
        local CUR_STEP
        CUR_STEP=$(get_step "$LAST")
        local STEP_NAME
        STEP_NAME=$(printf "%06d" "$CUR_STEP")
        mv "$LAST_CKPT" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        log "Symlink: last → $STEP_NAME"
    fi

    # Compute target step
    local CUR_STEP TARGET
    CUR_STEP=$(get_step "$LAST")
    TARGET=$((CUR_STEP + EXTRA_STEPS))
    log "Training Policy $NAME: step $CUR_STEP → $TARGET ($NUM_TRAIN_GPUS GPUs)"

    # Weighted sampling
    local SAMPLE_WEIGHTS_ARG=""
    local SW_PATH="$LAST/lerobot_dataset/meta/sampling_weights.json"
    if [ -f "$SW_PATH" ]; then
        SAMPLE_WEIGHTS_ARG="--sample_weights $SW_PATH"
        log "Using weighted sampling (1:1 old/new)"
    fi

    local PORT
    PORT=$(random_port)

    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$NUM_TRAIN_GPUS --master_port=$PORT \
        "${SCRIPT_DIR}/4_train.py" \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$LAST/lerobot_dataset" \
        --out "$LAST" \
        --steps $TARGET \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --n_action_steps $TRAIN_N_ACTION_STEPS \
        --noise_scheduler_type $NOISE_SCHEDULER_TYPE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --save_freq $EXTRA_STEPS \
        --skip_convert --resume --wandb \
        --wandb_project "$WANDB_PROJECT" \
        --seed $SEED \
        $SAMPLE_WEIGHTS_ARG

    log "Policy $NAME trained to step $TARGET"

    # Rotate: last → temp
    rm -rf "$TEMP"
    mv "$LAST" "$TEMP"
    log "Rotated: last → temp for Policy $NAME"
}

# Save iteration checkpoint
save_iter_checkpoint() {
    local ITER=$1
    local NAME=$2
    local TEMP=$3
    local CKPT_SRC
    CKPT_SRC=$(get_ckpt "$TEMP")
    local DEST="$BASE_DIR/iter${ITER}_ckpt_${NAME}"
    if [ -d "$CKPT_SRC" ] && [ ! -d "$DEST" ]; then
        log "Saving checkpoint: $DEST"
        cp -r "$CKPT_SRC" "$DEST"
    fi
}

# =============================================================================
# Initialize experiment
# =============================================================================
mkdir -p "$BASE_DIR" "$LOG_DIR" "$WEIGHTS_DIR" "$LEROBOT_DIR" "$WORK_DIR"

# Tee all output to log file with timestamps
exec > >(add_timestamps | tee -a "${LOG_DIR}/pipeline.log") 2>&1

print_header "Rev2Fwd Pick-and-Place Simulator Pipeline"
echo "  Experiment:  $BASE_DIR"
echo "  Train GPUs:  $TRAIN_GPUS ($NUM_TRAIN_GPUS)"
echo "  Collect GPU: $COLLECT_GPU"
echo "  Iterations:  $ITER_ROUNDS × $NUM_CYCLES cycles"
echo "  Batch size:  $BATCH_SIZE"
echo "  Horizon:     $HORIZON"
echo ""

# =============================================================================
# Phase 1: Data Preparation
# =============================================================================
print_header "Phase 1: Data Preparation"

# Step 1.1: Collect Task B data
if ! phase_done phase1_collect; then
    log "Collecting $NUM_COLLECT_EPISODES Task B episodes..."
    CUDA_VISIBLE_DEVICES=$COLLECT_GPU python "${SCRIPT_DIR}/1_collect_task_B.py" \
        --out "$TASK_B_NPZ" \
        --num_episodes $NUM_COLLECT_EPISODES \
        --goal_xy $GOAL_X $GOAL_Y \
        --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
        --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
        --seed $SEED \
        $HEADLESS \
        2>&1 | tee "${LOG_DIR}/collect_B.log"
    mark_done phase1_collect
    log "Task B collection complete: $TASK_B_NPZ"
else
    log "[Skip] Task B collection already done"
fi

# Step 1.2: Time-reverse to get Task A
if ! phase_done phase1_reverse; then
    log "Reversing Task B → Task A..."
    python "${SCRIPT_DIR}/2_reverse_to_task_A.py" \
        --input "$TASK_B_NPZ" \
        --out "$TASK_A_NPZ" \
        --success_only 1 \
        --verify \
        2>&1 | tee "${LOG_DIR}/reverse.log"
    mark_done phase1_reverse
    log "Time reversal complete: $TASK_A_NPZ"
else
    log "[Skip] Time reversal already done"
fi

# Step 1.3: Convert NPZ → LeRobot format
if ! phase_done phase1_convert_A; then
    log "Converting Task A NPZ → LeRobot..."
    python "${SCRIPT_DIR}/4_train.py" \
        --dataset "$TASK_A_NPZ" \
        --out "$PA_WEIGHTS" \
        --lerobot_dataset_dir "${LEROBOT_DIR}/task_A" \
        --convert_only \
        --seed $SEED \
        2>&1 | tee "${LOG_DIR}/convert_A.log"
    mark_done phase1_convert_A
else
    log "[Skip] Task A conversion already done"
fi

if ! phase_done phase1_convert_B; then
    log "Converting Task B NPZ → LeRobot..."
    python "${SCRIPT_DIR}/4_train.py" \
        --dataset "$TASK_B_NPZ" \
        --out "$PB_WEIGHTS" \
        --lerobot_dataset_dir "${LEROBOT_DIR}/task_B" \
        --convert_only \
        --seed $SEED \
        2>&1 | tee "${LOG_DIR}/convert_B.log"
    mark_done phase1_convert_B
else
    log "[Skip] Task B conversion already done"
fi

log "Phase 1 complete."

# =============================================================================
# Phase 2: Initial Training
# =============================================================================
print_header "Phase 2: Initial Training"

# Calculate training steps: ceil(total_frames * epochs / (batch_size * gpus))
STEPS_A=$(calc_steps_from_npz "$TASK_A_NPZ" $TARGET_EPOCHS $BATCH_SIZE $NUM_TRAIN_GPUS)
STEPS_B=$(calc_steps_from_npz "$TASK_B_NPZ" $TARGET_EPOCHS $BATCH_SIZE $NUM_TRAIN_GPUS)
log "Calculated training steps: A=$STEPS_A  B=$STEPS_B"

# Train Policy A
if ! phase_done phase2_train_A; then
    log "Training Policy A ($STEPS_A steps)..."
    local_port_A=$(random_port)
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$NUM_TRAIN_GPUS --master_port=$local_port_A \
        "${SCRIPT_DIR}/4_train.py" \
        --dataset "$TASK_A_NPZ" \
        --out "$PA_WEIGHTS" \
        --lerobot_dataset_dir "${LEROBOT_DIR}/task_A" \
        --steps $STEPS_A \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --n_action_steps $TRAIN_N_ACTION_STEPS \
        --noise_scheduler_type $NOISE_SCHEDULER_TYPE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --skip_convert --wandb \
        --wandb_project "$WANDB_PROJECT" \
        --seed $SEED \
        2>&1 | tee "${LOG_DIR}/train_A.log"
    mark_done phase2_train_A
    log "Policy A training complete"
else
    log "[Skip] Policy A initial training already done"
fi

# Train Policy B
if ! phase_done phase2_train_B; then
    log "Training Policy B ($STEPS_B steps)..."
    local_port_B=$(random_port)
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$NUM_TRAIN_GPUS --master_port=$local_port_B \
        "${SCRIPT_DIR}/4_train.py" \
        --dataset "$TASK_B_NPZ" \
        --out "$PB_WEIGHTS" \
        --lerobot_dataset_dir "${LEROBOT_DIR}/task_B" \
        --steps $STEPS_B \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --n_action_steps $TRAIN_N_ACTION_STEPS \
        --noise_scheduler_type $NOISE_SCHEDULER_TYPE \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --skip_convert --wandb \
        --wandb_project "$WANDB_PROJECT" \
        --seed $SEED \
        2>&1 | tee "${LOG_DIR}/train_B.log"
    mark_done phase2_train_B
    log "Policy B training complete"
else
    log "[Skip] Policy B initial training already done"
fi

log "Phase 2 complete."

# =============================================================================
# Prepare for Phase 3: Set up working directories
# =============================================================================
if ! phase_done phase3_init; then
    print_header "Phase 3: Preparing working directories"

    # Copy trained weights into temp working directories for iterative training
    mkdir -p "$PA_TEMP" "$PB_TEMP"

    # Policy A
    cp -r "$PA_WEIGHTS/checkpoints" "$PA_TEMP/checkpoints"
    cp -r "${LEROBOT_DIR}/task_A" "$PA_TEMP/lerobot_dataset"

    # Policy B
    cp -r "$PB_WEIGHTS/checkpoints" "$PB_TEMP/checkpoints"
    cp -r "${LEROBOT_DIR}/task_B" "$PB_TEMP/lerobot_dataset"

    # Initialize record.json
    python3 -c "
import json, datetime
json.dump({
    'description': 'Rev2Fwd Pick-and-Place iterative training',
    'config': {
        'exp_name': '$EXP_NAME',
        'created_at': datetime.datetime.now().isoformat(),
        'iter_rounds': $ITER_ROUNDS,
        'num_cycles': $NUM_CYCLES,
        'steps_per_iter': $STEPS_PER_ITER,
        'horizon': $HORIZON,
        'batch_size': $BATCH_SIZE,
        'distance_threshold': $DISTANCE_THRESHOLD,
        'train_n_action_steps': $TRAIN_N_ACTION_STEPS,
        'eval_n_action_steps': $EVAL_N_ACTION_STEPS,
        'goal_xy': [$GOAL_X, $GOAL_Y],
        'red_region_center': [$RED_REGION_CENTER_X, $RED_REGION_CENTER_Y],
        'red_region_size': [$RED_REGION_SIZE_X, $RED_REGION_SIZE_Y],
        'train_gpus': '$TRAIN_GPUS',
    },
    'iterations': []
}, open('$RECORD_FILE', 'w'), indent=2)
"

    mark_done phase3_init
    log "Working directories initialized"
else
    log "[Skip] Working directories already initialized"

    # Crash recovery: restore temp from last if needed
    for pair in "A:$PA_TEMP:$PA_LAST" "B:$PB_TEMP:$PB_LAST"; do
        IFS=: read -r label temp last <<< "$pair"
        if [ ! -d "$temp" ] && [ -d "$last" ]; then
            log "[Recovery] Restoring temp for Policy $label"
            mv "$last" "$temp"
        fi
    done
fi

# =============================================================================
# Phase 3: Iterative DAgger
# =============================================================================
print_header "Phase 3: Iterative DAgger ($ITER_ROUNDS rounds)"

for iter in $(seq 1 $ITER_ROUNDS); do
    print_header "Iteration $iter / $ITER_ROUNDS"

    ROLLOUT_A="${BASE_DIR}/iter${iter}_collect_A.npz"
    ROLLOUT_B="${BASE_DIR}/iter${iter}_collect_B.npz"
    STATS="${ROLLOUT_A%.npz}.stats.json"

    # --- Step 3.1: Parallel evaluation (cyclic + fair A + fair B) ---
    FAIR_A_ITER="${BASE_DIR}/iter${iter}_fair_A.stats.json"
    FAIR_B_ITER="${BASE_DIR}/iter${iter}_fair_B.stats.json"

    if ! phase_done "iter${iter}_eval"; then
        CKPT_A=$(get_ckpt "$PA_TEMP")
        CKPT_B=$(get_ckpt "$PB_TEMP")
        STEP_A=$(get_step "$PA_TEMP")
        STEP_B=$(get_step "$PB_TEMP")
        log "Policy A (step $STEP_A): $CKPT_A"
        log "Policy B (step $STEP_B): $CKPT_B"
        log "Launching 3 parallel evals on GPUs $EVAL_GPU_1, $EVAL_GPU_2, $EVAL_GPU_3..."

        # --- Cyclic eval on EVAL_GPU_1 ---
        (
            CUDA_VISIBLE_DEVICES=$EVAL_GPU_1 python -u "${SCRIPT_DIR}/6_eval_cyclic.py" \
                --policy_A "$CKPT_A" \
                --policy_B "$CKPT_B" \
                --out_A "$ROLLOUT_A" \
                --out_B "$ROLLOUT_B" \
                --num_cycles $NUM_CYCLES \
                --horizon $HORIZON \
                --distance_threshold $DISTANCE_THRESHOLD \
                --n_action_steps $EVAL_N_ACTION_STEPS \
                --goal_xy $GOAL_X $GOAL_Y \
                --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
                --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
                $HEADLESS \
                2>&1 | tee "${LOG_DIR}/iter${iter}_collect.log"
        ) &
        CYCLIC_PID=$!

        # Stagger launches to avoid GPU/Vulkan contention during Isaac Lab init
        sleep 60

        # --- Fair test A on EVAL_GPU_2 ---
        (
            CUDA_VISIBLE_DEVICES=$EVAL_GPU_2 python -u "${SCRIPT_DIR}/7_eval_fair.py" \
                --policy "$CKPT_A" \
                --task A \
                --num_episodes 50 \
                --horizon $HORIZON \
                --distance_threshold $DISTANCE_THRESHOLD \
                --n_action_steps $EVAL_N_ACTION_STEPS \
                --goal_xy $GOAL_X $GOAL_Y \
                --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
                --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
                --out "$FAIR_A_ITER" \
                $HEADLESS \
                2>&1 | tee "${LOG_DIR}/iter${iter}_fair_A.log"
        ) &
        FAIR_A_PID=$!

        # Stagger launches to avoid GPU/Vulkan contention during Isaac Lab init
        sleep 60

        # --- Fair test B on EVAL_GPU_3 ---
        (
            CUDA_VISIBLE_DEVICES=$EVAL_GPU_3 python -u "${SCRIPT_DIR}/7_eval_fair.py" \
                --policy "$CKPT_B" \
                --task B \
                --num_episodes 50 \
                --horizon $HORIZON \
                --distance_threshold $DISTANCE_THRESHOLD \
                --n_action_steps $EVAL_N_ACTION_STEPS \
                --goal_xy $GOAL_X $GOAL_Y \
                --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
                --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
                --out "$FAIR_B_ITER" \
                $HEADLESS \
                2>&1 | tee "${LOG_DIR}/iter${iter}_fair_B.log"
        ) &
        FAIR_B_PID=$!

        # --- Wait for all three ---
        EVAL_FAILED=0
        if ! wait $CYCLIC_PID; then
            log "ERROR: Cyclic eval failed!"; EVAL_FAILED=1
        fi
        if ! wait $FAIR_A_PID; then
            log "ERROR: Fair test A failed!"; EVAL_FAILED=1
        fi
        if ! wait $FAIR_B_PID; then
            log "ERROR: Fair test B failed!"; EVAL_FAILED=1
        fi
        if [ $EVAL_FAILED -ne 0 ]; then
            log "FATAL: One or more eval processes failed at iteration $iter"
            exit 1
        fi

        # Record metrics (cyclic + fair test)
        append_record $iter "$STATS" $STEP_A $STEP_B "$FAIR_A_ITER" "$FAIR_B_ITER"

        # Print summary
        if [ -f "$STATS" ]; then
            python3 -c "
import json
with open('$STATS') as f:
    s = json.load(f)['summary']
print(f'  Cyclic: A={s[\"task_A_success_count\"]}/{s[\"total_task_A_episodes\"]}={s[\"task_A_success_rate\"]*100:.1f}%  B={s[\"task_B_success_count\"]}/{s[\"total_task_B_episodes\"]}={s[\"task_B_success_rate\"]*100:.1f}%')
"
        fi
        for ff in "$FAIR_A_ITER" "$FAIR_B_ITER"; do
            if [ -f "$ff" ]; then
                python3 -c "
import json
with open('$ff') as fh:
    d = json.load(fh)
print(f'  Fair {d[\"task\"]}: {d[\"num_success\"]}/{d[\"num_total\"]}={d[\"success_rate\"]*100:.1f}%')
"
            fi
        done

        # Copy latest fair test results for compatibility
        [ -f "$FAIR_A_ITER" ] && cp "$FAIR_A_ITER" "${BASE_DIR}/fair_test_A.stats.json"
        [ -f "$FAIR_B_ITER" ] && cp "$FAIR_B_ITER" "${BASE_DIR}/fair_test_B.stats.json"

        mark_done "iter${iter}_eval"
    else
        log "[Skip] Iteration $iter evaluation already done"
    fi

    # --- Step 3.2: Finetune Policy A ---
    if ! phase_done "iter${iter}_train_A"; then
        log "Finetuning Policy A ($STEPS_PER_ITER steps)..."
        run_resume_train A "$PA_TEMP" "$PA_LAST" "$ROLLOUT_A" $STEPS_PER_ITER
        save_iter_checkpoint $iter A "$PA_TEMP"
        mark_done "iter${iter}_train_A"
    else
        log "[Skip] Iteration $iter Policy A training already done"
    fi

    # --- Step 3.3: Finetune Policy B ---
    if ! phase_done "iter${iter}_train_B"; then
        log "Finetuning Policy B ($STEPS_PER_ITER steps)..."
        run_resume_train B "$PB_TEMP" "$PB_LAST" "$ROLLOUT_B" $STEPS_PER_ITER
        save_iter_checkpoint $iter B "$PB_TEMP"
        mark_done "iter${iter}_train_B"
    else
        log "[Skip] Iteration $iter Policy B training already done"
    fi

    log "Iteration $iter complete"
done

log "Phase 3 complete."

# Print final fair test results
print_header "Final Fair Test Results"
FAIR_A="${BASE_DIR}/fair_test_A.stats.json"
FAIR_B="${BASE_DIR}/fair_test_B.stats.json"
for f in "$FAIR_A" "$FAIR_B"; do
    if [ -f "$f" ]; then
        python3 -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'  Task {d[\"task\"]}: {d[\"num_success\"]}/{d[\"num_total\"]} = {d[\"success_rate\"]*100:.1f}%')
"
    fi
done

# =============================================================================
# Done
# =============================================================================
touch "$BASE_DIR/.complete"

print_header "Pipeline Complete!"
echo "  Experiment:  $BASE_DIR"
echo "  Record:      $RECORD_FILE"
echo "  Fair test A: $FAIR_A"
echo "  Fair test B: $FAIR_B"
echo ""
echo "  Final checkpoints:"
echo "    Policy A: $(get_ckpt "$PA_TEMP")"
echo "    Policy B: $(get_ckpt "$PB_TEMP")"
echo ""
echo "  Phase 5 (optional): Run failure analysis manually:"
echo "    CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/8_eval_failure_analysis.py \\"
echo "        --policy $(get_ckpt "$PA_TEMP") --task A --num_episodes 100 \\"
echo "        --out_dir $BASE_DIR --headless"
echo ""

# Print iteration summary
python3 << PYSUMMARY
import json
from pathlib import Path
rf = Path("$RECORD_FILE")
if rf.exists():
    with open(rf) as f:
        rec = json.load(f)
    iters = rec.get("iterations", [])
    if iters:
        print("  Iter  | Cyclic A | Cyclic B | Fair A   | Fair B")
        print("  ------|----------|----------|----------|--------")
        for it in iters:
            m = it["performance_metrics"]
            fm = it.get("fair_test_metrics", {})
            ca = f"{m['task_A_success_rate']*100:5.1f}%" if m.get('task_A_success_rate') else "  N/A"
            cb = f"{m['task_B_success_rate']*100:5.1f}%" if m.get('task_B_success_rate') else "  N/A"
            fa = f"{fm['fair_A_success_rate']*100:5.1f}%" if fm.get('fair_A_success_rate') is not None else "  N/A"
            fb = f"{fm['fair_B_success_rate']*100:5.1f}%" if fm.get('fair_B_success_rate') is not None else "  N/A"
            print(f"  {it['iteration']:4d}  | {ca:>8s} | {cb:>8s} | {fa:>8s} | {fb:>6s}")
PYSUMMARY
