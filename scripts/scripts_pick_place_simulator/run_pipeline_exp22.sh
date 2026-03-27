#!/bin/bash
# =============================================================================
# Rev2Fwd Pick-and-Place Simulator Pipeline — Exp22 variant
# =============================================================================
#
# 与 Exp20 的区别 (参见 exp22/plan.md):
#   - Rollout 数据先抽帧到 400 帧再做时间反转 (加大 action 步幅)
#   - 仅运行 1 轮 iteration (不做后续循环)
#   - GPU 分配: 0,2,4,5,6,7
#
# 其余设置沿用 Exp20:
#   - 使用对侧策略 rollout 的倒放数据训练 (避免 self imitation)
#   - 训练数据不累积, 只用初始数据 + 最新一轮反转 rollout
#   - 均匀采样 (无 WeightedRandomSampler)
#   - Task B 成功判定区域不缩减
#
# =============================================================================

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================

EXP_NAME="${EXP_NAME:-exp22}"
BASE_DIR="data/pick_place_isaac_lab_simulation/${EXP_NAME}"
SRC_DIR="data/pick_place_isaac_lab_simulation/exp20"

# --- Region parameters (same as exp20) ---
GOAL_X="0.5"
GOAL_Y="-0.2"
DISTANCE_THRESHOLD="0.03"
RED_REGION_CENTER_X="0.5"
RED_REGION_CENTER_Y="0.2"
RED_REGION_SIZE_X="0.3"
RED_REGION_SIZE_Y="0.3"

# --- Subsample ---
TARGET_FRAMES=400

# --- Training ---
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-4}
TRAIN_N_ACTION_STEPS=16
EVAL_N_ACTION_STEPS=16
STEPS_PER_ITER=5000

# --- DDIM ---
NOISE_SCHEDULER_TYPE="DDIM"
NUM_INFERENCE_STEPS=10

# --- Eval ---
HORIZON=1200

# --- GPU ---
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-0,2,4,5,6,7}"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

# Eval GPUs (from env or defaults)
EVAL_GPU_1="${EVAL_GPU_1:-2}"
EVAL_GPU_2="${EVAL_GPU_2:-4}"

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

TASK_B_NPZ="${BASE_DIR}/task_B_100.npz"
TASK_A_NPZ="${BASE_DIR}/task_A_reversed_100.npz"

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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Verify a file exists and print its size
verify_file() {
    local filepath=$1
    local label=${2:-$1}
    if [ -f "$filepath" ]; then
        local size
        size=$(du -h "$filepath" | cut -f1)
        log "  ✓ $label  ($size)"
        return 0
    else
        log "  ✗ MISSING: $label"
        return 1
    fi
}

# Verify a directory exists and print its size
verify_dir() {
    local dirpath=$1
    local label=${2:-$1}
    if [ -d "$dirpath" ]; then
        local size
        size=$(du -sh "$dirpath" | cut -f1)
        log "  ✓ $label/  ($size)"
        return 0
    else
        log "  ✗ MISSING: $label/"
        return 1
    fi
}

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

# =============================================================================
# run_fresh_train — rebuild dataset from ORIGINAL lerobot each time
# =============================================================================
run_fresh_train() {
    local NAME=$1
    local TEMP=$2
    local LAST=$3
    local ROLLOUT=$4
    local ORIG_LEROBOT=$5
    local EXTRA_STEPS=$6

    # Crash recovery
    if [ -d "$LAST" ] && [ ! -d "$TEMP" ]; then
        log "[Recovery] Restoring temp from last for Policy $NAME"
        mv "$LAST" "$TEMP"
    fi

    local CKPT
    CKPT=$(get_ckpt "$TEMP")

    rm -rf "$LAST"
    mkdir -p "$LAST"

    local ROLLOUT_ARG=""
    if [ -n "$ROLLOUT" ] && [ -f "$ROLLOUT" ]; then
        ROLLOUT_ARG="--rollout_data $ROLLOUT"
        log "Merging reversed rollout into ORIGINAL dataset for Policy $NAME"
    fi

    python "${SCRIPT_DIR}/5_finetune.py" \
        --original_lerobot "$ORIG_LEROBOT" \
        $ROLLOUT_ARG \
        --checkpoint "$CKPT" \
        --out "$LAST" \
        --prepare_only --include_obj_pose --include_gripper

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

    local CUR_STEP TARGET
    CUR_STEP=$(get_step "$LAST")
    TARGET=$((CUR_STEP + EXTRA_STEPS))
    log "Training Policy $NAME: step $CUR_STEP → $TARGET ($NUM_TRAIN_GPUS GPUs)"

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
        --seed $SEED

    log "Policy $NAME trained to step $TARGET"

    rm -rf "$TEMP"
    mv "$LAST" "$TEMP"
    log "Rotated: last → temp for Policy $NAME"
}

# =============================================================================
# Initialize experiment
# =============================================================================
mkdir -p "$BASE_DIR" "$LOG_DIR" "$WEIGHTS_DIR" "$LEROBOT_DIR" "$WORK_DIR"

exec > >(add_timestamps | tee -a "${LOG_DIR}/pipeline.log") 2>&1

print_header "Rev2Fwd Pick-and-Place Pipeline (Exp22: subsample + reverse)"
echo "  Experiment:   $BASE_DIR"
echo "  Source:       $SRC_DIR"
echo "  Train GPUs:   $TRAIN_GPUS ($NUM_TRAIN_GPUS)"
echo "  Eval GPUs:    $EVAL_GPU_1, $EVAL_GPU_2"
echo "  Batch size:   $BATCH_SIZE"
echo "  LR:           $LR"
echo "  Subsample:    ${TARGET_FRAMES} frames/episode"
echo "  Horizon:      $HORIZON"
echo ""

# =============================================================================
# Phase 0: Copy data from exp20
# =============================================================================
print_header "Phase 0: Copy data from exp20"

if ! phase_done phase0_copy; then
    log "Copying initial data and weights from exp20..."

    # NPZ data
    cp -v "${SRC_DIR}/task_B_100.npz" "${BASE_DIR}/"
    cp -v "${SRC_DIR}/task_A_reversed_100.npz" "${BASE_DIR}/"

    # LeRobot datasets
    cp -rv "${SRC_DIR}/lerobot/task_A" "${LEROBOT_DIR}/"
    cp -rv "${SRC_DIR}/lerobot/task_B" "${LEROBOT_DIR}/"

    # Weights (iter0)
    cp -rv "${SRC_DIR}/weights/PP_A" "${WEIGHTS_DIR}/"
    cp -rv "${SRC_DIR}/weights/PP_B" "${WEIGHTS_DIR}/"

    # Iter1 rollout data
    cp -v "${SRC_DIR}/iter1_collect_A.npz" "${BASE_DIR}/"
    cp -v "${SRC_DIR}/iter1_collect_B.npz" "${BASE_DIR}/"

    # Also copy iter1 eval stats for reference
    [ -f "${SRC_DIR}/iter1_collect_A.stats.json" ] && cp -v "${SRC_DIR}/iter1_collect_A.stats.json" "${BASE_DIR}/"
    [ -f "${SRC_DIR}/iter1_fair_A.stats.json" ] && cp -v "${SRC_DIR}/iter1_fair_A.stats.json" "${BASE_DIR}/iter0_fair_A.stats.json"
    [ -f "${SRC_DIR}/iter1_fair_B.stats.json" ] && cp -v "${SRC_DIR}/iter1_fair_B.stats.json" "${BASE_DIR}/iter0_fair_B.stats.json"

    mark_done phase0_copy
    log "Phase 0 copy complete"
else
    log "[Skip] Data copy already done"
fi

# Verify copied files
log "Verifying copied data..."
verify_file "$TASK_B_NPZ" "task_B_100.npz"
verify_file "$TASK_A_NPZ" "task_A_reversed_100.npz"
verify_file "${BASE_DIR}/iter1_collect_A.npz" "iter1_collect_A.npz"
verify_file "${BASE_DIR}/iter1_collect_B.npz" "iter1_collect_B.npz"
verify_dir  "${LEROBOT_DIR}/task_A" "lerobot/task_A"
verify_dir  "${LEROBOT_DIR}/task_B" "lerobot/task_B"
verify_dir  "$PA_WEIGHTS" "weights/PP_A"
verify_dir  "$PB_WEIGHTS" "weights/PP_B"

# Initialize working directories
if ! phase_done phase0_init; then
    log "Initializing working directories..."

    mkdir -p "$PA_TEMP" "$PB_TEMP"
    cp -r "$PA_WEIGHTS/checkpoints" "$PA_TEMP/checkpoints"
    cp -r "${LEROBOT_DIR}/task_A" "$PA_TEMP/lerobot_dataset"
    cp -r "$PB_WEIGHTS/checkpoints" "$PB_TEMP/checkpoints"
    cp -r "${LEROBOT_DIR}/task_B" "$PB_TEMP/lerobot_dataset"

    # Initialize record.json
    python3 -c "
import json, datetime
json.dump({
    'description': 'Rev2Fwd Pick-and-Place Exp22: subsample ${TARGET_FRAMES} frames + reverse opposite rollout',
    'config': {
        'exp_name': '$EXP_NAME',
        'variant': 'exp22',
        'source': 'exp20',
        'created_at': datetime.datetime.now().isoformat(),
        'target_frames': $TARGET_FRAMES,
        'steps_per_iter': $STEPS_PER_ITER,
        'horizon': $HORIZON,
        'batch_size': $BATCH_SIZE,
        'lr': '$LR',
        'distance_threshold': $DISTANCE_THRESHOLD,
        'train_n_action_steps': $TRAIN_N_ACTION_STEPS,
        'eval_n_action_steps': $EVAL_N_ACTION_STEPS,
        'goal_xy': [$GOAL_X, $GOAL_Y],
        'red_region_center': [$RED_REGION_CENTER_X, $RED_REGION_CENTER_Y],
        'red_region_size': [$RED_REGION_SIZE_X, $RED_REGION_SIZE_Y],
        'train_gpus': '$TRAIN_GPUS',
        'changes': [
            'subsample_before_reverse: Downsample rollout to ${TARGET_FRAMES} frames before time reversal',
            'single_iteration: Only run iteration 1, no cyclic loop',
            'reverse_opposite_rollout: Train A with reversed B rollout, B with reversed A rollout',
            'non_cumulative: Only use initial data + current iteration reversed rollout',
            'uniform_sampling: No WeightedRandomSampler',
        ],
    },
    'iterations': []
}, open('$RECORD_FILE', 'w'), indent=2)
"

    mark_done phase0_init
    log "Working directories initialized"
else
    log "[Skip] Working directories already initialized"

    # Crash recovery
    for pair in "A:$PA_TEMP:$PA_LAST" "B:$PB_TEMP:$PB_LAST"; do
        IFS=: read -r label temp last <<< "$pair"
        if [ ! -d "$temp" ] && [ -d "$last" ]; then
            log "[Recovery] Restoring temp for Policy $label"
            mv "$last" "$temp"
        fi
    done
fi

# =============================================================================
# Phase 1: Subsample + Reverse
# =============================================================================
print_header "Phase 1: Subsample + Reverse rollout data"

ROLLOUT_A="${BASE_DIR}/iter1_collect_A.npz"
ROLLOUT_B="${BASE_DIR}/iter1_collect_B.npz"
ROLLOUT_B_SUB="${BASE_DIR}/iter1_collect_B_subsampled.npz"
ROLLOUT_A_SUB="${BASE_DIR}/iter1_collect_A_subsampled.npz"
ROLLOUT_B_SUB_REV="${BASE_DIR}/iter1_collect_B_subsampled_reversed.npz"
ROLLOUT_A_SUB_REV="${BASE_DIR}/iter1_collect_A_subsampled_reversed.npz"

# Step 1.1: Subsample
if ! phase_done phase1_subsample; then
    log "Subsampling rollout data to ${TARGET_FRAMES} frames/episode..."

    if [ ! -f "$ROLLOUT_B" ]; then
        log "FATAL: $ROLLOUT_B not found!"
        exit 1
    fi
    if [ ! -f "$ROLLOUT_A" ]; then
        log "FATAL: $ROLLOUT_A not found!"
        exit 1
    fi

    log "  Subsampling B rollout → $(basename $ROLLOUT_B_SUB)"
    python "${SCRIPT_DIR}/3_subsample_episodes.py" \
        --input "$ROLLOUT_B" \
        --out "$ROLLOUT_B_SUB" \
        --target_frames $TARGET_FRAMES \
        --success_only 1 \
        2>&1 | add_timestamps | tee "${LOG_DIR}/subsample_B.log"

    log "  Subsampling A rollout → $(basename $ROLLOUT_A_SUB)"
    python "${SCRIPT_DIR}/3_subsample_episodes.py" \
        --input "$ROLLOUT_A" \
        --out "$ROLLOUT_A_SUB" \
        --target_frames $TARGET_FRAMES \
        --success_only 1 \
        2>&1 | add_timestamps | tee "${LOG_DIR}/subsample_A.log"

    # Verify subsampled files were created
    log "Verifying subsampled files:"
    verify_file "$ROLLOUT_B_SUB" "iter1_collect_B_subsampled.npz" || exit 1
    verify_file "$ROLLOUT_A_SUB" "iter1_collect_A_subsampled.npz" || exit 1

    mark_done phase1_subsample
    log "Subsampling complete"
else
    log "[Skip] Subsampling already done"
    verify_file "$ROLLOUT_B_SUB" "iter1_collect_B_subsampled.npz"
    verify_file "$ROLLOUT_A_SUB" "iter1_collect_A_subsampled.npz"
fi

# Step 1.2: Reverse subsampled data
if ! phase_done phase1_reverse; then
    log "Reversing subsampled rollout data..."

    log "  Reversing B subsampled → $(basename $ROLLOUT_B_SUB_REV) (for Policy A)"
    python "${SCRIPT_DIR}/2_reverse_to_task_A.py" \
        --input "$ROLLOUT_B_SUB" \
        --out "$ROLLOUT_B_SUB_REV" \
        --success_only 1 \
        --verify \
        2>&1 | add_timestamps | tee "${LOG_DIR}/reverse_B_sub.log"

    log "  Reversing A subsampled → $(basename $ROLLOUT_A_SUB_REV) (for Policy B)"
    python "${SCRIPT_DIR}/2_reverse_to_task_A.py" \
        --input "$ROLLOUT_A_SUB" \
        --out "$ROLLOUT_A_SUB_REV" \
        --success_only 1 \
        --verify \
        2>&1 | add_timestamps | tee "${LOG_DIR}/reverse_A_sub.log"

    # Verify reversed files were created
    log "Verifying reversed files:"
    verify_file "$ROLLOUT_B_SUB_REV" "iter1_collect_B_subsampled_reversed.npz" || exit 1
    verify_file "$ROLLOUT_A_SUB_REV" "iter1_collect_A_subsampled_reversed.npz" || exit 1

    mark_done phase1_reverse
    log "Reversal complete"
else
    log "[Skip] Reversal already done"
    verify_file "$ROLLOUT_B_SUB_REV" "iter1_collect_B_subsampled_reversed.npz"
    verify_file "$ROLLOUT_A_SUB_REV" "iter1_collect_A_subsampled_reversed.npz"
fi

# =============================================================================
# Phase 2: Train Iteration 1
# =============================================================================
print_header "Phase 2: Train Iteration 1"

# Step 2.1: Train Policy A (initial data + reversed B subsampled rollout)
if ! phase_done iter1_train_A; then
    log "Training Policy A with subsampled+reversed B rollout ($STEPS_PER_ITER steps)..."
    log "  Rollout data: $ROLLOUT_B_SUB_REV"
    log "  Original LeRobot: ${LEROBOT_DIR}/task_A"
    run_fresh_train A "$PA_TEMP" "$PA_LAST" "$ROLLOUT_B_SUB_REV" "${LEROBOT_DIR}/task_A" $STEPS_PER_ITER \
        2>&1 | add_timestamps | tee "${LOG_DIR}/iter1_train_A.log"

    # Save checkpoint
    CKPT_SRC=$(get_ckpt "$PA_TEMP")
    DEST="${BASE_DIR}/iter1_ckpt_A"
    [ -d "$CKPT_SRC" ] && [ ! -d "$DEST" ] && cp -r "$CKPT_SRC" "$DEST"

    mark_done iter1_train_A
else
    log "[Skip] Iteration 1 Policy A training already done"
fi

# Step 2.2: Train Policy B (initial data + reversed A subsampled rollout)
if ! phase_done iter1_train_B; then
    log "Training Policy B with subsampled+reversed A rollout ($STEPS_PER_ITER steps)..."
    log "  Rollout data: $ROLLOUT_A_SUB_REV"
    log "  Original LeRobot: ${LEROBOT_DIR}/task_B"
    run_fresh_train B "$PB_TEMP" "$PB_LAST" "$ROLLOUT_A_SUB_REV" "${LEROBOT_DIR}/task_B" $STEPS_PER_ITER \
        2>&1 | add_timestamps | tee "${LOG_DIR}/iter1_train_B.log"

    # Save checkpoint
    CKPT_SRC=$(get_ckpt "$PB_TEMP")
    DEST="${BASE_DIR}/iter1_ckpt_B"
    [ -d "$CKPT_SRC" ] && [ ! -d "$DEST" ] && cp -r "$CKPT_SRC" "$DEST"

    mark_done iter1_train_B
else
    log "[Skip] Iteration 1 Policy B training already done"
fi

log "Phase 2 complete."

# =============================================================================
# Phase 3: Fair Test Evaluation
# =============================================================================
print_header "Phase 3: Fair Test Evaluation"

FAIR_A="${BASE_DIR}/iter1_fair_A.stats.json"
FAIR_B="${BASE_DIR}/iter1_fair_B.stats.json"

if ! phase_done iter1_fair_test; then
    CKPT_A=$(get_ckpt "$PA_TEMP")
    CKPT_B=$(get_ckpt "$PB_TEMP")
    log "Policy A checkpoint: $CKPT_A"
    log "Policy B checkpoint: $CKPT_B"
    log "Launching fair tests on GPUs $EVAL_GPU_1, $EVAL_GPU_2..."

    # Fair test A on EVAL_GPU_1
    (
        CUDA_VISIBLE_DEVICES=$EVAL_GPU_1 python -u "${SCRIPT_DIR}/7_eval_fair.py" \
            --policy "$CKPT_A" \
            --task A \
            --num_episodes 50 \
            --horizon $HORIZON \
            --distance_threshold $DISTANCE_THRESHOLD \
            --n_action_steps $EVAL_N_ACTION_STEPS \
            --goal_xy $GOAL_X $GOAL_Y \
            --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
            --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
            --out "$FAIR_A" \
            $HEADLESS \
            2>&1 | add_timestamps | tee "${LOG_DIR}/iter1_fair_A.log"
    ) &
    FAIR_A_PID=$!

    # Stagger launch
    sleep 60

    # Fair test B on EVAL_GPU_2
    (
        CUDA_VISIBLE_DEVICES=$EVAL_GPU_2 python -u "${SCRIPT_DIR}/7_eval_fair.py" \
            --policy "$CKPT_B" \
            --task B \
            --num_episodes 50 \
            --horizon $HORIZON \
            --distance_threshold $DISTANCE_THRESHOLD \
            --n_action_steps $EVAL_N_ACTION_STEPS \
            --goal_xy $GOAL_X $GOAL_Y \
            --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
            --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
            --out "$FAIR_B" \
            $HEADLESS \
            2>&1 | add_timestamps | tee "${LOG_DIR}/iter1_fair_B.log"
    ) &
    FAIR_B_PID=$!

    # Wait for both
    EVAL_FAILED=0
    if ! wait $FAIR_A_PID; then
        log "ERROR: Fair test A failed!"; EVAL_FAILED=1
    fi
    if ! wait $FAIR_B_PID; then
        log "ERROR: Fair test B failed!"; EVAL_FAILED=1
    fi
    if [ $EVAL_FAILED -ne 0 ]; then
        log "FATAL: Fair test failed"
        exit 1
    fi

    # Record metrics
    STEP_A=$(get_step "$PA_TEMP")
    STEP_B=$(get_step "$PB_TEMP")

    python3 << PYEOF
import json
from pathlib import Path

rec = json.load(open("$RECORD_FILE"))
entry = {
    "iteration": 1,
    "checkpoint_info": {"policy_A_step": $STEP_A, "policy_B_step": $STEP_B},
    "fair_test_metrics": {},
}
fm = entry["fair_test_metrics"]
fa = Path("$FAIR_A")
fb = Path("$FAIR_B")
if fa.exists():
    d = json.load(open(fa))
    fm["fair_A_success_rate"] = d.get("success_rate", 0)
    fm["fair_A_num_success"] = d.get("num_success", 0)
    fm["fair_A_num_total"] = d.get("num_total", 0)
if fb.exists():
    d = json.load(open(fb))
    fm["fair_B_success_rate"] = d.get("success_rate", 0)
    fm["fair_B_num_success"] = d.get("num_success", 0)
    fm["fair_B_num_total"] = d.get("num_total", 0)

rec["iterations"] = [entry]
json.dump(rec, open("$RECORD_FILE", "w"), indent=2)

print(f"  Fair A: {fm.get('fair_A_num_success', 0)}/{fm.get('fair_A_num_total', 0)} = {fm.get('fair_A_success_rate', 0)*100:.1f}%")
print(f"  Fair B: {fm.get('fair_B_num_success', 0)}/{fm.get('fair_B_num_total', 0)} = {fm.get('fair_B_success_rate', 0)*100:.1f}%")
PYEOF

    mark_done iter1_fair_test
else
    log "[Skip] Fair test already done"
fi

# =============================================================================
# Print results
# =============================================================================
print_header "Exp22 Results"

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
echo "  Iteration 1 checkpoints:"
echo "    Policy A: $(get_ckpt "$PA_TEMP")"
echo "    Policy B: $(get_ckpt "$PB_TEMP")"
echo ""

# List all intermediate and output files with sizes
print_header "Artifact inventory"
log "Data files:"
for f in \
    "$TASK_B_NPZ" "$TASK_A_NPZ" \
    "$ROLLOUT_A" "$ROLLOUT_B" \
    "$ROLLOUT_A_SUB" "$ROLLOUT_B_SUB" \
    "$ROLLOUT_A_SUB_REV" "$ROLLOUT_B_SUB_REV" \
    "$FAIR_A" "$FAIR_B" \
    "$RECORD_FILE"; do
    verify_file "$f" "$(basename "$f")"
done
log "Directories:"
for d in \
    "${LEROBOT_DIR}/task_A" "${LEROBOT_DIR}/task_B" \
    "$PA_WEIGHTS" "$PB_WEIGHTS" \
    "$PA_TEMP" "$PB_TEMP" \
    "${BASE_DIR}/iter1_ckpt_A" "${BASE_DIR}/iter1_ckpt_B" \
    "$LOG_DIR"; do
    verify_dir "$d" "$(basename "$d")"
done
log "Log files:"
ls -lh "$LOG_DIR"/ 2>/dev/null || true
echo ""
