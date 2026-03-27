#!/bin/bash
# =============================================================================
# Rev2Fwd Pick-and-Place Simulator Pipeline — Exp21
# =============================================================================
#
# Exp21: 用 Policy A 的成功 rollout 轨迹的时间反转从零训练 Policy B
#
# Pipeline:
#   Phase 1: (pre-done) Policy A checkpoint 已从 exp20 复制
#   Phase 2: Policy A rollout 200 条成功轨迹 (fair test 环境)
#   Phase 3: 时间反转 + 转换为 LeRobot 格式
#   Phase 4: 从零训练 Policy B (400 epochs, 10 GPUs)
#   Phase 5: Fair test 评估 Policy B
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
EXP_NAME="${EXP_NAME:-exp21}"
BASE_DIR="data/pick_place_isaac_lab_simulation/${EXP_NAME}"

# --- Region parameters (Mode 3, fixed) ---
GOAL_X="0.5"
GOAL_Y="-0.2"
DISTANCE_THRESHOLD="0.03"
RED_REGION_CENTER_X="0.5"
RED_REGION_CENTER_Y="0.2"
RED_REGION_SIZE_X="0.3"
RED_REGION_SIZE_Y="0.3"

# --- Rollout collection (parallel across all GPUs) ---
TARGET_SUCCESSES=200
SUCCESSES_PER_GPU=20
MAX_ATTEMPTS_PER_GPU=60
HORIZON=1200

# --- Training ---
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-3e-4}
TRAIN_N_ACTION_STEPS=16
EVAL_N_ACTION_STEPS=16
TARGET_EPOCHS=400

# --- DDIM ---
NOISE_SCHEDULER_TYPE="DDIM"
NUM_INFERENCE_STEPS=10

# --- Evaluation ---
FAIR_TEST_EPISODES=50

# --- GPU ---
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9}"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
COLLECT_GPU=${COLLECT_GPU:-0}

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

PA_WEIGHTS="${WEIGHTS_DIR}/PP_A"
PA_CKPT="${PA_WEIGHTS}/checkpoints/checkpoints/last/pretrained_model"

ROLLOUT_NPZ="${BASE_DIR}/rollout_A_${TARGET_SUCCESSES}.npz"
ROLLOUT_STATS="${BASE_DIR}/rollout_A_${TARGET_SUCCESSES}.stats.json"
REVERSED_NPZ="${BASE_DIR}/rollout_A_${TARGET_SUCCESSES}_reversed.npz"

LEROBOT_DIR="${BASE_DIR}/lerobot/task_B_from_rollout"

PB_WEIGHTS="${BASE_DIR}/weights_B_new/PP_B"

FAIR_TEST_STATS="${BASE_DIR}/fair_test_B.stats.json"
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

phase_done() { [ -f "$BASE_DIR/.done_${1}" ]; }
mark_done()  { touch "$BASE_DIR/.done_${1}"; }

random_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

calc_steps_from_npz() {
    # Usage: calc_steps_from_npz <npz_path> <epochs> <batch_size> <num_gpus>
    python3 -c "
import numpy as np, math
data = np.load('$1', allow_pickle=True)
eps = data['episodes']
total_frames = sum(len(e['images']) for e in eps)
steps = math.ceil(total_frames * $2 / ($3 * $4))
print(steps)
"
}

# =============================================================================
# Initialize experiment
# =============================================================================
mkdir -p "$BASE_DIR" "$LOG_DIR" "$WEIGHTS_DIR"

# Tee all output to log file with timestamps
exec > >(add_timestamps | tee -a "${LOG_DIR}/pipeline.log") 2>&1

print_header "Rev2Fwd Pick-and-Place Pipeline (Exp21)"
echo "  Experiment:     $BASE_DIR"
echo "  Train GPUs:     $TRAIN_GPUS ($NUM_TRAIN_GPUS)"
echo "  Collect GPU:    $COLLECT_GPU"
echo "  Batch size:     $BATCH_SIZE"
echo "  LR:             $LR"
echo "  Target epochs:  $TARGET_EPOCHS"
echo "  Horizon:        $HORIZON"
echo "  Target success: $TARGET_SUCCESSES"
echo "  Variant:        rollout A → reverse → train B from scratch"
echo ""

# Initialize record.json
if [ ! -f "$RECORD_FILE" ]; then
    python3 -c "
import json, datetime
json.dump({
    'description': 'Exp21: Train Policy B from reversed Policy A rollout (from scratch)',
    'config': {
        'exp_name': '$EXP_NAME',
        'created_at': datetime.datetime.now().isoformat(),
        'target_successes': $TARGET_SUCCESSES,
        'max_attempts': $MAX_ATTEMPTS,
        'target_epochs': $TARGET_EPOCHS,
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
        'num_train_gpus': $NUM_TRAIN_GPUS,
        'seed': $SEED,
    },
    'phases': {}
}, open('$RECORD_FILE', 'w'), indent=2)
"
fi

# =============================================================================
# Phase 1: Verify Policy A checkpoint
# =============================================================================
print_header "Phase 1: Verify Policy A Checkpoint"

if [ ! -d "$PA_CKPT" ]; then
    log "FATAL: Policy A checkpoint not found at $PA_CKPT"
    log "Please copy from exp20: cp -r exp20/weights/PP_A exp21/weights/PP_A"
    exit 1
fi
log "Policy A checkpoint found: $PA_CKPT"

# =============================================================================
# Phase 2: Parallel Rollout Collection (10 GPUs × 20 successes each)
# =============================================================================
print_header "Phase 2: Parallel Rollout Collection (${NUM_TRAIN_GPUS} GPUs × ${SUCCESSES_PER_GPU} each = ${TARGET_SUCCESSES})"

if ! phase_done phase2_rollout; then
    IFS=',' read -ra GPU_LIST <<< "$TRAIN_GPUS"
    ROLLOUT_PIDS=()
    ROLLOUT_FAILED=0

    log "Launching ${#GPU_LIST[@]} parallel rollout workers..."
    log "  Policy:          $PA_CKPT"
    log "  Horizon:         $HORIZON"
    log "  Per-GPU target:  $SUCCESSES_PER_GPU successes"
    log "  Per-GPU max:     $MAX_ATTEMPTS_PER_GPU attempts"

    for i in "${!GPU_LIST[@]}"; do
        GPU_ID="${GPU_LIST[$i]}"
        WORKER_SEED=$((SEED + i * 1000))
        WORKER_NPZ="${BASE_DIR}/rollout_A_worker_${i}.npz"
        WORKER_STATS="${BASE_DIR}/rollout_A_worker_${i}.stats.json"
        WORKER_LOG="${LOG_DIR}/rollout_A_gpu${GPU_ID}.log"

        log "  Worker $i → GPU $GPU_ID (seed=$WORKER_SEED)"

        (
            CUDA_VISIBLE_DEVICES=$GPU_ID python -u "${SCRIPT_DIR}/9_collect_rollout.py" \
                --policy "$PA_CKPT" \
                --task A \
                --target_successes $SUCCESSES_PER_GPU \
                --max_attempts $MAX_ATTEMPTS_PER_GPU \
                --horizon $HORIZON \
                --distance_threshold $DISTANCE_THRESHOLD \
                --n_action_steps $EVAL_N_ACTION_STEPS \
                --goal_xy $GOAL_X $GOAL_Y \
                --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
                --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
                --out_npz "$WORKER_NPZ" \
                --out_stats "$WORKER_STATS" \
                --seed $WORKER_SEED \
                $HEADLESS \
                2>&1 | tee "$WORKER_LOG"
        ) &
        ROLLOUT_PIDS+=($!)

        # Stagger launches by 30s to avoid Isaac Lab init contention
        if [ $i -lt $((${#GPU_LIST[@]} - 1)) ]; then
            sleep 30
        fi
    done

    # Wait for all workers
    for i in "${!ROLLOUT_PIDS[@]}"; do
        if ! wait ${ROLLOUT_PIDS[$i]}; then
            log "ERROR: Rollout worker $i (PID ${ROLLOUT_PIDS[$i]}) failed!"
            ROLLOUT_FAILED=1
        fi
    done

    if [ $ROLLOUT_FAILED -ne 0 ]; then
        log "FATAL: One or more rollout workers failed"
        exit 1
    fi

    # Merge all worker NPZ files into one
    log "Merging worker outputs..."
    python3 << 'MERGE_PY'
import numpy as np, json, glob, sys
from pathlib import Path

base = "BASEDIR_PLACEHOLDER"
MERGE_PY

    python3 -c "
import numpy as np, json, time
from pathlib import Path
from datetime import datetime

base = Path('$BASE_DIR')
num_workers = ${#GPU_LIST[@]}

all_episodes = []
all_stats = []
for i in range(num_workers):
    npz_path = base / f'rollout_A_worker_{i}.npz'
    stats_path = base / f'rollout_A_worker_{i}.stats.json'
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        eps = list(data['episodes'])
        all_episodes.extend(eps)
        print(f'  Worker {i}: {len(eps)} episodes')
    else:
        print(f'  Worker {i}: NPZ not found!')
    if stats_path.exists():
        with open(stats_path) as f:
            all_stats.append(json.load(f))

print(f'  Total merged: {len(all_episodes)} episodes')

# Save merged NPZ
out_npz = base / 'rollout_A_${TARGET_SUCCESSES}.npz'
np.savez_compressed(out_npz, episodes=np.array(all_episodes, dtype=object))
file_mb = out_npz.stat().st_size / (1024 * 1024)
total_frames = sum(len(e['images']) for e in all_episodes)
avg_len = total_frames / len(all_episodes) if all_episodes else 0

# Save merged stats
total_attempts = sum(s.get('total_attempts', 0) for s in all_stats)
total_elapsed = max((s.get('elapsed_sec', 0) for s in all_stats), default=0)
merged_stats = {
    'task': 'A',
    'target_successes': $TARGET_SUCCESSES,
    'collected_successes': len(all_episodes),
    'total_attempts': total_attempts,
    'success_rate': len(all_episodes) / total_attempts if total_attempts else 0,
    'total_frames': total_frames,
    'avg_episode_length': avg_len,
    'elapsed_sec': total_elapsed,
    'num_workers': num_workers,
    'timestamp': datetime.now().isoformat(),
}
out_stats = base / 'rollout_A_${TARGET_SUCCESSES}.stats.json'
with open(out_stats, 'w') as f:
    json.dump(merged_stats, f, indent=2)

print(f'  Saved: {out_npz} ({file_mb:.1f} MB, {total_frames} frames, avg_len={avg_len:.1f})')
print(f'  Stats: {out_stats}')
"

    # Verify
    if [ ! -f "$ROLLOUT_NPZ" ]; then
        log "FATAL: Merged rollout NPZ not generated!"
        exit 1
    fi

    COLLECTED=$(python3 -c "
import numpy as np
data = np.load('$ROLLOUT_NPZ', allow_pickle=True)
print(len(data['episodes']))
")
    log "Collected $COLLECTED successful episodes (merged from $NUM_TRAIN_GPUS workers)"

    # Update record
    python3 -c "
import json
with open('$RECORD_FILE') as f:
    rec = json.load(f)
with open('$ROLLOUT_STATS') as f:
    stats = json.load(f)
rec['phases']['phase2_rollout'] = {
    'status': 'done',
    'collected': stats['collected_successes'],
    'attempts': stats['total_attempts'],
    'success_rate': stats['success_rate'],
    'total_frames': stats['total_frames'],
    'avg_episode_length': stats['avg_episode_length'],
    'elapsed_sec': stats['elapsed_sec'],
    'num_workers': stats.get('num_workers', 1),
}
with open('$RECORD_FILE', 'w') as f:
    json.dump(rec, f, indent=2)
"

    mark_done phase2_rollout
    log "Phase 2 complete: $ROLLOUT_NPZ"
else
    log "[Skip] Rollout collection already done"
fi

# =============================================================================
# Phase 3a: Time-Reverse Rollout Data
# =============================================================================
print_header "Phase 3a: Time-Reverse Rollout Data"

if ! phase_done phase3_reverse; then
    log "Reversing rollout data..."
    log "  Input:  $ROLLOUT_NPZ"
    log "  Output: $REVERSED_NPZ"

    python "${SCRIPT_DIR}/2_reverse_to_task_A.py" \
        --input "$ROLLOUT_NPZ" \
        --out "$REVERSED_NPZ" \
        --success_only 1 \
        --verify \
        2>&1 | tee "${LOG_DIR}/reverse.log"

    if [ ! -f "$REVERSED_NPZ" ]; then
        log "FATAL: Reversed NPZ not generated!"
        exit 1
    fi

    REVERSED_FRAMES=$(python3 -c "
import numpy as np
data = np.load('$REVERSED_NPZ', allow_pickle=True)
eps = data['episodes']
total = sum(len(e['images']) for e in eps)
print(total)
")
    log "Reversed data: $REVERSED_FRAMES total frames"

    mark_done phase3_reverse
    log "Phase 3a complete: $REVERSED_NPZ"
else
    log "[Skip] Time reversal already done"
fi

# =============================================================================
# Phase 3b: Convert to LeRobot Format
# =============================================================================
print_header "Phase 3b: Convert to LeRobot Format"

if ! phase_done phase3_convert; then
    log "Converting reversed data to LeRobot format..."
    log "  Input:  $REVERSED_NPZ"
    log "  Output: $LEROBOT_DIR"

    mkdir -p "$(dirname "$PB_WEIGHTS")"

    python "${SCRIPT_DIR}/4_train.py" \
        --dataset "$REVERSED_NPZ" \
        --out "$PB_WEIGHTS" \
        --lerobot_dataset_dir "$LEROBOT_DIR" \
        --convert_only \
        --seed $SEED \
        2>&1 | tee "${LOG_DIR}/convert_B.log"

    if [ ! -d "$LEROBOT_DIR" ]; then
        log "FATAL: LeRobot dataset not generated!"
        exit 1
    fi

    mark_done phase3_convert
    log "Phase 3b complete: $LEROBOT_DIR"
else
    log "[Skip] LeRobot conversion already done"
fi

# =============================================================================
# Phase 4: Train Policy B from Scratch
# =============================================================================
print_header "Phase 4: Train Policy B from Scratch (${TARGET_EPOCHS} epochs, ${NUM_TRAIN_GPUS} GPUs)"

if ! phase_done phase4_train; then
    # Calculate training steps
    STEPS_B=$(calc_steps_from_npz "$REVERSED_NPZ" $TARGET_EPOCHS $BATCH_SIZE $NUM_TRAIN_GPUS)
    log "Calculated training steps: $STEPS_B"
    log "  Data:       $REVERSED_NPZ"
    log "  Epochs:     $TARGET_EPOCHS"
    log "  Batch size: $BATCH_SIZE"
    log "  GPUs:       $NUM_TRAIN_GPUS"
    log "  LR:         $LR"

    PORT=$(random_port)
    log "Training Policy B (step 0 → $STEPS_B) on port $PORT..."

    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$NUM_TRAIN_GPUS --master_port=$PORT \
        "${SCRIPT_DIR}/4_train.py" \
        --dataset "$REVERSED_NPZ" \
        --out "$PB_WEIGHTS" \
        --lerobot_dataset_dir "$LEROBOT_DIR" \
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

    log "Policy B training complete"

    # Update record
    python3 -c "
import json
with open('$RECORD_FILE') as f:
    rec = json.load(f)
rec['phases']['phase4_train'] = {
    'status': 'done',
    'steps': $STEPS_B,
    'epochs': $TARGET_EPOCHS,
    'batch_size': $BATCH_SIZE,
    'lr': '$LR',
    'num_gpus': $NUM_TRAIN_GPUS,
}
with open('$RECORD_FILE', 'w') as f:
    json.dump(rec, f, indent=2)
"

    mark_done phase4_train
    log "Phase 4 complete: $PB_WEIGHTS"
else
    log "[Skip] Policy B training already done"
fi

# =============================================================================
# Phase 5: Fair Test Policy B
# =============================================================================
print_header "Phase 5: Fair Test Policy B (${FAIR_TEST_EPISODES} episodes)"

if ! phase_done phase5_test; then
    PB_CKPT="${PB_WEIGHTS}/checkpoints/checkpoints/last/pretrained_model"

    if [ ! -d "$PB_CKPT" ]; then
        log "FATAL: Policy B checkpoint not found at $PB_CKPT"
        exit 1
    fi

    log "Testing Policy B..."
    log "  Checkpoint: $PB_CKPT"
    log "  Episodes:   $FAIR_TEST_EPISODES"
    log "  GPU:        $COLLECT_GPU"

    CUDA_VISIBLE_DEVICES=$COLLECT_GPU python -u "${SCRIPT_DIR}/7_eval_fair.py" \
        --policy "$PB_CKPT" \
        --task B \
        --num_episodes $FAIR_TEST_EPISODES \
        --horizon $HORIZON \
        --distance_threshold $DISTANCE_THRESHOLD \
        --n_action_steps $EVAL_N_ACTION_STEPS \
        --goal_xy $GOAL_X $GOAL_Y \
        --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
        --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
        --out "$FAIR_TEST_STATS" \
        $HEADLESS \
        2>&1 | tee "${LOG_DIR}/fair_test_B.log"

    # Update record
    python3 -c "
import json
with open('$RECORD_FILE') as f:
    rec = json.load(f)
if '$FAIR_TEST_STATS':
    with open('$FAIR_TEST_STATS') as f:
        stats = json.load(f)
    rec['phases']['phase5_test'] = {
        'status': 'done',
        'success_rate': stats['success_rate'],
        'num_success': stats['num_success'],
        'num_total': stats['num_total'],
        'elapsed_sec': stats['elapsed_sec'],
    }
with open('$RECORD_FILE', 'w') as f:
    json.dump(rec, f, indent=2)
"

    mark_done phase5_test
    log "Phase 5 complete: $FAIR_TEST_STATS"
else
    log "[Skip] Fair test already done"
fi

# =============================================================================
# Done
# =============================================================================
touch "$BASE_DIR/.complete"

print_header "Pipeline Complete!"
echo "  Experiment:     $BASE_DIR"
echo "  Record:         $RECORD_FILE"
echo ""

# Print summary
python3 << 'PYSUMMARY'
import json
from pathlib import Path

record = Path("RECORD_PLACEHOLDER")
PYSUMMARY

python3 -c "
import json
from pathlib import Path

rf = Path('$RECORD_FILE')
if rf.exists():
    with open(rf) as f:
        rec = json.load(f)
    phases = rec.get('phases', {})

    print('  Pipeline Summary:')
    print('  ' + '-' * 50)

    p2 = phases.get('phase2_rollout', {})
    if p2:
        print(f'  Phase 2 (Rollout):  {p2[\"collected\"]} episodes, '
              f'rate={p2[\"success_rate\"]*100:.1f}%, '
              f'{p2[\"total_frames\"]} frames')

    p4 = phases.get('phase4_train', {})
    if p4:
        print(f'  Phase 4 (Train B):  {p4[\"steps\"]} steps, '
              f'{p4[\"epochs\"]} epochs, '
              f'{p4[\"num_gpus\"]} GPUs')

    p5 = phases.get('phase5_test', {})
    if p5:
        print(f'  Phase 5 (Test B):   {p5[\"num_success\"]}/{p5[\"num_total\"]} = '
              f'{p5[\"success_rate\"]*100:.1f}%')
    print()
"
