#!/bin/bash
# =============================================================================
# Rev2Fwd Pick-and-Place Simulator Pipeline — Exp22 Iterative Extension
# =============================================================================
#
# Continues exp22 from iteration 2 to iteration 10.
# Requires iter 0+1 (the original exp22 pipeline) to be complete.
#
# For each iteration N (2..10):
#   1. Cyclic collection with iter (N-1) models
#      — 5 parallel Isaac Lab instances, 10 cycles each
#   2. Merge collection parts → subsample to 400 frames → reverse
#   3. Train Policy A & B (non-cumulative: original demo + reversed data)
#
# NOTE: Fair test evaluation is SKIPPED in this experiment to save time.
#       Collection success rates (from 6_eval_cyclic.py) are used as
#       approximate success rates instead. This is a temporary decision
#       for exp22 only and should NOT be adopted in future experiments.
#
# Key design:
#   - Training data per iteration = original 100 demo + current iter reversed
#     subsampled rollout (NO accumulation across iterations)
#   - Opposite rollout training: Policy A ← reversed B rollout, B ← reversed A
#   - Step-size amplification via subsampling (400 frames)
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

# --- Region parameters (same as exp20/22) ---
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

# --- Iteration ---
ITER_START=2
ITER_END=10
NUM_COLLECT_PARTS=5
NUM_CYCLES_PER_PART=10

# --- GPU ---
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-0,2,4,5,7}"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)

# Parse individual GPUs for parallel collection and evaluation
IFS=',' read -ra GPU_LIST <<< "$TRAIN_GPUS"
EVAL_GPU_1="${EVAL_GPU_1:-${GPU_LIST[1]}}"
EVAL_GPU_2="${EVAL_GPU_2:-${GPU_LIST[2]}}"

# --- NCCL robustness ---
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

# --- Misc ---
HEADLESS="--headless"
WANDB_PROJECT="${WANDB_PROJECT:-rev2fwd-${EXP_NAME}}"
SEED=42
COLLECT_SEED_BASE=1000

# --- Script paths ---
SCRIPT_DIR="scripts/scripts_pick_place_simulator"

# =============================================================================
# Derived paths
# =============================================================================
LOG_DIR="${BASE_DIR}/logs"
LEROBOT_DIR="${BASE_DIR}/lerobot"
WORK_DIR="${BASE_DIR}/work"

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
# append_record — Record iteration metrics to record.json
# Uses collection success rates (no separate fair test).
# =============================================================================
append_record() {
    local iteration=$1
    local step_A=$2
    local step_B=$3

    python3 << PYEOF
import json, glob
from pathlib import Path

with open("$RECORD_FILE") as f:
    rec = json.load(f)

# Aggregate collection stats across all parts
total_cycles_A, success_A = 0, 0
total_cycles_B, success_B = 0, 0
for j in range($NUM_COLLECT_PARTS):
    sf = Path("$BASE_DIR") / f"iter{$iteration}_collect_A_p{j}.stats.json"
    if sf.exists():
        s = json.load(open(sf)).get("summary", {})
        total_cycles_A += s.get("total_cycles", 0)
        success_A += s.get("task_A_success_count", 0)
        success_B += s.get("task_B_success_count", 0)

rate_A = success_A / total_cycles_A if total_cycles_A > 0 else 0.0
rate_B = success_B / total_cycles_A if total_cycles_A > 0 else 0.0

entry = {
    "iteration": $iteration,
    "checkpoint_info": {"policy_A_step": $step_A, "policy_B_step": $step_B},
    "collection_metrics_as_eval": {
        "note": "Using collection success rates as approximate eval (no separate fair test). Temporary for exp22 only.",
        "total_cycles": total_cycles_A,
        "task_A_success_count": success_A,
        "task_B_success_count": success_B,
        "task_A_success_rate": rate_A,
        "task_B_success_rate": rate_B,
    },
}

# Replace existing entry for this iteration (idempotent on re-run)
rec["iterations"] = [i for i in rec["iterations"] if i["iteration"] != $iteration]
rec["iterations"].append(entry)
rec["iterations"].sort(key=lambda x: x["iteration"])

with open("$RECORD_FILE", "w") as f:
    json.dump(rec, f, indent=2)

print(f"  Recorded iter $iteration: A={success_A}/{total_cycles_A}={rate_A*100:.1f}%  B={success_B}/{total_cycles_A}={rate_B*100:.1f}%  (collection success rates, no fair test)")
PYEOF
}

# =============================================================================
# Initialize
# =============================================================================
mkdir -p "$LOG_DIR"

exec > >(add_timestamps | tee -a "${LOG_DIR}/pipeline_iter.log") 2>&1

print_header "Exp22 Iterative Extension (Iterations ${ITER_START}–${ITER_END})"
echo "  Experiment:   $BASE_DIR"
echo "  Train GPUs:   $TRAIN_GPUS ($NUM_TRAIN_GPUS)"
echo "  Collection:   ${NUM_COLLECT_PARTS} parallel × ${NUM_CYCLES_PER_PART} cycles"
echo "  Subsample:    ${TARGET_FRAMES} frames/episode"
echo "  Steps/iter:   $STEPS_PER_ITER"
echo "  Fair test:    SKIPPED (using collection success rates)"
echo ""

# =============================================================================
# Verify prerequisites (iter 1 must be complete)
# =============================================================================
print_header "Verifying prerequisites"

for marker in phase0_copy phase0_init phase1_subsample phase1_reverse \
              iter1_train_A iter1_train_B iter1_fair_test; do
    if ! phase_done "$marker"; then
        log "FATAL: Missing prerequisite done marker: .done_$marker"
        log "Please run the original exp22 pipeline first."
        exit 1
    fi
done

verify_dir "$PA_TEMP" "work/PP_A_temp" || exit 1
verify_dir "$PB_TEMP" "work/PP_B_temp" || exit 1
verify_dir "${LEROBOT_DIR}/task_A" "lerobot/task_A" || exit 1
verify_dir "${LEROBOT_DIR}/task_B" "lerobot/task_B" || exit 1

log "All prerequisites satisfied."

# Crash recovery for working dirs
for pair in "A:$PA_TEMP:$PA_LAST" "B:$PB_TEMP:$PB_LAST"; do
    IFS=: read -r label temp last <<< "$pair"
    if [ ! -d "$temp" ] && [ -d "$last" ]; then
        log "[Recovery] Restoring temp for Policy $label"
        mv "$last" "$temp"
    fi
done

# =============================================================================
# Main iteration loop
# =============================================================================

for iter in $(seq $ITER_START $ITER_END); do
    prev=$((iter - 1))
    print_header "Iteration $iter / $ITER_END (collecting from iter $prev models)"

    ROLLOUT_A="${BASE_DIR}/iter${iter}_collect_A.npz"
    ROLLOUT_B="${BASE_DIR}/iter${iter}_collect_B.npz"
    ROLLOUT_B_SUB="${BASE_DIR}/iter${iter}_collect_B_subsampled.npz"
    ROLLOUT_A_SUB="${BASE_DIR}/iter${iter}_collect_A_subsampled.npz"
    ROLLOUT_B_SUB_REV="${BASE_DIR}/iter${iter}_collect_B_subsampled_reversed.npz"
    ROLLOUT_A_SUB_REV="${BASE_DIR}/iter${iter}_collect_A_subsampled_reversed.npz"

    # =========================================================================
    # Step 1: Cyclic data collection (5 parallel instances × 10 cycles)
    # =========================================================================
    if ! phase_done "iter${iter}_collect"; then
        CKPT_A=$(get_ckpt "$PA_TEMP")
        CKPT_B=$(get_ckpt "$PB_TEMP")
        STEP_A=$(get_step "$PA_TEMP")
        STEP_B=$(get_step "$PB_TEMP")
        log "Collecting with iter $prev models (A step=$STEP_A, B step=$STEP_B)"
        log "  Policy A: $CKPT_A"
        log "  Policy B: $CKPT_B"
        log "  Launching $NUM_COLLECT_PARTS parallel instances..."

        PIDS=()
        for j in $(seq 0 $((NUM_COLLECT_PARTS - 1))); do
            GPU="${GPU_LIST[$j]}"
            OUT_A="${BASE_DIR}/iter${iter}_collect_A_p${j}.npz"
            OUT_B="${BASE_DIR}/iter${iter}_collect_B_p${j}.npz"
            SEED_J=$((COLLECT_SEED_BASE + iter * 100 + j))

            log "  Instance $j: GPU=$GPU, seed=$SEED_J"

            (
                CUDA_VISIBLE_DEVICES=$GPU python -u "${SCRIPT_DIR}/6_eval_cyclic.py" \
                    --policy_A "$CKPT_A" \
                    --policy_B "$CKPT_B" \
                    --out_A "$OUT_A" \
                    --out_B "$OUT_B" \
                    --num_cycles $NUM_CYCLES_PER_PART \
                    --horizon $HORIZON \
                    --distance_threshold $DISTANCE_THRESHOLD \
                    --n_action_steps $EVAL_N_ACTION_STEPS \
                    --goal_xy $GOAL_X $GOAL_Y \
                    --red_region_center_xy $RED_REGION_CENTER_X $RED_REGION_CENTER_Y \
                    --red_region_size_xy $RED_REGION_SIZE_X $RED_REGION_SIZE_Y \
                    --seed $SEED_J \
                    $HEADLESS \
                    2>&1 | add_timestamps | tee "${LOG_DIR}/iter${iter}_collect_p${j}.log"
            ) &
            PIDS+=($!)
            sleep 30  # Stagger launches for Isaac Lab init
        done

        # Wait for all collection instances
        COLLECT_FAILED=0
        for j in $(seq 0 $((NUM_COLLECT_PARTS - 1))); do
            if ! wait ${PIDS[$j]}; then
                log "ERROR: Collection instance $j failed!"
                COLLECT_FAILED=1
            fi
        done
        if [ $COLLECT_FAILED -ne 0 ]; then
            log "FATAL: One or more collection instances failed at iter $iter"
            exit 1
        fi

        # Merge collection parts into single NPZ per task
        log "Merging $NUM_COLLECT_PARTS collection parts..."
        python3 << MERGE_EOF
import numpy as np
from pathlib import Path

base_dir = "$BASE_DIR"
iter_num = $iter
num_parts = $NUM_COLLECT_PARTS

for task in ['A', 'B']:
    parts = []
    for j in range(num_parts):
        f = Path(base_dir) / f"iter{iter_num}_collect_{task}_p{j}.npz"
        if f.exists():
            data = np.load(f, allow_pickle=True)
            if 'episodes' in data:
                eps = list(data['episodes'])
                parts.extend(eps)
                print(f"  Part {j} task {task}: {len(eps)} episodes")
            else:
                print(f"  Part {j} task {task}: no episodes key")
        else:
            print(f"  Part {j} task {task}: file not found (0 successes)")

    out = Path(base_dir) / f"iter{iter_num}_collect_{task}.npz"
    if parts:
        np.savez_compressed(str(out), episodes=np.array(parts, dtype=object))
        print(f"Merged task {task}: {len(parts)} episodes -> {out.name}")
    else:
        print(f"WARNING: No episodes collected for task {task}!")
MERGE_EOF

        verify_file "$ROLLOUT_A" "iter${iter}_collect_A.npz" || log "  (0 episodes for task A — continuing)"
        verify_file "$ROLLOUT_B" "iter${iter}_collect_B.npz" || log "  (0 episodes for task B — continuing)"

        mark_done "iter${iter}_collect"
    else
        log "[Skip] Iteration $iter collection already done"
    fi

    # =========================================================================
    # Step 2: Subsample + Reverse
    # =========================================================================
    if ! phase_done "iter${iter}_subsample"; then
        log "Subsampling to $TARGET_FRAMES frames/episode..."

        for task in B A; do
            INPUT="${BASE_DIR}/iter${iter}_collect_${task}.npz"
            OUTPUT="${BASE_DIR}/iter${iter}_collect_${task}_subsampled.npz"
            if [ -f "$INPUT" ]; then
                log "  Subsampling ${task} rollout → $(basename "$OUTPUT")"
                python "${SCRIPT_DIR}/3_subsample_episodes.py" \
                    --input "$INPUT" \
                    --out "$OUTPUT" \
                    --target_frames $TARGET_FRAMES \
                    --success_only 1 \
                    2>&1 | add_timestamps | tee "${LOG_DIR}/iter${iter}_subsample_${task}.log"
            else
                log "  WARNING: $INPUT not found, skipping subsample for task $task"
            fi
        done

        verify_file "$ROLLOUT_B_SUB" "iter${iter}_collect_B_subsampled.npz" || true
        verify_file "$ROLLOUT_A_SUB" "iter${iter}_collect_A_subsampled.npz" || true

        mark_done "iter${iter}_subsample"
    else
        log "[Skip] Iteration $iter subsample already done"
    fi

    if ! phase_done "iter${iter}_reverse"; then
        log "Reversing subsampled data..."

        if [ -f "$ROLLOUT_B_SUB" ]; then
            log "  Reversing B subsampled → $(basename "$ROLLOUT_B_SUB_REV") (for Policy A)"
            python "${SCRIPT_DIR}/2_reverse_to_task_A.py" \
                --input "$ROLLOUT_B_SUB" \
                --out "$ROLLOUT_B_SUB_REV" \
                --success_only 1 \
                --verify \
                2>&1 | add_timestamps | tee "${LOG_DIR}/iter${iter}_reverse_B.log"
        else
            log "  WARNING: $ROLLOUT_B_SUB not found, skipping B reversal"
        fi

        if [ -f "$ROLLOUT_A_SUB" ]; then
            log "  Reversing A subsampled → $(basename "$ROLLOUT_A_SUB_REV") (for Policy B)"
            python "${SCRIPT_DIR}/2_reverse_to_task_A.py" \
                --input "$ROLLOUT_A_SUB" \
                --out "$ROLLOUT_A_SUB_REV" \
                --success_only 1 \
                --verify \
                2>&1 | add_timestamps | tee "${LOG_DIR}/iter${iter}_reverse_A.log"
        else
            log "  WARNING: $ROLLOUT_A_SUB not found, skipping A reversal"
        fi

        verify_file "$ROLLOUT_B_SUB_REV" "iter${iter}_collect_B_subsampled_reversed.npz" || true
        verify_file "$ROLLOUT_A_SUB_REV" "iter${iter}_collect_A_subsampled_reversed.npz" || true

        mark_done "iter${iter}_reverse"
    else
        log "[Skip] Iteration $iter reverse already done"
    fi

    # =========================================================================
    # Step 3: Train Policy A & B
    # =========================================================================
    if ! phase_done "iter${iter}_train_A"; then
        log "Training Policy A ($STEPS_PER_ITER steps)..."
        log "  Rollout data: $ROLLOUT_B_SUB_REV"
        log "  Original LeRobot: ${LEROBOT_DIR}/task_A"
        run_fresh_train A "$PA_TEMP" "$PA_LAST" "$ROLLOUT_B_SUB_REV" "${LEROBOT_DIR}/task_A" $STEPS_PER_ITER \
            2>&1 | add_timestamps | tee "${LOG_DIR}/iter${iter}_train_A.log"

        # Save iteration checkpoint
        CKPT_SRC=$(get_ckpt "$PA_TEMP")
        DEST="${BASE_DIR}/iter${iter}_ckpt_A"
        [ -d "$CKPT_SRC" ] && [ ! -d "$DEST" ] && cp -r "$CKPT_SRC" "$DEST"

        mark_done "iter${iter}_train_A"
    else
        log "[Skip] Iteration $iter Policy A training already done"
    fi

    if ! phase_done "iter${iter}_train_B"; then
        log "Training Policy B ($STEPS_PER_ITER steps)..."
        log "  Rollout data: $ROLLOUT_A_SUB_REV"
        log "  Original LeRobot: ${LEROBOT_DIR}/task_B"
        run_fresh_train B "$PB_TEMP" "$PB_LAST" "$ROLLOUT_A_SUB_REV" "${LEROBOT_DIR}/task_B" $STEPS_PER_ITER \
            2>&1 | add_timestamps | tee "${LOG_DIR}/iter${iter}_train_B.log"

        # Save iteration checkpoint
        CKPT_SRC=$(get_ckpt "$PB_TEMP")
        DEST="${BASE_DIR}/iter${iter}_ckpt_B"
        [ -d "$CKPT_SRC" ] && [ ! -d "$DEST" ] && cp -r "$CKPT_SRC" "$DEST"

        mark_done "iter${iter}_train_B"
    else
        log "[Skip] Iteration $iter Policy B training already done"
    fi

    # =========================================================================
    # Step 4: Record collection success rates (NO separate fair test)
    # NOTE: This is a temporary simplification for exp22 only.
    #       Collection success rates ≈ eval success rates.
    # =========================================================================
    if ! phase_done "iter${iter}_record"; then
        STEP_A=$(get_step "$PA_TEMP")
        STEP_B=$(get_step "$PB_TEMP")
        append_record $iter $STEP_A $STEP_B
        mark_done "iter${iter}_record"
    else
        log "[Skip] Iteration $iter record already done"
    fi

    log "Iteration $iter complete"
done

# =============================================================================
# Final Summary
# =============================================================================
print_header "Exp22 Iterative Pipeline Complete (Iterations 1–${ITER_END})"
log "NOTE: Iterations 2–10 use collection success rates (no fair test). Temporary for exp22."
log ""
log "Results summary:"
python3 << SUMMARY_EOF
import json
from pathlib import Path

rec = json.load(open("$RECORD_FILE"))
for entry in rec.get("iterations", []):
    it = entry["iteration"]
    # iter 1 has fair_test_metrics, iter 2+ has collection_metrics_as_eval
    fm = entry.get("fair_test_metrics", {})
    cm = entry.get("collection_metrics_as_eval", {})
    if fm:
        a = fm.get("fair_A_success_rate", 0)
        b = fm.get("fair_B_success_rate", 0)
        an = fm.get("fair_A_num_success", 0)
        at = fm.get("fair_A_num_total", 0)
        bn = fm.get("fair_B_num_success", 0)
        bt = fm.get("fair_B_num_total", 0)
        print(f"  Iter {it}: A={an}/{at}={a*100:.1f}%  B={bn}/{bt}={b*100:.1f}%  (fair test)")
    elif cm:
        a = cm.get("task_A_success_rate", 0)
        b = cm.get("task_B_success_rate", 0)
        sa = cm.get("task_A_success_count", 0)
        sb = cm.get("task_B_success_count", 0)
        tc = cm.get("total_cycles", 0)
        print(f"  Iter {it}: A={sa}/{tc}={a*100:.1f}%  B={sb}/{tc}={b*100:.1f}%  (collection)")
SUMMARY_EOF

echo ""
log "Record: $RECORD_FILE"
log "Done!"
