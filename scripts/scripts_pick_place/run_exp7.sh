#!/bin/bash
# =============================================================================
# Exp7: DAgger Pipeline with DiT Flow Policy
# =============================================================================
#
# Same DAgger pipeline as exp5/exp6 but using DiT Flow policy instead of
# Diffusion Policy. DiT Flow uses:
#   - Transformer architecture (instead of U-Net)
#   - Flow Matching objective (instead of DDPM)
#   - Faster inference (~100 steps vs 100 DDIM steps)
#
# Pipeline overview:
#   Phase 0 — Pretrain base DiT Flow policies A and B from the same data
#             used for PP_A_circle / PP_B_circle (50,000 steps each)
#
#   Iteration 0-9 (3 phases each):
#     Phase 1 — Parallel evaluation (4 GPUs):
#       GPU 0: Data collection — 50 A-B cycles (9_eval_with_recovery.py)
#       GPU 1,4,5: Fair test — ~33 episodes/task each, merged (10_eval_independent.py)
#     Phase 2 — Train Policy A (GPU 0,1,4,5 joint, torchrun distributed)
#     Phase 3 — Train Policy B (GPU 0,1,4,5 joint, torchrun distributed)
#
#   Iteration 10 (final):
#     Fair test on all 4 GPUs — the 11th evaluation point
#
# GPU allocation: 0,1,4,5
#   - Collection: GPU 0 (1 GPU)
#   - Fair test:  GPU 1,4,5 (3 GPUs parallel, sharded)
#   - Training:   GPU 0,1,4,5 (4 GPUs)
#
# Auto-resume: re-run the same command to resume from where it left off.
#
# Usage:
#   nohup bash scripts/scripts_pick_place/run_exp7.sh > exp7_nohup.log 2>&1 &
#
# =============================================================================

set -e

# Use full path for conda (needed when running via nohup)
__CONDA_PREFIX="/mnt/dongxu-fs1/data-ssd/qiyuanqiao/conda/miniconda3"
source "$__CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
MODE="pipeline_fair_ditflow"  # DiT Flow DAgger pipeline with fair testing
MAX_ITERATIONS=10             # Iterations 0-9 collect+train; iteration 10 test-only
NUM_CYCLES=50                 # A-B eval cycles per iteration (data collection)
NUM_FAIR_TEST_EPISODES=100    # Independent episodes per task (fair test)
STEPS_PER_ITER=5000           # Training steps per DAgger iteration
PRETRAIN_STEPS=50000          # Steps for initial base policy pretraining
HORIZON=400                   # Max steps per robot attempt
BATCH_SIZE=32
DISTANCE_THRESHOLD=0.03
N_ACTION_STEPS=16

# Source data — reuse the same lerobot_dataset from diffusion pretraining
# (the data format is identical; only the policy architecture differs)
DATA_A_SRC="runs/PP_A_circle/lerobot_dataset"
DATA_B_SRC="runs/PP_B_circle/lerobot_dataset"

GOAL_X=0.5
GOAL_Y=0.0

# GPUs — 0,1,4,5 are all available
COLLECT_GPU=0                  # GPU for data collection
FAIR_TEST_GPUS=(1 4 5)         # GPUs for parallel fair testing (one env per GPU)
NUM_FAIR_TEST_GPUS=${#FAIR_TEST_GPUS[@]}
TRAINING_GPUS="0,1,4,5"        # GPUs for distributed training
NUM_TRAINING_GPUS=4
FAIR_TEST_GPU_LIST=$(printf "%s," "${FAIR_TEST_GPUS[@]}" | sed 's/,$//')

# NCCL robustness (prevents hangs on multi-GPU training)
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

HEADLESS="--headless"
SAVE_VIDEO=""               # Set to "--save_video" to capture evaluation videos

# =============================================================================
# Base path
# =============================================================================
BASE_DIR="data/pick_place_isaac_lab_simulation"

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

phase_done() { [ -f "$EXP_DIR/.done_iter${1}_${2}" ]; }
mark_done()  { touch "$EXP_DIR/.done_iter${1}_${2}"; }

save_iter_checkpoint() {
    local ITER=$1
    local NAME=$2
    local TEMP=$3
    local CKPT_SRC
    CKPT_SRC=$(get_ckpt "$TEMP")
    local DEST="$EXP_DIR/iter${ITER}_ckpt_${NAME}"
    if [ -d "$CKPT_SRC" ] && [ ! -d "$DEST" ]; then
        echo "  Saving checkpoint: $DEST"
        cp -r "$CKPT_SRC" "$DEST"
    elif [ -d "$DEST" ]; then
        echo "  Checkpoint already saved: $DEST"
    else
        echo "  WARNING: No checkpoint found at $CKPT_SRC"
    fi
}

# =============================================================================
# Find or Create Experiment Directory
# =============================================================================
find_or_create_exp() {
    mkdir -p "$BASE_DIR"
    local max_num=0
    local resume_num=0

    for d in "$BASE_DIR"/exp*/; do
        [ -d "$d" ] || continue
        local n
        n=$(basename "$d")
        n=${n#exp}
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        [ "$n" -gt "$max_num" ] && max_num=$n

        if [ -f "$d/config.json" ] && [ ! -f "$d/.complete" ]; then
            local m
            m=$(python3 -c "import json; print(json.load(open('${d}config.json'))['mode'])")
            if [ "$m" = "$MODE" ] && [ "$n" -gt "$resume_num" ]; then
                resume_num=$n
            fi
        fi
    done

    if [ "$resume_num" -gt 0 ]; then
        EXP_NUM=$resume_num
        EXP_DIR="$BASE_DIR/exp${EXP_NUM}"
        IS_RESUME=true
        echo "Resuming experiment: $EXP_DIR"
    else
        EXP_NUM=$((max_num + 1))
        EXP_DIR="$BASE_DIR/exp${EXP_NUM}"
        IS_RESUME=false
        mkdir -p "$EXP_DIR"
        echo "New experiment: $EXP_DIR"
    fi
}

# =============================================================================
# Pretrain Base DiT Flow Policies
# =============================================================================
pretrain_base_policies() {
    print_header "Phase 0: Pretrain Base DiT Flow Policies ($PRETRAIN_STEPS steps)"

    local PRETRAIN_A="runs/exp${EXP_NUM}_ditflow_pretrain_A"
    local PRETRAIN_B="runs/exp${EXP_NUM}_ditflow_pretrain_B"

    # --- Pretrain Policy A ---
    if [ ! -f "$EXP_DIR/.done_pretrain_A" ]; then
        echo "  Training base DiT Flow Policy A ($PRETRAIN_STEPS steps)..."

        # Copy source lerobot_dataset
        mkdir -p "$PRETRAIN_A"
        if [ ! -d "$PRETRAIN_A/lerobot_dataset" ]; then
            cp -r "$DATA_A_SRC" "$PRETRAIN_A/lerobot_dataset"
        fi

        local MASTER_PORT
        MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

        CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS --master_port=$MASTER_PORT \
            scripts/scripts_pick_place/4_train_ditflow.py \
            --dataset dummy.npz \
            --lerobot_dataset_dir "$PRETRAIN_A/lerobot_dataset" \
            --out "$PRETRAIN_A" \
            --steps $PRETRAIN_STEPS \
            --batch_size $BATCH_SIZE \
            --n_action_steps $N_ACTION_STEPS \
            --save_freq 10000 \
            --skip_convert \
            --include_obj_pose --include_gripper --wandb \
            > "$EXP_DIR/pretrain_A.log" 2>&1

        touch "$EXP_DIR/.done_pretrain_A"
        echo "  ✓ Policy A pretrained"
    else
        echo "  [Pretrain A] Already done — skipping"
    fi

    # --- Pretrain Policy B ---
    if [ ! -f "$EXP_DIR/.done_pretrain_B" ]; then
        echo "  Training base DiT Flow Policy B ($PRETRAIN_STEPS steps)..."

        mkdir -p "$PRETRAIN_B"
        if [ ! -d "$PRETRAIN_B/lerobot_dataset" ]; then
            cp -r "$DATA_B_SRC" "$PRETRAIN_B/lerobot_dataset"
        fi

        local MASTER_PORT
        MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

        CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS --master_port=$MASTER_PORT \
            scripts/scripts_pick_place/4_train_ditflow.py \
            --dataset dummy.npz \
            --lerobot_dataset_dir "$PRETRAIN_B/lerobot_dataset" \
            --out "$PRETRAIN_B" \
            --steps $PRETRAIN_STEPS \
            --batch_size $BATCH_SIZE \
            --n_action_steps $N_ACTION_STEPS \
            --save_freq 10000 \
            --skip_convert \
            --include_obj_pose --include_gripper --wandb \
            > "$EXP_DIR/pretrain_B.log" 2>&1

        touch "$EXP_DIR/.done_pretrain_B"
        echo "  ✓ Policy B pretrained"
    else
        echo "  [Pretrain B] Already done — skipping"
    fi

    # Set these as the source policies for the DAgger loop
    POLICY_A_SRC="$PRETRAIN_A"
    POLICY_B_SRC="$PRETRAIN_B"
}

# =============================================================================
# Initialize New Experiment
# =============================================================================
init_experiment() {
    print_header "Initializing Experiment $EXP_NUM ($MODE)"

    # Pre-flight checks: verify source data exists
    for d in "$DATA_A_SRC" "$DATA_B_SRC"; do
        [ -d "$d" ] || { echo "ERROR: Source data not found: $d"; exit 1; }
    done

    # Write config
    python3 -c "
import json, datetime
json.dump({
    'mode': '$MODE',
    'policy_type': 'ditflow',
    'created_at': datetime.datetime.now().isoformat(),
    'max_iterations': $MAX_ITERATIONS,
    'pretrain_steps': $PRETRAIN_STEPS,
    'num_cycles': $NUM_CYCLES,
    'num_fair_test_episodes': $NUM_FAIR_TEST_EPISODES,
    'steps_per_iter': $STEPS_PER_ITER,
    'horizon': $HORIZON,
    'batch_size': $BATCH_SIZE,
    'distance_threshold': $DISTANCE_THRESHOLD,
    'n_action_steps': $N_ACTION_STEPS,
    'data_A_source': '$DATA_A_SRC',
    'data_B_source': '$DATA_B_SRC',
    'goal_xy': [$GOAL_X, $GOAL_Y],
    'collect_gpu': $COLLECT_GPU,
    'fair_test_gpus': [$FAIR_TEST_GPU_LIST],
    'training_gpus': '$TRAINING_GPUS',
}, open('$EXP_DIR/config.json', 'w'), indent=2)
"

    # Initialize record
    python3 -c "
import json
json.dump({
    'description': 'DiT Flow DAgger pipeline with parallel fair testing (11 data points, iter 0-10)',
    'config': json.load(open('$EXP_DIR/config.json')),
    'iterations': []
}, open('$RECORD_FILE', 'w'), indent=2)
"

    # Pretrain base policies
    pretrain_base_policies

    # Copy pretrained policies → temp working directories
    echo "  Copying pretrained DiT Flow Policy A → temp"
    mkdir -p "$PA_TEMP"
    cp -r "$POLICY_A_SRC/checkpoints" "$PA_TEMP/checkpoints"
    cp -r "$POLICY_A_SRC/lerobot_dataset" "$PA_TEMP/lerobot_dataset"

    echo "  Copying pretrained DiT Flow Policy B → temp"
    mkdir -p "$PB_TEMP"
    cp -r "$POLICY_B_SRC/checkpoints" "$PB_TEMP/checkpoints"
    cp -r "$POLICY_B_SRC/lerobot_dataset" "$PB_TEMP/lerobot_dataset"

    echo "✓ Experiment initialized"
}

# =============================================================================
# Append Collection Results to Record
# =============================================================================
append_collection_record() {
    local iteration=$1
    local stats_file=$2
    local step_A=$3
    local step_B=$4

    python3 << PYEOF
import json
from pathlib import Path

with open("$RECORD_FILE") as f:
    rec = json.load(f)

entry = None
for e in rec["iterations"]:
    if e["iteration"] == $iteration:
        entry = e
        break
if entry is None:
    entry = {"iteration": $iteration, "checkpoint_info": {}, "collection_metrics": {}, "fair_test_metrics": {}}
    rec["iterations"].append(entry)

entry["checkpoint_info"] = {"policy_A_step": $step_A, "policy_B_step": $step_B}

sf = Path("$stats_file")
if sf.exists():
    with open(sf) as f:
        st = json.load(f)
    s = st.get("summary", {})
    cm = {}
    for k in ("task_A_success_rate", "task_B_success_rate",
              "task_A_success_count", "task_B_success_count",
              "total_task_A_episodes", "total_task_B_episodes",
              "total_elapsed_seconds"):
        cm[k] = s.get(k, 0)
    for task, key in [("A", "episodes_A"), ("B", "episodes_B")]:
        eps = st.get(key, [])
        steps = [e["success_step"] for e in eps if e.get("success") and e.get("success_step")]
        cm[f"avg_success_step_{task}"] = sum(steps) / len(steps) if steps else None
    entry["collection_metrics"] = cm
else:
    entry["collection_metrics"] = dict(
        task_A_success_rate=0, task_B_success_rate=0,
        task_A_success_count=0, task_B_success_count=0,
        total_task_A_episodes=0, total_task_B_episodes=0,
    )

rec["iterations"].sort(key=lambda x: x["iteration"])
with open("$RECORD_FILE", "w") as f:
    json.dump(rec, f, indent=2)

cm = entry["collection_metrics"]
print(f"  Recorded collection iter $iteration: A={cm.get('task_A_success_rate',0)*100:.1f}%  B={cm.get('task_B_success_rate',0)*100:.1f}%")
PYEOF
}

# =============================================================================
# Append Fair Test Results to Record
# =============================================================================
append_fair_test_record() {
    local iteration=$1
    local stats_file=$2
    local step_A=$3
    local step_B=$4

    python3 << PYEOF
import json
from pathlib import Path

with open("$RECORD_FILE") as f:
    rec = json.load(f)

entry = None
for e in rec["iterations"]:
    if e["iteration"] == $iteration:
        entry = e
        break
if entry is None:
    entry = {"iteration": $iteration, "checkpoint_info": {}, "collection_metrics": {}, "fair_test_metrics": {}}
    rec["iterations"].append(entry)

if not entry.get("checkpoint_info"):
    entry["checkpoint_info"] = {"policy_A_step": $step_A, "policy_B_step": $step_B}

sf = Path("$stats_file")
if sf.exists():
    with open(sf) as f:
        st = json.load(f)
    s = st.get("summary", {})
    ft = {
        "task_A_success_rate": s.get("task_A_success_rate", 0),
        "task_B_success_rate": s.get("task_B_success_rate", 0),
        "task_A_success_count": s.get("task_A_success_count", 0),
        "task_B_success_count": s.get("task_B_success_count", 0),
        "total_task_A_episodes": s.get("task_A_total_episodes", s.get("total_task_A_episodes", 0)),
        "total_task_B_episodes": s.get("task_B_total_episodes", s.get("total_task_B_episodes", 0)),
        "total_elapsed_seconds": s.get("total_elapsed_seconds", 0),
        "avg_success_step_A": s.get("avg_success_step_A"),
        "avg_success_step_B": s.get("avg_success_step_B"),
    }
    entry["fair_test_metrics"] = ft
else:
    entry["fair_test_metrics"] = dict(
        task_A_success_rate=0, task_B_success_rate=0,
        task_A_success_count=0, task_B_success_count=0,
        total_task_A_episodes=0, total_task_B_episodes=0,
    )

rec["iterations"].sort(key=lambda x: x["iteration"])
with open("$RECORD_FILE", "w") as f:
    json.dump(rec, f, indent=2)

ft = entry["fair_test_metrics"]
print(f"  Recorded fair test iter $iteration: A={ft.get('task_A_success_rate',0)*100:.1f}%  B={ft.get('task_B_success_rate',0)*100:.1f}%")
PYEOF
}

# =============================================================================
# Train One Policy (A or B) — DAgger: merge rollout data, train with DiT Flow
# =============================================================================
train_policy() {
    local NAME=$1        # "A" or "B"
    local TEMP=$2        # temp working directory
    local LAST=$3        # last working directory (will be created)
    local ROLLOUT=$4     # rollout npz file path

    # --- Crash recovery ---
    if [ -d "$LAST" ] && [ ! -d "$TEMP" ]; then
        echo "  [Recovery] Restoring temp from last for Policy $NAME"
        mv "$LAST" "$TEMP"
    fi

    local CKPT
    CKPT=$(get_ckpt "$TEMP")

    rm -rf "$LAST"
    mkdir -p "$LAST"

    # --- Prepare dataset + checkpoint (reuse 7_finetune_with_rollout.py) ---
    local ROLLOUT_ARG=""
    if [ -n "$ROLLOUT" ] && [ -f "$ROLLOUT" ]; then
        ROLLOUT_ARG="--rollout_data $ROLLOUT"
        echo "  Merging rollout data into dataset (DAgger)"
    else
        echo "  Using original dataset only (no rollout data available)"
    fi

    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_lerobot "$TEMP/lerobot_dataset" \
        $ROLLOUT_ARG \
        --checkpoint "$CKPT" \
        --out "$LAST" \
        --prepare_only --include_obj_pose --include_gripper

    # --- Copy wandb for resumed logging ---
    if [ -d "$TEMP/checkpoints/wandb" ]; then
        cp -r "$TEMP/checkpoints/wandb" "$LAST/checkpoints/wandb"
        local WANDB_LATEST="$LAST/checkpoints/wandb/latest-run"
        if [ -d "$WANDB_LATEST" ]; then
            for mf in "$WANDB_LATEST"/files/wandb-metadata.json; do
                [ -f "$mf" ] && sed -i "s|$TEMP|$LAST|g" "$mf"
            done
        fi
    fi

    # --- Convert 'last' checkpoint dir → numbered symlink ---
    local CKPT_DIR="$LAST/checkpoints/checkpoints"
    local LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        local CUR_STEP
        CUR_STEP=$(get_step "$LAST")
        local STEP_NAME
        STEP_NAME=$(printf "%06d" "$CUR_STEP")
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  Symlink: last → $STEP_NAME"
    fi

    # --- Train with DiT Flow (key difference from run_ablation.sh) ---
    local CUR_STEP TARGET
    CUR_STEP=$(get_step "$LAST")
    TARGET=$((CUR_STEP + STEPS_PER_ITER))
    echo "  Training Policy $NAME (DiT Flow): step $CUR_STEP → $TARGET ($NUM_TRAINING_GPUS GPUs)"

    local MASTER_PORT
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
    echo "  Using master_port=$MASTER_PORT"

    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS --master_port=$MASTER_PORT \
        scripts/scripts_pick_place/4_train_ditflow.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$LAST/lerobot_dataset" \
        --out "$LAST" \
        --steps $TARGET \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose --include_gripper --wandb

    echo "  ✓ Policy $NAME trained (DiT Flow)"

    # --- Rotate: last → temp ---
    rm -rf "$TEMP"
    mv "$LAST" "$TEMP"
    echo "  ✓ Rotated: last → temp"
}

# =============================================================================
# MAIN
# =============================================================================
find_or_create_exp

# Scoped working directories (per-experiment, avoids collisions)
PA_TEMP="runs/exp${EXP_NUM}_PP_A_temp"
PB_TEMP="runs/exp${EXP_NUM}_PP_B_temp"
PA_LAST="runs/exp${EXP_NUM}_PP_A_last"
PB_LAST="runs/exp${EXP_NUM}_PP_B_last"
RECORD_FILE="$EXP_DIR/record.json"

echo ""
echo "  Experiment: $EXP_DIR  (resume=$IS_RESUME)"
echo "  Mode:       $MODE (DiT Flow)"
echo "  Iterations: 0-$MAX_ITERATIONS (11 fair-test points, 10 training rounds)"
echo "  Collection: $NUM_CYCLES A-B cycles/iter on GPU $COLLECT_GPU"
echo "  Fair test:  $NUM_FAIR_TEST_EPISODES episodes/task on GPUs ${FAIR_TEST_GPUS[*]} ($NUM_FAIR_TEST_GPUS parallel)"
echo "  Training:   $STEPS_PER_ITER steps/iter, GPUs=$TRAINING_GPUS ($NUM_TRAINING_GPUS)"
echo "  Pretrain:   $PRETRAIN_STEPS steps (base DiT Flow policies)"
echo ""

# Tee all output to log file (append on resume), with timestamps
exec > >(add_timestamps | tee -a "$EXP_DIR/${MODE}.log") 2>&1

if [ "$IS_RESUME" = false ]; then
    init_experiment
else
    print_header "Resuming Experiment $EXP_NUM ($MODE)"

    # On resume, re-derive POLICY_A_SRC / POLICY_B_SRC for pretrain check
    POLICY_A_SRC="runs/exp${EXP_NUM}_ditflow_pretrain_A"
    POLICY_B_SRC="runs/exp${EXP_NUM}_ditflow_pretrain_B"

    # If pretraining wasn't finished, complete it
    if [ ! -f "$EXP_DIR/.done_pretrain_A" ] || [ ! -f "$EXP_DIR/.done_pretrain_B" ]; then
        pretrain_base_policies
    fi

    # If temp dirs don't exist yet (pretrain just finished on resume), copy from pretrain
    if [ ! -d "$PA_TEMP" ] && [ ! -d "$PA_LAST" ]; then
        echo "  Copying pretrained DiT Flow Policy A → temp"
        mkdir -p "$PA_TEMP"
        cp -r "$POLICY_A_SRC/checkpoints" "$PA_TEMP/checkpoints"
        cp -r "$POLICY_A_SRC/lerobot_dataset" "$PA_TEMP/lerobot_dataset"
    fi
    if [ ! -d "$PB_TEMP" ] && [ ! -d "$PB_LAST" ]; then
        echo "  Copying pretrained DiT Flow Policy B → temp"
        mkdir -p "$PB_TEMP"
        cp -r "$POLICY_B_SRC/checkpoints" "$PB_TEMP/checkpoints"
        cp -r "$POLICY_B_SRC/lerobot_dataset" "$PB_TEMP/lerobot_dataset"
    fi

    # Verify temp dirs exist (or recover from crash)
    for pair in "A:$PA_TEMP:$PA_LAST" "B:$PB_TEMP:$PB_LAST"; do
        IFS=: read -r label temp last <<< "$pair"
        if [ ! -d "$temp" ] && [ -d "$last" ]; then
            echo "  [Recovery] Restoring temp for Policy $label"
            mv "$last" "$temp"
        fi
        [ -d "$temp" ] || { echo "ERROR: Working dir missing: $temp"; exit 1; }
    done
fi

# ========================== Main Loop ==========================
for iter in $(seq 0 $MAX_ITERATIONS); do
    IS_LAST_ITER=false
    [ "$iter" -eq "$MAX_ITERATIONS" ] && IS_LAST_ITER=true

    if [ "$IS_LAST_ITER" = true ]; then
        print_header "Iteration $iter / $MAX_ITERATIONS (FINAL — fair test only)"
    else
        print_header "Iteration $iter / $MAX_ITERATIONS"
    fi

    CKPT_A=$(get_ckpt "$PA_TEMP")
    CKPT_B=$(get_ckpt "$PB_TEMP")
    STEP_A=$(get_step "$PA_TEMP")
    STEP_B=$(get_step "$PB_TEMP")

    ROLLOUT_A="$EXP_DIR/iter${iter}_collect_A.npz"
    ROLLOUT_B="$EXP_DIR/iter${iter}_collect_B.npz"
    COLLECT_STATS="${ROLLOUT_A%.npz}.stats.json"
    FAIR_TEST_STATS="$EXP_DIR/iter${iter}_fair_test.stats.json"

    # ---- Phase 1: Parallel Collection + Fair Test ----
    NEED_COLLECT=false
    NEED_FAIR_TEST=false

    if [ "$IS_LAST_ITER" = false ] && ! phase_done $iter collect; then
        NEED_COLLECT=true
    fi
    if ! phase_done $iter fair_test; then
        NEED_FAIR_TEST=true
    fi

    if [ "$NEED_COLLECT" = true ] || [ "$NEED_FAIR_TEST" = true ]; then
        echo "--- Phase 1: Parallel Evaluation ---"
        echo "  Policy A (step $STEP_A): $CKPT_A"
        echo "  Policy B (step $STEP_B): $CKPT_B"

        COLLECT_PID=""

        # Launch data collection (background, 1 GPU) — skip for last iteration
        if [ "$NEED_COLLECT" = true ]; then
            echo "  [GPU $COLLECT_GPU] Starting data collection ($NUM_CYCLES A-B cycles)..."
            rm -f "$ROLLOUT_A" "$ROLLOUT_B" "$COLLECT_STATS"

            CUDA_VISIBLE_DEVICES=$COLLECT_GPU python scripts/scripts_pick_place/9_eval_with_recovery.py \
                --policy_A "$CKPT_A" \
                --policy_B "$CKPT_B" \
                --out_A "$ROLLOUT_A" \
                --out_B "$ROLLOUT_B" \
                --num_cycles $NUM_CYCLES \
                --horizon $HORIZON \
                --distance_threshold $DISTANCE_THRESHOLD \
                --n_action_steps $N_ACTION_STEPS \
                --goal_xy $GOAL_X $GOAL_Y \
                $SAVE_VIDEO \
                $HEADLESS \
                > "$EXP_DIR/iter${iter}_collect.log" 2>&1 &
            COLLECT_PID=$!
            echo "  [GPU $COLLECT_GPU] Collection PID: $COLLECT_PID"
        fi

        # Launch parallel fair tests (one per GPU, sharded)
        if [ "$NEED_FAIR_TEST" = true ]; then
            EPISODES_PER_GPU=$((NUM_FAIR_TEST_EPISODES / NUM_FAIR_TEST_GPUS))
            echo "  Starting parallel fair test: $NUM_FAIR_TEST_GPUS GPUs × $EPISODES_PER_GPU episodes = $((EPISODES_PER_GPU * NUM_FAIR_TEST_GPUS)) total"
            rm -f "$FAIR_TEST_STATS"

            FAIR_TEST_PIDS=()
            FAIR_TEST_SHARD_FILES=()
            for shard_idx in $(seq 0 $((NUM_FAIR_TEST_GPUS - 1))); do
                GPU_ID=${FAIR_TEST_GPUS[$shard_idx]}
                SHARD_STATS="$EXP_DIR/iter${iter}_fair_test_shard${shard_idx}.stats.json"
                SHARD_SEED=$((iter * 100 + shard_idx))
                FAIR_TEST_SHARD_FILES+=("$SHARD_STATS")
                rm -f "$SHARD_STATS"

                echo "  [GPU $GPU_ID] Shard $shard_idx: $EPISODES_PER_GPU episodes, seed=$SHARD_SEED"
                CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/scripts_pick_place/10_eval_independent.py \
                    --policy_A "$CKPT_A" \
                    --policy_B "$CKPT_B" \
                    --out "$SHARD_STATS" \
                    --num_episodes $EPISODES_PER_GPU \
                    --horizon $HORIZON \
                    --distance_threshold $DISTANCE_THRESHOLD \
                    --n_action_steps $N_ACTION_STEPS \
                    --goal_xy $GOAL_X $GOAL_Y \
                    --seed $SHARD_SEED \
                    $HEADLESS \
                    > "$EXP_DIR/iter${iter}_fair_test_shard${shard_idx}.log" 2>&1 &
                FAIR_TEST_PIDS+=($!)
                echo "  [GPU $GPU_ID] PID: ${FAIR_TEST_PIDS[-1]}"
            done
        fi

        # Wait for all processes
        echo "  Waiting for parallel tasks to complete..."
        set +e
        COLLECT_RC=0
        FAIR_TEST_RC=0

        if [ -n "$COLLECT_PID" ]; then
            wait $COLLECT_PID
            COLLECT_RC=$?
            if [ $COLLECT_RC -eq 0 ]; then
                echo "  ✓ Data collection finished successfully"
            else
                echo "  ✗ Data collection FAILED (exit code $COLLECT_RC)"
                echo "    See: $EXP_DIR/iter${iter}_collect.log"
            fi
        fi

        if [ "$NEED_FAIR_TEST" = true ]; then
            for pidx in $(seq 0 $((${#FAIR_TEST_PIDS[@]} - 1))); do
                wait ${FAIR_TEST_PIDS[$pidx]}
                rc=$?
                if [ $rc -eq 0 ]; then
                    echo "  ✓ Fair test shard $pidx finished successfully"
                else
                    echo "  ✗ Fair test shard $pidx FAILED (exit code $rc)"
                    echo "    See: $EXP_DIR/iter${iter}_fair_test_shard${pidx}.log"
                    FAIR_TEST_RC=$rc
                fi
            done

            # Merge shard results
            if [ $FAIR_TEST_RC -eq 0 ]; then
                echo "  Merging $NUM_FAIR_TEST_GPUS fair test shards..."
                python scripts/scripts_pick_place/merge_fair_test_stats.py \
                    --inputs ${FAIR_TEST_SHARD_FILES[*]} \
                    --out "$FAIR_TEST_STATS"
            fi
        fi
        set -e

        # Abort on failure
        if [ $COLLECT_RC -ne 0 ] || [ $FAIR_TEST_RC -ne 0 ]; then
            echo "ERROR: One or more parallel tasks failed. Aborting."
            exit 1
        fi

        if [ "$NEED_COLLECT" = true ]; then
            append_collection_record $iter "$COLLECT_STATS" $STEP_A $STEP_B

            if [ -f "$COLLECT_STATS" ]; then
                python3 -c "
import json
with open('$COLLECT_STATS') as f:
    s = json.load(f)['summary']
print(f\"  Collection: A={s['task_A_success_count']}/{s['total_task_A_episodes']} = {s['task_A_success_rate']*100:.1f}%  B={s['task_B_success_count']}/{s['total_task_B_episodes']} = {s['task_B_success_rate']*100:.1f}%\")
"
            fi
            mark_done $iter collect
        fi

        if [ "$NEED_FAIR_TEST" = true ]; then
            append_fair_test_record $iter "$FAIR_TEST_STATS" $STEP_A $STEP_B

            if [ -f "$FAIR_TEST_STATS" ]; then
                python3 -c "
import json
with open('$FAIR_TEST_STATS') as f:
    s = json.load(f)['summary']
print(f\"  Fair test:  A={s['task_A_success_count']}/{s['task_A_total_episodes']} = {s['task_A_success_rate']*100:.1f}%  B={s['task_B_success_count']}/{s['task_B_total_episodes']} = {s['task_B_success_rate']*100:.1f}%\")
"
            fi
            mark_done $iter fair_test
        fi
    else
        echo "  [Phase 1] Already done — skipping"
    fi

    # ---- Skip training on last iteration ----
    if [ "$IS_LAST_ITER" = true ]; then
        echo "  ✓ Iteration $iter complete (test only, no training)"
        echo "  Updating plots..."
        python scripts/scripts_pick_place/plot_success_rate.py \
            --record "$RECORD_FILE" \
            --out "$EXP_DIR/fair_test_curve.png" \
            --metrics_key fair_test_metrics 2>/dev/null || true
        python scripts/scripts_pick_place/plot_success_rate.py \
            --record "$RECORD_FILE" \
            --out "$EXP_DIR/collection_curve.png" \
            --metrics_key collection_metrics 2>/dev/null || true
        echo "  ✓ Plots saved to $EXP_DIR/"
        continue
    fi

    # ---- Phase 2: Train Policy A (DAgger with DiT Flow) ----
    if ! phase_done $iter train_A; then
        echo "--- Train Policy A (DiT Flow, $STEPS_PER_ITER steps, DAgger) ---"
        train_policy A "$PA_TEMP" "$PA_LAST" "$ROLLOUT_A"
        save_iter_checkpoint $iter A "$PA_TEMP"
        mark_done $iter train_A
    else
        echo "  [Train A] Already done — skipping"
    fi

    # ---- Phase 3: Train Policy B (DAgger with DiT Flow) ----
    if ! phase_done $iter train_B; then
        echo "--- Train Policy B (DiT Flow, $STEPS_PER_ITER steps, DAgger) ---"
        train_policy B "$PB_TEMP" "$PB_LAST" "$ROLLOUT_B"
        save_iter_checkpoint $iter B "$PB_TEMP"
        mark_done $iter train_B
    else
        echo "  [Train B] Already done — skipping"
    fi

    echo "  ✓ Iteration $iter complete"

    # ---- Update plots after every iteration ----
    echo "  Updating plots..."
    python scripts/scripts_pick_place/plot_success_rate.py \
        --record "$RECORD_FILE" \
        --out "$EXP_DIR/fair_test_curve.png" \
        --metrics_key fair_test_metrics 2>/dev/null || true
    python scripts/scripts_pick_place/plot_success_rate.py \
        --record "$RECORD_FILE" \
        --out "$EXP_DIR/collection_curve.png" \
        --metrics_key collection_metrics 2>/dev/null || true
    echo "  ✓ Plots saved to $EXP_DIR/"
done

# ========================== Done ==========================
touch "$EXP_DIR/.complete"
print_header "Experiment $EXP_NUM Complete!"
echo "  Results: $EXP_DIR"
echo "  Record:  $RECORD_FILE"
echo "  Plots:   $EXP_DIR/fair_test_curve.png"
echo "           $EXP_DIR/collection_curve.png"
