#!/bin/bash
# =============================================================================
# DAgger Iterative Training Pipeline (Auto-Resumable)
# =============================================================================
#
# Loop: Evaluate → Aggregate successful rollout data → Finetune → Repeat
#
# Results stored in: data/pick_place_isaac_lab_simulation/exp{N}/
#   - Auto-creates new experiment directory (exp1, exp2, ...)
#   - On crash, re-run this script — auto-resumes from last completed phase
#
# Usage:
#   bash scripts/scripts_pick_place/run_pipeline.sh
#
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
MODE="pipeline"             # pipeline = DAgger (accumulate rollout data)
MAX_ITERATIONS=10           # Total evaluate-train cycles
NUM_CYCLES=50               # A-B eval cycles per iteration
STEPS_PER_ITER=5000         # Training steps per iteration
HORIZON=400                 # Max steps per robot attempt
BATCH_SIZE=32
DISTANCE_THRESHOLD=0.05
N_ACTION_STEPS=16

# Pretrained source policies (read-only, never modified)
POLICY_A_SRC="runs/PP_A_circle"
POLICY_B_SRC="runs/PP_B_circle"

GOAL_X=0.5
GOAL_Y=0.0

# GPUs — single for eval, multi for training
DATA_COLLECTION_GPU=0
TRAINING_GPUS="${CUDA_VISIBLE_DEVICES:-0,4,5,6}"
NUM_TRAINING_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

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

        # Check: same mode + incomplete → candidate for resume
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
# Initialize New Experiment
# =============================================================================
init_experiment() {
    print_header "Initializing Experiment $EXP_NUM ($MODE)"

    # Pre-flight checks
    for d in "$POLICY_A_SRC/lerobot_dataset" "$POLICY_B_SRC/lerobot_dataset" \
             "$(get_ckpt "$POLICY_A_SRC")" "$(get_ckpt "$POLICY_B_SRC")"; do
        [ -d "$d" ] || { echo "ERROR: Not found: $d"; exit 1; }
    done

    # Copy source → temp working directories
    echo "  Copying Policy A → temp"
    mkdir -p "$PA_TEMP"
    cp -r "$POLICY_A_SRC/checkpoints" "$PA_TEMP/checkpoints"
    cp -r "$POLICY_A_SRC/lerobot_dataset" "$PA_TEMP/lerobot_dataset"

    echo "  Copying Policy B → temp"
    mkdir -p "$PB_TEMP"
    cp -r "$POLICY_B_SRC/checkpoints" "$PB_TEMP/checkpoints"
    cp -r "$POLICY_B_SRC/lerobot_dataset" "$PB_TEMP/lerobot_dataset"

    # Write config
    python3 -c "
import json, datetime
json.dump({
    'mode': '$MODE',
    'created_at': datetime.datetime.now().isoformat(),
    'max_iterations': $MAX_ITERATIONS,
    'num_cycles': $NUM_CYCLES,
    'steps_per_iter': $STEPS_PER_ITER,
    'horizon': $HORIZON,
    'batch_size': $BATCH_SIZE,
    'distance_threshold': $DISTANCE_THRESHOLD,
    'n_action_steps': $N_ACTION_STEPS,
    'policy_A_source': '$POLICY_A_SRC',
    'policy_B_source': '$POLICY_B_SRC',
    'goal_xy': [$GOAL_X, $GOAL_Y],
    'training_gpus': '$TRAINING_GPUS',
}, open('$EXP_DIR/config.json', 'w'), indent=2)
"

    # Initialize record
    python3 -c "
import json
json.dump({
    'description': 'Iterative $MODE training',
    'config': json.load(open('$EXP_DIR/config.json')),
    'iterations': []
}, open('$RECORD_FILE', 'w'), indent=2)
"

    echo "✓ Experiment initialized"
}

# =============================================================================
# Append Evaluation Results to Record
# =============================================================================
append_record() {
    local iteration=$1
    local stats_file=$2
    local step_A=$3
    local step_B=$4

    python3 << PYEOF
import json
from pathlib import Path

with open("$RECORD_FILE") as f:
    rec = json.load(f)

entry = {
    "iteration": $iteration,
    "checkpoint_info": {"policy_A_step": $step_A, "policy_B_step": $step_B},
    "performance_metrics": {},
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
    for task, key in [("A", "episodes_A"), ("B", "episodes_B")]:
        eps = st.get(key, [])
        steps = [e["success_step"] for e in eps if e.get("success") and e.get("success_step")]
        pm[f"avg_success_step_{task}"] = sum(steps) / len(steps) if steps else None
else:
    entry["performance_metrics"] = dict(
        task_A_success_rate=0, task_B_success_rate=0,
        task_A_success_count=0, task_B_success_count=0,
        total_task_A_episodes=0, total_task_B_episodes=0,
    )

# Replace existing entry for this iteration (idempotent on re-run)
rec["iterations"] = [i for i in rec["iterations"] if i["iteration"] != $iteration]
rec["iterations"].append(entry)
rec["iterations"].sort(key=lambda x: x["iteration"])

with open("$RECORD_FILE", "w") as f:
    json.dump(rec, f, indent=2)

m = entry["performance_metrics"]
print(f"  Recorded iter $iteration: A={m['task_A_success_rate']*100:.1f}%  B={m['task_B_success_rate']*100:.1f}%")
PYEOF
}

# =============================================================================
# Train One Policy (A or B)
# =============================================================================
train_policy() {
    local NAME=$1        # "A" or "B"
    local TEMP=$2        # temp working directory
    local LAST=$3        # last working directory (will be created)
    local ROLLOUT=$4     # rollout npz file path

    # --- Crash recovery ---
    # If only LAST exists (temp deleted mid-rotation), restore it
    if [ -d "$LAST" ] && [ ! -d "$TEMP" ]; then
        echo "  [Recovery] Restoring temp from last for Policy $NAME"
        mv "$LAST" "$TEMP"
    fi

    local CKPT
    CKPT=$(get_ckpt "$TEMP")

    rm -rf "$LAST"
    mkdir -p "$LAST"

    # --- Prepare dataset + checkpoint ---
    local ROLLOUT_ARG=""
    if [ "$MODE" = "pipeline" ] && [ -n "$ROLLOUT" ] && [ -f "$ROLLOUT" ]; then
        ROLLOUT_ARG="--rollout_data $ROLLOUT"
        echo "  Merging rollout data into dataset"
    else
        echo "  Using original dataset only"
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

    # --- Train ---
    local CUR_STEP TARGET
    CUR_STEP=$(get_step "$LAST")
    TARGET=$((CUR_STEP + STEPS_PER_ITER))
    echo "  Training Policy $NAME: step $CUR_STEP → $TARGET ($NUM_TRAINING_GPUS GPUs)"

    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS \
        scripts/scripts_pick_place/4_train_diffusion.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$LAST/lerobot_dataset" \
        --out "$LAST" \
        --steps $TARGET \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose --include_gripper --wandb

    echo "  ✓ Policy $NAME trained"

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
echo "  Mode:       $MODE"
echo "  Iterations: $MAX_ITERATIONS × $NUM_CYCLES cycles"
echo "  Training:   $STEPS_PER_ITER steps/iter, GPUs=$TRAINING_GPUS ($NUM_TRAINING_GPUS)"
echo ""

# Tee all output to log file (append on resume)
exec > >(tee -a "$EXP_DIR/${MODE}.log") 2>&1

if [ "$IS_RESUME" = false ]; then
    init_experiment
else
    print_header "Resuming Experiment $EXP_NUM ($MODE)"
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
for iter in $(seq 1 $MAX_ITERATIONS); do
    print_header "Iteration $iter / $MAX_ITERATIONS"

    ROLLOUT_A="$EXP_DIR/iter${iter}_eval_A.npz"
    ROLLOUT_B="$EXP_DIR/iter${iter}_eval_B.npz"
    STATS="${ROLLOUT_A%.npz}.stats.json"

    # ---- Phase 1: Evaluate ----
    if ! phase_done $iter eval; then
        echo "--- Evaluate ($NUM_CYCLES A-B cycles) ---"
        CKPT_A=$(get_ckpt "$PA_TEMP")
        CKPT_B=$(get_ckpt "$PB_TEMP")
        STEP_A=$(get_step "$PA_TEMP")
        STEP_B=$(get_step "$PB_TEMP")
        echo "  Policy A (step $STEP_A): $CKPT_A"
        echo "  Policy B (step $STEP_B): $CKPT_B"

        rm -f "$ROLLOUT_A" "$ROLLOUT_B" "$STATS"

        CUDA_VISIBLE_DEVICES=$DATA_COLLECTION_GPU python scripts/scripts_pick_place/9_eval_with_recovery.py \
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
            $HEADLESS

        append_record $iter "$STATS" $STEP_A $STEP_B

        if [ -f "$STATS" ]; then
            python3 -c "
import json
with open('$STATS') as f:
    s = json.load(f)['summary']
print(f\"  A: {s['task_A_success_count']}/{s['total_task_A_episodes']} = {s['task_A_success_rate']*100:.1f}%\")
print(f\"  B: {s['task_B_success_count']}/{s['total_task_B_episodes']} = {s['task_B_success_rate']*100:.1f}%\")
"
        fi

        mark_done $iter eval
    else
        echo "  [Eval] Already done — skipping"
    fi

    # ---- Phase 2: Train Policy A ----
    if ! phase_done $iter train_A; then
        echo "--- Train Policy A ($STEPS_PER_ITER steps) ---"
        train_policy A "$PA_TEMP" "$PA_LAST" "$ROLLOUT_A"
        mark_done $iter train_A
    else
        echo "  [Train A] Already done — skipping"
    fi

    # ---- Phase 3: Train Policy B ----
    if ! phase_done $iter train_B; then
        echo "--- Train Policy B ($STEPS_PER_ITER steps) ---"
        train_policy B "$PB_TEMP" "$PB_LAST" "$ROLLOUT_B"
        mark_done $iter train_B
    else
        echo "  [Train B] Already done — skipping"
    fi

    echo "  ✓ Iteration $iter complete"
done

# =============================================================================
# Done
# =============================================================================
touch "$EXP_DIR/.complete"

print_header "Generating Plot"
python scripts/scripts_pick_place/plot_success_rate.py \
    --record "$RECORD_FILE" \
    --out "$EXP_DIR/success_rate_curve.png"

print_header "Pipeline Complete!"
echo "  Experiment: $EXP_DIR"
echo "  Record:     $RECORD_FILE"
echo "  Plot:       $EXP_DIR/success_rate_curve.png"
echo "  Checkpoints:"
echo "    Policy A: $(get_ckpt "$PA_TEMP")"
echo "    Policy B: $(get_ckpt "$PB_TEMP")"
echo ""

python3 << PYSUMMARY
import json
with open("$RECORD_FILE") as f:
    record = json.load(f)
iters = record.get("iterations", [])
if iters:
    print("  Iter  |  Task A  |  Task B")
    print("  ------|----------|--------")
    for it in iters:
        m = it["performance_metrics"]
        print(f"  {it['iteration']:4d}  |  {m['task_A_success_rate']*100:5.1f}%  |  {m['task_B_success_rate']*100:5.1f}%")
    first_a = iters[0]["performance_metrics"]["task_A_success_rate"] * 100
    last_a  = iters[-1]["performance_metrics"]["task_A_success_rate"] * 100
    first_b = iters[0]["performance_metrics"]["task_B_success_rate"] * 100
    last_b  = iters[-1]["performance_metrics"]["task_B_success_rate"] * 100
    print(f"\n  Task A: {first_a:.1f}% → {last_a:.1f}%")
    print(f"  Task B: {first_b:.1f}% → {last_b:.1f}%")
PYSUMMARY
