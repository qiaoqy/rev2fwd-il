#!/bin/bash
# =============================================================================
# DAgger Iterative Training Pipeline (Auto-Resumable)
# =============================================================================
#
# 概述:
#   DAgger (Dataset Aggregation) 迭代训练管线。每轮迭代:
#     1. 评估: 用当前 Policy A 和 B 在仿真中跑 N 个 A-B cycle, 收集成功的 rollout 数据
#     2. 训练 A: 将 rollout 数据合并到原始数据集, 对 Policy A 微调
#     3. 训练 B: 同理微调 Policy B
#   重复以上循环, 策略性能逐步提升。
#
# 目录结构:
#   data/pick_place_isaac_lab_simulation/exp{N}/   — 实验结果
#     config.json          — 实验配置
#     record.json          — 每轮迭代的成功率记录
#     iter{i}_eval_A.npz   — 第 i 轮 Policy A 的 rollout 数据
#     iter{i}_eval_B.npz   — 第 i 轮 Policy B 的 rollout 数据
#     .done_iter{i}_{phase} — 阶段完成标记 (用于断点恢复)
#     success_rate_curve.png — 训练结束后的成功率曲线图
#   runs/exp{N}_PP_{A,B}_{temp,last}/              — 工作目录 (checkpoint + dataset)
#
# 自动恢复:
#   脚本通过 .done_iter{i}_{phase} 标记文件跟踪进度。
#   如果中途崩溃, 直接重新运行同一命令即可自动从上次完成的阶段恢复。
#
# =============================================================================
# 使用示例
# =============================================================================
#
# 1) 基本用法 — 使用 CUDA_VISIBLE_DEVICES 指定训练 GPU:
#
#      CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/scripts_pick_place/run_pipeline.sh
#
#    说明: 评估用 GPU 0, 训练用 GPU 0,1,2,3 (4 卡并行)。
#          如果不设 CUDA_VISIBLE_DEVICES, 默认使用 0,4,5,6。
#
# 2) 后台运行 + 日志输出:
#
#      CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/scripts_pick_place/run_pipeline.sh \
#          > pipeline_run.log 2>&1 &
#
#    说明: 适合长时间运行。脚本内部也会自动将输出 tee 到
#          data/pick_place_isaac_lab_simulation/exp{N}/pipeline.log。
#
# 3) 崩溃后恢复 — 直接重新运行:
#
#      CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/scripts_pick_place/run_pipeline.sh
#
#    说明: 脚本会自动检测未完成的实验 (同 mode + 无 .complete 标记),
#          跳过已完成的阶段, 从断点继续。
#
# 4) 查看训练进度:
#
#      # 实时查看日志
#      tail -f data/pick_place_isaac_lab_simulation/exp{N}/pipeline.log
#
#      # 查看各轮成功率
#      python3 -c "
#      import json
#      with open('data/pick_place_isaac_lab_simulation/exp{N}/record.json') as f:
#          rec = json.load(f)
#      for it in rec['iterations']:
#          m = it['performance_metrics']
#          print(f'Iter {it[\"iteration\"]:2d}: A={m[\"task_A_success_rate\"]*100:.1f}%  B={m[\"task_B_success_rate\"]*100:.1f}%')
#      "
#
# =============================================================================
# 关键配置参数说明
# =============================================================================
#
#   MAX_ITERATIONS    迭代总轮数 (默认 10)
#   NUM_CYCLES        每轮评估的 A-B cycle 数 (默认 50, 即每个策略评估 50 个 episode)
#   STEPS_PER_ITER    每轮训练步数 (默认 5000)
#   HORIZON           每个 episode 最大步数 (默认 400)
#   BATCH_SIZE        训练 batch size (默认 32)
#   N_ACTION_STEPS    Diffusion policy 的 action chunk 长度 (默认 16)
#   DISTANCE_THRESHOLD 判定 pick/place 成功的距离阈值 (默认 0.05m)
#   POLICY_A_SRC      Policy A 的预训练源路径 (只读, 不会被修改)
#   POLICY_B_SRC      Policy B 的预训练源路径 (只读, 不会被修改)
#   TRAINING_GPUS     训练用 GPU 列表, 从 CUDA_VISIBLE_DEVICES 继承
#   DATA_COLLECTION_GPU  评估/数据收集用的单 GPU (默认 0)
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

# Timestamp prefix for log lines: [2026-03-06 22:41:05]
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

# Save a copy of the pretrained_model checkpoint for a given iteration
save_iter_checkpoint() {
    local ITER=$1       # iteration number
    local NAME=$2       # "A" or "B"
    local TEMP=$3       # temp working directory (has latest checkpoint)
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

    # Build weighted sampling flag if metadata exists
    local SAMPLE_WEIGHTS_ARG=""
    local SW_PATH="$LAST/lerobot_dataset/meta/sampling_weights.json"
    if [ -f "$SW_PATH" ]; then
        SAMPLE_WEIGHTS_ARG="--sample_weights $SW_PATH"
        echo "  Using weighted sampling (balancing old/new data 1:1)"
    fi

    # Pick a random free port to avoid EADDRINUSE on default 29500
    local MASTER_PORT
    MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
    echo "  Using master_port=$MASTER_PORT"

    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS --master_port=$MASTER_PORT \
        scripts/scripts_pick_place/4_train_diffusion.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$LAST/lerobot_dataset" \
        --out "$LAST" \
        --steps $TARGET \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose --include_gripper --wandb \
        $SAMPLE_WEIGHTS_ARG

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

# Tee all output to log file (append on resume), with timestamps
exec > >(add_timestamps | tee -a "$EXP_DIR/${MODE}.log") 2>&1

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
        save_iter_checkpoint $iter A "$PA_TEMP"
        mark_done $iter train_A
    else
        echo "  [Train A] Already done — skipping"
    fi

    # ---- Phase 3: Train Policy B ----
    if ! phase_done $iter train_B; then
        echo "--- Train Policy B ($STEPS_PER_ITER steps) ---"
        train_policy B "$PB_TEMP" "$PB_LAST" "$ROLLOUT_B"
        save_iter_checkpoint $iter B "$PB_TEMP"
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
