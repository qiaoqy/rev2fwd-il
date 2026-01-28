#!/bin/bash
# =============================================================================
# Iterative Training Script for Rev2Fwd Imitation Learning
# =============================================================================
# This script automates the test → collect data → finetune loop for
# iterative policy improvement.
#
# Workflow per iteration:
#   1. Run alternating test (6_test_alternating.py)
#      - Executes A→B→A→B... cycles until failure
#      - Collects rollout data for both tasks
#   2. Finetune Policy A (7_finetune_with_rollout.py)
#      - Merges original A data with rollout A data
#      - Continues training from checkpoint
#   3. Finetune Policy B (7_finetune_with_rollout.py)
#      - Merges original B data with rollout B data
#      - Continues training from checkpoint
#
# Usage:
#   bash scripts/scripts_pick_place/run_iterative_training.sh
#
# Configuration:
#   Edit the variables below to customize the training loop.
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Conda Environment Activation
# =============================================================================
# Initialize conda for bash script
eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
MAX_ITERATIONS=20           # Maximum number of test-finetune iterations
STEPS_PER_ITER=5000         # Training steps per finetuning iteration
MAX_CYCLES=50               # Maximum A→B cycles per alternating test
HORIZON=400                 # Maximum steps per task attempt
BATCH_SIZE=32               # Training batch size

# Thresholds for success detection
# Note: New success criteria requires object on table (z<0.05), at goal (XY distance), and gripper open
DISTANCE_THRESHOLD=0.05     # Maximum distance from target for success

# Action chunk settings
N_ACTION_STEPS=16           # Number of action steps to execute per inference

# Policy directories (MODIFY THESE FOR YOUR SETUP)
# Source checkpoints (read-only, will not be modified)
POLICY_A_DIR="runs/PP_A_circle"
POLICY_B_DIR="runs/PP_B_circle"

# Finetune output directories
POLICY_A_DIR_TEMP="runs/PP_A_circle_finetune_temp"
POLICY_B_DIR_TEMP="runs/PP_B_circle_finetune_temp"
POLICY_A_DIR_LAST="runs/PP_A_circle_finetune_last"
POLICY_B_DIR_LAST="runs/PP_B_circle_finetune_last"

# Original LeRobot datasets (MODIFY THESE FOR YOUR SETUP)
# These are the pre-converted LeRobot datasets from initial training
LEROBOT_A="${POLICY_A_DIR}/lerobot_dataset"
LEROBOT_B="${POLICY_B_DIR}/lerobot_dataset"

# Goal position (plate center)
GOAL_X=0.5
GOAL_Y=0.0

# GPU configuration
# Data collection GPU (single GPU for rollout)
DATA_COLLECTION_GPU=2

# Training GPUs (multi-GPU for finetuning)
TRAINING_GPUS="${CUDA_VISIBLE_DEVICES:-2,3,4,8,9}"  # Default to GPUs if not set
NUM_TRAINING_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)  # Count number of GPUs
echo "Data collection GPU: $DATA_COLLECTION_GPU"
echo "Training GPUs: $TRAINING_GPUS (total: $NUM_TRAINING_GPUS)"

# Additional flags
HEADLESS="--headless"                   # Run in headless mode
SAVE_VIDEO="--save_video"               # Save video for debugging
VISUALIZE_ACTION_CHUNK="--visualize_action_chunk"  # Visualize action chunks

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

check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        exit 1
    fi
}

get_checkpoint_path() {
    # Returns the path to the latest checkpoint for a policy
    local policy_dir=$1
    echo "${policy_dir}/checkpoints/checkpoints/last/pretrained_model"
}

get_current_step() {
    # Returns the current training step from checkpoint's training_state
    # Args: policy_dir (e.g., runs/PP_A_circle_finetune_test)
    local policy_dir=$1
    local training_step_file="${policy_dir}/checkpoints/checkpoints/last/training_state/training_step.json"
    
    if [ -f "$training_step_file" ]; then
        # Extract step value from JSON using grep and sed
        local step=$(grep -o '"step": *[0-9]*' "$training_step_file" | sed 's/"step": *//')
        echo "$step"
    else
        echo "0"
    fi
}

# Rollout record file
ROLLOUT_RECORD_FILE="data/rollout_record.json"

init_rollout_record() {
    # Backup old record file if it exists, then create a new one
    # This ensures each run of the script starts with a fresh record
    
    if [ -f "$ROLLOUT_RECORD_FILE" ]; then
        # Generate backup filename with timestamp
        local backup_timestamp=$(date +%Y%m%d_%H%M%S)
        local backup_file="${ROLLOUT_RECORD_FILE%.json}_backup_${backup_timestamp}.json"
        echo "Backing up existing rollout record..."
        echo "  From: $ROLLOUT_RECORD_FILE"
        echo "  To:   $backup_file"
        mv "$ROLLOUT_RECORD_FILE" "$backup_file"
        echo "✓ Old record backed up"
    fi
    
    # Create new record file
    echo "Creating new rollout record file: $ROLLOUT_RECORD_FILE"
    local timestamp=$(date -Iseconds)
    python3 -c "
import json
data = {
    'description': 'Rollout performance record for iterative training',
    'created_at': '$timestamp',
    'config': {
        'max_iterations': $MAX_ITERATIONS,
        'steps_per_iter': $STEPS_PER_ITER,
        'max_cycles': $MAX_CYCLES,
        'horizon': $HORIZON,
        'batch_size': $BATCH_SIZE,
        'distance_threshold': $DISTANCE_THRESHOLD,
        'n_action_steps': $N_ACTION_STEPS,
        'policy_A_source': '$POLICY_A_DIR',
        'policy_B_source': '$POLICY_B_DIR',
        'goal_xy': [$GOAL_X, $GOAL_Y]
    },
    'iterations': []
}
with open('$ROLLOUT_RECORD_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
    echo "✓ New rollout record initialized"
}

append_rollout_record() {
    # Append rollout statistics to the record file
    # Args: iteration, retry_count, stats_file_A, stats_file_B, 
    #       checkpoint_A_step, checkpoint_B_step, data_collected_A, data_collected_B
    local iteration=$1
    local retry_count=$2
    local stats_file=$3  # The .stats.json file from 6_test_alternating.py
    local checkpoint_A_step=$4
    local checkpoint_B_step=$5
    local data_collected_A=$6  # "true" or "false"
    local data_collected_B=$7  # "true" or "false"
    
    echo "  Recording rollout statistics to $ROLLOUT_RECORD_FILE..."
    
    python3 << PYTHON_SCRIPT
import json
from datetime import datetime
from pathlib import Path

record_file = "$ROLLOUT_RECORD_FILE"
stats_file = "$stats_file"
iteration = $iteration
retry_count = $retry_count
checkpoint_A_step = $checkpoint_A_step
checkpoint_B_step = $checkpoint_B_step
data_collected_A = "$data_collected_A" == "true"
data_collected_B = "$data_collected_B" == "true"

# Load existing record
with open(record_file, 'r') as f:
    record = json.load(f)

# Load stats from this rollout if available
rollout_stats = None
if Path(stats_file).exists():
    with open(stats_file, 'r') as f:
        rollout_stats = json.load(f)

# Build iteration entry
iteration_entry = {
    "iteration": iteration,
    "timestamp": datetime.now().isoformat(),
    "retry_attempt": retry_count,
    "checkpoint_info": {
        "policy_A_training_step": checkpoint_A_step,
        "policy_B_training_step": checkpoint_B_step,
    },
    "data_collection": {
        "task_A_data_collected": data_collected_A,
        "task_B_data_collected": data_collected_B,
    },
    "rollout_results": None,
}

if rollout_stats:
    iteration_entry["rollout_results"] = {
        "summary": rollout_stats.get("summary", {}),
        "config_used": rollout_stats.get("config", {}),
        "episodes_A_details": rollout_stats.get("episodes_A", []),
        "episodes_B_details": rollout_stats.get("episodes_B", []),
    }
    
    # Add computed metrics
    summary = rollout_stats.get("summary", {})
    iteration_entry["performance_metrics"] = {
        "consecutive_successes": summary.get("consecutive_successes", 0),
        "task_A_success_rate": summary.get("task_A_success_rate", 0),
        "task_B_success_rate": summary.get("task_B_success_rate", 0),
        "task_A_success_count": summary.get("task_A_success_count", 0),
        "task_B_success_count": summary.get("task_B_success_count", 0),
        "total_task_A_episodes": summary.get("total_task_A_episodes", 0),
        "total_task_B_episodes": summary.get("total_task_B_episodes", 0),
        "total_elapsed_seconds": summary.get("total_elapsed_seconds", 0),
    }
    
    # Calculate average success steps
    episodes_A = rollout_stats.get("episodes_A", [])
    episodes_B = rollout_stats.get("episodes_B", [])
    
    success_steps_A = [ep.get("success_step") for ep in episodes_A if ep.get("success") and ep.get("success_step")]
    success_steps_B = [ep.get("success_step") for ep in episodes_B if ep.get("success") and ep.get("success_step")]
    
    iteration_entry["performance_metrics"]["avg_success_step_A"] = sum(success_steps_A) / len(success_steps_A) if success_steps_A else None
    iteration_entry["performance_metrics"]["avg_success_step_B"] = sum(success_steps_B) / len(success_steps_B) if success_steps_B else None
    iteration_entry["performance_metrics"]["min_success_step_A"] = min(success_steps_A) if success_steps_A else None
    iteration_entry["performance_metrics"]["min_success_step_B"] = min(success_steps_B) if success_steps_B else None
    iteration_entry["performance_metrics"]["max_success_step_A"] = max(success_steps_A) if success_steps_A else None
    iteration_entry["performance_metrics"]["max_success_step_B"] = max(success_steps_B) if success_steps_B else None
else:
    iteration_entry["rollout_results"] = {
        "error": "No stats file found or rollout failed completely"
    }
    iteration_entry["performance_metrics"] = {
        "consecutive_successes": 0,
        "task_A_success_rate": 0,
        "task_B_success_rate": 0,
        "task_A_success_count": 0,
        "task_B_success_count": 0,
        "total_task_A_episodes": 0,
        "total_task_B_episodes": 0,
    }

# Append to iterations list
record["iterations"].append(iteration_entry)

# Update record with latest summary
record["last_updated"] = datetime.now().isoformat()
record["total_iterations_completed"] = len(record["iterations"])

# Save updated record
with open(record_file, 'w') as f:
    json.dump(record, f, indent=2)

print(f"  ✓ Recorded iteration {iteration} (retry {retry_count})")
if rollout_stats:
    summary = rollout_stats.get("summary", {})
    print(f"    Consecutive successes: {summary.get('consecutive_successes', 0)}")
    print(f"    Task A: {summary.get('task_A_success_count', 0)}/{summary.get('total_task_A_episodes', 0)} success")
    print(f"    Task B: {summary.get('task_B_success_count', 0)}/{summary.get('total_task_B_episodes', 0)} success")
PYTHON_SCRIPT
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks"

# Check that original LeRobot datasets exist
if [ ! -d "$LEROBOT_A" ]; then
    echo "ERROR: LeRobot dataset A not found: $LEROBOT_A"
    echo "Please run initial training first (see PIPELINE_README.md Step 4)"
    exit 1
fi

if [ ! -d "$LEROBOT_B" ]; then
    echo "ERROR: LeRobot dataset B not found: $LEROBOT_B"
    echo "Please run initial training first (see PIPELINE_README.md Step 4)"
    exit 1
fi

# Check that initial policies exist
CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR")
CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR")

if [ ! -d "$CHECKPOINT_A" ]; then
    echo "ERROR: Policy A checkpoint not found: $CHECKPOINT_A"
    echo "Please run initial training first (see PIPELINE_README.md Step 4)"
    exit 1
fi

if [ ! -d "$CHECKPOINT_B" ]; then
    echo "ERROR: Policy B checkpoint not found: $CHECKPOINT_B"
    echo "Please run initial training first (see PIPELINE_README.md Step 4)"
    exit 1
fi

echo "✓ LeRobot datasets found"
echo "✓ Initial policy checkpoints found"

# =============================================================================
# Configuration Summary
# =============================================================================
print_header "Configuration Summary"
echo "  MAX_ITERATIONS:     $MAX_ITERATIONS"
echo "  STEPS_PER_ITER:     $STEPS_PER_ITER"
echo "  MAX_CYCLES:         $MAX_CYCLES"
echo "  HORIZON:            $HORIZON"
echo "  BATCH_SIZE:         $BATCH_SIZE"
echo "  DISTANCE_THRESHOLD: $DISTANCE_THRESHOLD"
echo "  N_ACTION_STEPS:     $N_ACTION_STEPS"
echo ""
echo "  POLICY_A_DIR:       $POLICY_A_DIR (source)"
echo "  POLICY_B_DIR:       $POLICY_B_DIR (source)"
echo "  POLICY_A_DIR_TEMP:  $POLICY_A_DIR_TEMP (previous iteration)"
echo "  POLICY_A_DIR_LAST:  $POLICY_A_DIR_LAST (latest finetune)"
echo "  POLICY_B_DIR_TEMP:  $POLICY_B_DIR_TEMP (previous iteration)"
echo "  POLICY_B_DIR_LAST:  $POLICY_B_DIR_LAST (latest finetune)"
echo "  LEROBOT_A:          $LEROBOT_A"
echo "  LEROBOT_B:          $LEROBOT_B"
echo "  GOAL:               ($GOAL_X, $GOAL_Y)"

# =============================================================================
# Initialize Rollout Record
# =============================================================================
print_header "Initializing Rollout Record"
init_rollout_record

# =============================================================================
# Backup Old Finetune Directories
# =============================================================================
print_header "Backing Up Old Finetune Directories"

# Generate a single timestamp for all backups in this run
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup old temp and last directories if they exist
for dir in "$POLICY_A_DIR_TEMP" "$POLICY_A_DIR_LAST" "$POLICY_B_DIR_TEMP" "$POLICY_B_DIR_LAST"; do
    if [ -d "$dir" ]; then
        backup_dir="${dir}_backup_${BACKUP_TIMESTAMP}"
        echo "Backing up: $dir"
        echo "       To: $backup_dir"
        mv "$dir" "$backup_dir"
        echo "  ✓ Done"
    fi
done

echo "✓ Old directories backed up (if any existed)"

# =============================================================================
# Initialize Directories (copy from origin to temp)
# =============================================================================
print_header "Initializing Fresh Directories"

# 数据流设计:
# - origin (POLICY_A_DIR/POLICY_B_DIR): 原始数据，只读
# - temp: 上一轮的结果，用于累积新数据
# - last: 当前训练目录
#
# 每轮迭代:
# 1. 在 temp 中累积新数据 → 保存到 last
# 2. temp 的 checkpoint → last
# 3. 在 last 训练
# 4. 删除 temp，last 重命名为 temp

# 由于旧目录已备份，这里直接从 origin 复制到 temp
echo "Initializing Policy A temp directory from origin..."
echo "  From: $POLICY_A_DIR"
echo "  To:   $POLICY_A_DIR_TEMP"
mkdir -p "$POLICY_A_DIR_TEMP"
cp -r "$POLICY_A_DIR/checkpoints" "$POLICY_A_DIR_TEMP/checkpoints"
cp -r "$LEROBOT_A" "$POLICY_A_DIR_TEMP/lerobot_dataset"
echo "  ✓ Policy A temp initialized (checkpoints + lerobot_dataset)"

echo "Initializing Policy B temp directory from origin..."
echo "  From: $POLICY_B_DIR"
echo "  To:   $POLICY_B_DIR_TEMP"
mkdir -p "$POLICY_B_DIR_TEMP"
cp -r "$POLICY_B_DIR/checkpoints" "$POLICY_B_DIR_TEMP/checkpoints"
cp -r "$LEROBOT_B" "$POLICY_B_DIR_TEMP/lerobot_dataset"
echo "  ✓ Policy B temp initialized (checkpoints + lerobot_dataset)"

# 注意：last 目录会在每轮迭代中创建，不需要预先初始化

# =============================================================================
# Main Loop
# =============================================================================

for iter in $(seq 1 $MAX_ITERATIONS); do
    print_header "Iteration $iter / $MAX_ITERATIONS"

    # Define output paths for this iteration
    ROLLOUT_A="data/rollout_A_circle_iter${iter}.npz"
    ROLLOUT_B="data/rollout_B_circle_iter${iter}.npz"

    # 每次循环都用 temp 目录的权重（上一轮训练的结果）
    CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")
    
    # =========================================================================
    # Step 1: Alternating Test (with retry until at least one task collects data)
    # =========================================================================
    print_section "[Step 1] Running alternating test..."
    
    echo "  Checkpoint A: $CHECKPOINT_A"
    echo "  Checkpoint B: $CHECKPOINT_B"
    echo "  Output A: $ROLLOUT_A"
    echo "  Output B: $ROLLOUT_B"
    
    # Get current training steps for record
    CURRENT_STEP_A_FOR_RECORD=$(get_current_step "$POLICY_A_DIR_TEMP")
    CURRENT_STEP_B_FOR_RECORD=$(get_current_step "$POLICY_B_DIR_TEMP")
    
    MAX_RETRIES=10  # Maximum number of retries for data collection
    retry_count=0
    final_retry_count=0  # Track which retry succeeded
    
    while true; do
        retry_count=$((retry_count + 1))
        echo ""
        echo "  [Attempt $retry_count/$MAX_RETRIES] Running alternating test..."
        
        # Remove any existing partial data files before retry
        rm -f "$ROLLOUT_A" "$ROLLOUT_B"
        # Also remove stats file
        STATS_FILE="${ROLLOUT_A%.npz}.stats.json"
        rm -f "$STATS_FILE"
        
        CUDA_VISIBLE_DEVICES=$DATA_COLLECTION_GPU python scripts/scripts_pick_place/6_test_alternating.py \
            --policy_A "$CHECKPOINT_A" \
            --policy_B "$CHECKPOINT_B" \
            --out_A "$ROLLOUT_A" \
            --out_B "$ROLLOUT_B" \
            --max_cycles $MAX_CYCLES \
            --horizon $HORIZON \
            --distance_threshold $DISTANCE_THRESHOLD \
            --n_action_steps $N_ACTION_STEPS \
            --goal_xy $GOAL_X $GOAL_Y \
            $SAVE_VIDEO $VISUALIZE_ACTION_CHUNK \
            $HEADLESS
        
        final_retry_count=$retry_count
        
        # Check if AT LEAST ONE rollout data file was collected
        DATA_COLLECTED_A="false"
        DATA_COLLECTED_B="false"
        if [ -f "$ROLLOUT_A" ]; then
            DATA_COLLECTED_A="true"
        fi
        if [ -f "$ROLLOUT_B" ]; then
            DATA_COLLECTED_B="true"
        fi
        
        if [ -f "$ROLLOUT_A" ] || [ -f "$ROLLOUT_B" ]; then
            if [ -f "$ROLLOUT_A" ] && [ -f "$ROLLOUT_B" ]; then
                echo "  ✓ Both Task A and Task B rollout data collected!"
            elif [ -f "$ROLLOUT_A" ]; then
                echo "  ✓ Task A rollout data collected (Task B missing)"
            else
                echo "  ✓ Task B rollout data collected (Task A missing)"
            fi
            break
        else
            echo "  ✗ No rollout data collected."
        fi
        
        if [ $retry_count -ge $MAX_RETRIES ]; then
            echo "  WARNING: Failed to collect any rollout data after $MAX_RETRIES attempts."
            echo "  Will continue with finetuning using original data only."
            break
        fi
        
        echo "  Waiting 2 seconds before retry..."
        sleep 2
    done
    
    # Record rollout statistics
    STATS_FILE="${ROLLOUT_A%.npz}.stats.json"
    append_rollout_record $iter $final_retry_count "$STATS_FILE" \
        $CURRENT_STEP_A_FOR_RECORD $CURRENT_STEP_B_FOR_RECORD \
        "$DATA_COLLECTED_A" "$DATA_COLLECTED_B"
    
    # =========================================================================
    # Step 2: Finetune Policy A (always finetune, with or without rollout data)
    # =========================================================================
    print_section "[Step 2] Finetuning Policy A ($STEPS_PER_ITER steps)..."
    
    echo "  Source (temp): $POLICY_A_DIR_TEMP"
    echo "  Target (last): $POLICY_A_DIR_LAST"
    if [ -f "$ROLLOUT_A" ]; then
        echo "  Rollout data: $ROLLOUT_A"
        ROLLOUT_A_ARG="--rollout_data $ROLLOUT_A"
    else
        echo "  Rollout data: (none - using original data only)"
        ROLLOUT_A_ARG=""
    fi
    echo "  Checkpoint: $CHECKPOINT_A"
    echo "  Steps: $STEPS_PER_ITER"
    
    # Step 2a: 创建新的 last 目录，准备数据
    # 数据流: temp/lerobot_dataset + rollout_data -> last/lerobot_dataset
    echo ""
    echo "  [Step 2a] Preparing last directory and accumulating data..."
    rm -rf "$POLICY_A_DIR_LAST"
    mkdir -p "$POLICY_A_DIR_LAST"
    
    # 准备数据：从 temp 读取，累积新数据，写入 last
    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_lerobot "$POLICY_A_DIR_TEMP/lerobot_dataset" \
        $ROLLOUT_A_ARG \
        --checkpoint "$CHECKPOINT_A" \
        --out "$POLICY_A_DIR_LAST" \
        --steps $STEPS_PER_ITER \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --prepare_only
    
    # Step 2b: 复制 wandb 目录并更新 metadata
    # 注意：7_finetune_with_rollout.py 已经复制了 checkpoints/checkpoints/last 下的内容
    # 这里只需要复制 wandb 目录，并更新其中的路径信息
    echo ""
    echo "  [Step 2b] Copying wandb directory and updating metadata..."
    mkdir -p "$POLICY_A_DIR_LAST/checkpoints"
    cp -r "$POLICY_A_DIR_TEMP/checkpoints/wandb" "$POLICY_A_DIR_LAST/checkpoints/wandb"
    
    # 更新 wandb metadata 中的路径信息
    WANDB_LATEST_DIR="$POLICY_A_DIR_LAST/checkpoints/wandb/latest-run"
    if [ -d "$WANDB_LATEST_DIR" ]; then
        for metadata_file in "$WANDB_LATEST_DIR"/files/wandb-metadata.json; do
            if [ -f "$metadata_file" ]; then
                # 更新 args 中的 config_path 和 root 字段
                sed -i "s|$POLICY_A_DIR_TEMP|$POLICY_A_DIR_LAST|g" "$metadata_file"
                echo "  ✓ Updated wandb metadata: $metadata_file"
            fi
        done
    fi
    echo "  ✓ Wandb directory copied and metadata updated"
    
    # Step 2b2: 将 last 目录转换为符号链接格式
    # LeRobot 期望 checkpoints/checkpoints/last 是符号链接指向步数目录（如 010000）
    # 但我们复制的是真实目录，需要转换
    echo ""
    echo "  [Step 2b2] Converting last directory to symlink format..."
    CKPT_DIR="$POLICY_A_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        # last 是真实目录，需要转换为符号链接
        # 使用当前步数作为目录名（补零到6位）
        CURRENT_STEP_A=$(get_current_step "$POLICY_A_DIR_LAST")
        STEP_DIR_NAME=$(printf "%06d" $CURRENT_STEP_A)
        echo "    Current step: $CURRENT_STEP_A -> directory name: $STEP_DIR_NAME"
        
        # 重命名 last -> 步数目录
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_DIR_NAME"
        
        # 创建符号链接 last -> 步数目录
        cd "$CKPT_DIR"
        ln -s "$STEP_DIR_NAME" last
        cd - > /dev/null
        
        echo "  ✓ Converted: last -> $STEP_DIR_NAME (symlink)"
    else
        echo "  ✓ last is already a symlink or doesn't exist"
    fi
    
    # Step 2c: Train with multi-GPU in last directory
    CURRENT_STEP_A=$(get_current_step "$POLICY_A_DIR_LAST")
    TARGET_STEPS_A=$((CURRENT_STEP_A + STEPS_PER_ITER))
    
    echo "  Resuming from step $CURRENT_STEP_A → target $TARGET_STEPS_A (resume wandb run)"
    
    echo ""
    echo "  [Step 2c] Training with $NUM_TRAINING_GPUS GPUs..."
    echo "    Target steps: $TARGET_STEPS_A"
    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS \
        scripts/scripts_pick_place/4_train_diffusion.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$POLICY_A_DIR_LAST/lerobot_dataset" \
        --out "$POLICY_A_DIR_LAST" \
        --steps $TARGET_STEPS_A \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose \
        --include_gripper \
        --wandb
    
    echo "✓ Policy A finetuning complete (saved to $POLICY_A_DIR_LAST)"
    
    # Step 2d: 删除 temp，last 重命名为 temp（为下一轮准备）
    echo ""
    echo "  [Step 2d] Rotating directories: last -> temp..."
    rm -rf "$POLICY_A_DIR_TEMP"
    mv "$POLICY_A_DIR_LAST" "$POLICY_A_DIR_TEMP"
    echo "  ✓ Policy A directories rotated (last is now temp)"
    
    # =========================================================================
    # Step 3: Finetune Policy B (always finetune, with or without rollout data)
    # =========================================================================
    print_section "[Step 3] Finetuning Policy B ($STEPS_PER_ITER steps)..."
    
    echo "  Source (temp): $POLICY_B_DIR_TEMP"
    echo "  Target (last): $POLICY_B_DIR_LAST"
    if [ -f "$ROLLOUT_B" ]; then
        echo "  Rollout data: $ROLLOUT_B"
        ROLLOUT_B_ARG="--rollout_data $ROLLOUT_B"
    else
        echo "  Rollout data: (none - using original data only)"
        ROLLOUT_B_ARG=""
    fi
    echo "  Checkpoint: $CHECKPOINT_B"
    echo "  Steps: $STEPS_PER_ITER"
    
    # Step 3a: 创建新的 last 目录，准备数据
    # 数据流: temp/lerobot_dataset + rollout_data -> last/lerobot_dataset
    echo ""
    echo "  [Step 3a] Preparing last directory and accumulating data..."
    rm -rf "$POLICY_B_DIR_LAST"
    mkdir -p "$POLICY_B_DIR_LAST"
    
    # 准备数据：从 temp 读取，累积新数据，写入 last
    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_lerobot "$POLICY_B_DIR_TEMP/lerobot_dataset" \
        $ROLLOUT_B_ARG \
        --checkpoint "$CHECKPOINT_B" \
        --out "$POLICY_B_DIR_LAST" \
        --steps $STEPS_PER_ITER \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --prepare_only
    
    # Step 3b: 复制 wandb 目录并更新 metadata
    # 注意：7_finetune_with_rollout.py 已经复制了 checkpoints/checkpoints/last 下的内容
    # 这里只需要复制 wandb 目录，并更新其中的路径信息
    echo ""
    echo "  [Step 3b] Copying wandb directory and updating metadata..."
    mkdir -p "$POLICY_B_DIR_LAST/checkpoints"
    cp -r "$POLICY_B_DIR_TEMP/checkpoints/wandb" "$POLICY_B_DIR_LAST/checkpoints/wandb"
    
    # 更新 wandb metadata 中的路径信息
    WANDB_LATEST_DIR="$POLICY_B_DIR_LAST/checkpoints/wandb/latest-run"
    if [ -d "$WANDB_LATEST_DIR" ]; then
        for metadata_file in "$WANDB_LATEST_DIR"/files/wandb-metadata.json; do
            if [ -f "$metadata_file" ]; then
                # 更新 args 中的 config_path 和 root 字段
                sed -i "s|$POLICY_B_DIR_TEMP|$POLICY_B_DIR_LAST|g" "$metadata_file"
                echo "  ✓ Updated wandb metadata: $metadata_file"
            fi
        done
    fi
    echo "  ✓ Wandb directory copied and metadata updated"
    
    # Step 3b2: 将 last 目录转换为符号链接格式
    # LeRobot 期望 checkpoints/checkpoints/last 是符号链接指向步数目录（如 010000）
    # 但我们复制的是真实目录，需要转换
    echo ""
    echo "  [Step 3b2] Converting last directory to symlink format..."
    CKPT_DIR="$POLICY_B_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        # last 是真实目录，需要转换为符号链接
        # 使用当前步数作为目录名（补零到6位）
        CURRENT_STEP_B=$(get_current_step "$POLICY_B_DIR_LAST")
        STEP_DIR_NAME=$(printf "%06d" $CURRENT_STEP_B)
        echo "    Current step: $CURRENT_STEP_B -> directory name: $STEP_DIR_NAME"
        
        # 重命名 last -> 步数目录
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_DIR_NAME"
        
        # 创建符号链接 last -> 步数目录
        cd "$CKPT_DIR"
        ln -s "$STEP_DIR_NAME" last
        cd - > /dev/null
        
        echo "  ✓ Converted: last -> $STEP_DIR_NAME (symlink)"
    else
        echo "  ✓ last is already a symlink or doesn't exist"
    fi

    # Step 3c: Train with multi-GPU in last directory
    CURRENT_STEP_B=$(get_current_step "$POLICY_B_DIR_LAST")
    TARGET_STEPS_B=$((CURRENT_STEP_B + STEPS_PER_ITER))
    
    echo "  Resuming from step $CURRENT_STEP_B → target $TARGET_STEPS_B (resume wandb run)"
    
    echo ""
    echo "  [Step 3c] Training with $NUM_TRAINING_GPUS GPUs..."
    echo "    Target steps: $TARGET_STEPS_B"
    CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node=$NUM_TRAINING_GPUS \
        scripts/scripts_pick_place/4_train_diffusion.py \
        --dataset dummy.npz \
        --lerobot_dataset_dir "$POLICY_B_DIR_LAST/lerobot_dataset" \
        --out "$POLICY_B_DIR_LAST" \
        --steps $TARGET_STEPS_B \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        --save_freq $STEPS_PER_ITER \
        --skip_convert --resume \
        --include_obj_pose \
        --include_gripper \
        --wandb
    
    echo "✓ Policy B finetuning complete (saved to $POLICY_B_DIR_LAST)"
    
    # Step 3d: 删除 temp，last 重命名为 temp（为下一轮准备）
    echo ""
    echo "  [Step 3d] Rotating directories: last -> temp..."
    rm -rf "$POLICY_B_DIR_TEMP"
    mv "$POLICY_B_DIR_LAST" "$POLICY_B_DIR_TEMP"
    echo "  ✓ Policy B directories rotated (last is now temp)"
    
    print_section "Iteration $iter complete!"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
print_header "Iterative Training Finished!"
echo ""
echo "Rollout record file (detailed statistics):"
echo "  $ROLLOUT_RECORD_FILE"
if [ -f "$ROLLOUT_RECORD_FILE" ]; then
    # Print summary from rollout record
    python3 << 'PYTHON_SUMMARY'
import json
with open("data/rollout_record.json", 'r') as f:
    record = json.load(f)
iterations = record.get("iterations", [])
print(f"  Total iterations recorded: {len(iterations)}")
if iterations:
    # Calculate overall statistics
    total_consecutive = sum(it.get("performance_metrics", {}).get("consecutive_successes", 0) for it in iterations)
    avg_consecutive = total_consecutive / len(iterations)
    max_consecutive = max(it.get("performance_metrics", {}).get("consecutive_successes", 0) for it in iterations)
    print(f"  Average consecutive successes per iteration: {avg_consecutive:.1f}")
    print(f"  Best consecutive successes: {max_consecutive}")
    
    # Success rate trend
    a_rates = [it.get("performance_metrics", {}).get("task_A_success_rate", 0) for it in iterations]
    b_rates = [it.get("performance_metrics", {}).get("task_B_success_rate", 0) for it in iterations]
    if a_rates:
        print(f"  Task A success rate: {a_rates[0]*100:.1f}% (iter 1) -> {a_rates[-1]*100:.1f}% (iter {len(a_rates)})")
    if b_rates:
        print(f"  Task B success rate: {b_rates[0]*100:.1f}% (iter 1) -> {b_rates[-1]*100:.1f}% (iter {len(b_rates)})")
PYTHON_SUMMARY
fi
echo ""
echo "Rollout data saved in data/ directory:"
ls -la data/rollout_*_iter*.npz 2>/dev/null || echo "  (no rollout data files found)"
echo ""
echo "Rollout statistics files:"
ls -la data/rollout_*_iter*.stats.json 2>/dev/null || echo "  (no stats files found)"
echo ""
echo "Finetuned checkpoints (in temp directories after rotation):"
ls -la $POLICY_A_DIR_TEMP/checkpoints/checkpoints/last/pretrained_model 2>/dev/null || echo "  Policy A: (not created)"
ls -la $POLICY_B_DIR_TEMP/checkpoints/checkpoints/last/pretrained_model 2>/dev/null || echo "  Policy B: (not created)"
echo ""
echo "Original checkpoints preserved at:"
echo "  $POLICY_A_DIR"
echo "  $POLICY_B_DIR"
echo ""
echo "To view rollout performance history:"
echo "  cat $ROLLOUT_RECORD_FILE | python -m json.tool"
echo ""
echo "To evaluate the final policies, run:"
echo "  python scripts/scripts_pick_place/5_eval_diffusion.py \\"
echo "      --checkpoint $(get_checkpoint_path "$POLICY_A_DIR_TEMP") \\"
echo "      --out_dir ${POLICY_A_DIR_TEMP}/videos_final \\"
echo "      --num_episodes 10 --headless"
echo ""
