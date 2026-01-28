#!/bin/bash
# =============================================================================
# TEST VERSION: Iterative Training Script for Rev2Fwd Imitation Learning
# =============================================================================
# This is a test version with reduced parameters for quick validation.
# After testing passes, use run_iterative_training.sh for full training.
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
# Configuration (REDUCED FOR TESTING)
# =============================================================================
MAX_ITERATIONS=1            # Set to 1 for debugging (Step 1 is skipped, only iter1 data exists)
STEPS_PER_ITER=100          # Reduced for debugging (was 5000)
MAX_CYCLES=10               # 10 A→B cycles for data collection
HORIZON=400                 # Match default horizon (increased from 300)
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

# Test output directories (finetuned checkpoints will be saved here)

# 微调临时和最终输出目录
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

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks (TEST MODE)"

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
print_header "Configuration Summary (TEST MODE)"
echo "  MAX_ITERATIONS:     $MAX_ITERATIONS (REDUCED)"
echo "  STEPS_PER_ITER:     $STEPS_PER_ITER (REDUCED)"
echo "  MAX_CYCLES:         $MAX_CYCLES (REDUCED)"
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
# Initialize Test Directories (copy from origin to temp)
# =============================================================================
print_header "Initializing Test Directories"

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

# 初始化：只有第一次运行时，从 origin 复制到 temp
# 需要同时检查 checkpoints 和 lerobot_dataset 都存在
if [ ! -d "$POLICY_A_DIR_TEMP/checkpoints" ] || [ ! -d "$POLICY_A_DIR_TEMP/lerobot_dataset" ]; then
    echo "Initializing Policy A temp directory from origin..."
    echo "  From: $POLICY_A_DIR"
    echo "  To:   $POLICY_A_DIR_TEMP"
    # 清理可能存在的部分目录
    rm -rf "$POLICY_A_DIR_TEMP"
    mkdir -p "$POLICY_A_DIR_TEMP"
    cp -r "$POLICY_A_DIR/checkpoints" "$POLICY_A_DIR_TEMP/checkpoints"
    cp -r "$LEROBOT_A" "$POLICY_A_DIR_TEMP/lerobot_dataset"
    echo "  ✓ Policy A temp initialized (checkpoints + lerobot_dataset)"
else
    echo "✓ Policy A temp already exists, skipping initialization"
fi

if [ ! -d "$POLICY_B_DIR_TEMP/checkpoints" ] || [ ! -d "$POLICY_B_DIR_TEMP/lerobot_dataset" ]; then
    echo "Initializing Policy B temp directory from origin..."
    echo "  From: $POLICY_B_DIR"
    echo "  To:   $POLICY_B_DIR_TEMP"
    # 清理可能存在的部分目录
    rm -rf "$POLICY_B_DIR_TEMP"
    mkdir -p "$POLICY_B_DIR_TEMP"
    cp -r "$POLICY_B_DIR/checkpoints" "$POLICY_B_DIR_TEMP/checkpoints"
    cp -r "$LEROBOT_B" "$POLICY_B_DIR_TEMP/lerobot_dataset"
    echo "  ✓ Policy B temp initialized (checkpoints + lerobot_dataset)"
else
    echo "✓ Policy B temp already exists, skipping initialization"
fi

# 注意：last 目录会在每轮迭代中创建，不需要预先初始化

# =============================================================================
# Main Loop
# =============================================================================

for iter in $(seq 1 $MAX_ITERATIONS); do
    print_header "Iteration $iter / $MAX_ITERATIONS (TEST MODE)"

    # Define output paths for this iteration (use _test suffix to avoid overwriting real data)
    ROLLOUT_A="data/rollout_A_circle_iter${iter}_test.npz"
    ROLLOUT_B="data/rollout_B_circle_iter${iter}_test.npz"

    # 每次循环都用 temp 目录的权重（上一轮训练的结果）
    CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")
    
    # =========================================================================
    # Step 1: Alternating Test (with retry until at least one task collects data)
    # =========================================================================
    print_section "[Step 1] Running alternating test (TEST MODE)..."
    
    echo "  Checkpoint A: $CHECKPOINT_A"
    echo "  Checkpoint B: $CHECKPOINT_B"
    echo "  Output A: $ROLLOUT_A"
    echo "  Output B: $ROLLOUT_B"
    
    MAX_RETRIES=3  # Maximum number of retries for data collection (reduced from 10)
    retry_count=0
    
    while true; do
        retry_count=$((retry_count + 1))
        echo ""
        echo "  [Attempt $retry_count/$MAX_RETRIES] Running alternating test..."
        
        # Remove any existing partial data files before retry
        rm -f "$ROLLOUT_A" "$ROLLOUT_B"
        
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
        
        # Check if AT LEAST ONE rollout data file was collected
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
    
    # =========================================================================
    # Step 2: Finetune Policy A (always finetune, with or without rollout data)
    # =========================================================================
    print_section "[Step 2] Finetuning Policy A (TEST MODE - $STEPS_PER_ITER steps)..."
    
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
    print_section "[Step 3] Finetuning Policy B (TEST MODE - $STEPS_PER_ITER steps)..."
    
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
print_header "Test Run Finished!"
echo ""
echo "Test rollout data:"
ls -la data/rollout_*_test.npz 2>/dev/null || echo "  (no test rollout data files found)"
echo ""
echo "Finetuned test checkpoints (in temp directories after rotation):"
ls -la $POLICY_A_DIR_TEMP/checkpoints/checkpoints/last/pretrained_model 2>/dev/null || echo "  Policy A: (not created)"
ls -la $POLICY_B_DIR_TEMP/checkpoints/checkpoints/last/pretrained_model 2>/dev/null || echo "  Policy B: (not created)"
echo ""
echo "Original checkpoints preserved at:"
echo "  $POLICY_A_DIR"
echo "  $POLICY_B_DIR"
echo ""
echo "To run full iterative training, use:"
echo "  bash scripts/scripts_pick_place/run_iterative_training.sh"
echo ""
