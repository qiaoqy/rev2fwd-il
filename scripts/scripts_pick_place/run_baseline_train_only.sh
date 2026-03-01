#!/bin/bash
# =============================================================================
# Baseline Training ONLY — runs continuously on GPUs 1,5,6
# =============================================================================
#
# Trains Policy A and B for 5000 steps each iteration using ORIGINAL data only.
# After each iteration, saves checkpoints to a staging area and signals readiness
# so the eval script (running in parallel) can pick them up.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1,5,6 bash scripts/scripts_pick_place/run_baseline_train_only.sh 2>&1 | tee data/baseline_train.log
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
TOTAL_TRAIN_ITERS=9         # Training iterations 1-9 produce checkpoints for eval 2-10
STEPS_PER_ITER=5000
BATCH_SIZE=32
N_ACTION_STEPS=16

# Source policies
POLICY_A_DIR="runs/PP_A_circle"
POLICY_B_DIR="runs/PP_B_circle"

# Working directories
POLICY_A_DIR_TEMP="runs/PP_A_baseline_temp"
POLICY_B_DIR_TEMP="runs/PP_B_baseline_temp"
POLICY_A_DIR_LAST="runs/PP_A_baseline_last"
POLICY_B_DIR_LAST="runs/PP_B_baseline_last"

# Staging directory for eval script to read
STAGING_DIR="data/baseline_checkpoints"

# GPU config — 3 GPUs for training
TRAINING_GPUS="${CUDA_VISIBLE_DEVICES:-1,5,6}"
NUM_TRAINING_GPUS=$(echo "$TRAINING_GPUS" | tr ',' '\n' | wc -l)

echo "Training GPUs: $TRAINING_GPUS (total: $NUM_TRAINING_GPUS)"

# NCCL robustness
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

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

# =============================================================================
# Cleanup interrupted state
# =============================================================================
print_header "Cleaning up from interrupted run"

# Remove partial A_last dir from interrupted training
if [ -d "$POLICY_A_DIR_LAST" ]; then
    echo "  Removing partial $POLICY_A_DIR_LAST"
    rm -rf "$POLICY_A_DIR_LAST"
fi
if [ -d "$POLICY_B_DIR_LAST" ]; then
    echo "  Removing partial $POLICY_B_DIR_LAST"
    rm -rf "$POLICY_B_DIR_LAST"
fi

# Verify temp dirs are at step 10000 (iteration 1 starting point)
STEP_A=$(get_current_step "$POLICY_A_DIR_TEMP")
STEP_B=$(get_current_step "$POLICY_B_DIR_TEMP")
echo "  Current steps: A=$STEP_A, B=$STEP_B"

# Create staging directory
mkdir -p "$STAGING_DIR"

echo "✓ Cleanup done"

# =============================================================================
# Main Training Loop: iterations 1-9
# =============================================================================
# Training iteration K produces checkpoints for eval iteration K+1
# After training iter K:
#   - A is at step 10000 + K*5000
#   - B is at step 10000 + K*5000
#   - Checkpoints staged at data/baseline_checkpoints/iter_{K+1}/

for train_iter in $(seq 1 $TOTAL_TRAIN_ITERS); do
    EVAL_ITER=$((train_iter + 1))
    print_header "Training Iteration $train_iter (for Eval Iter $EVAL_ITER)"

    CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
    STEP_A=$(get_current_step "$POLICY_A_DIR_TEMP")

    # =====================================================================
    # Train Policy A
    # =====================================================================
    print_section "Training Policy A ($STEPS_PER_ITER steps, original data only)"

    rm -rf "$POLICY_A_DIR_LAST"
    mkdir -p "$POLICY_A_DIR_LAST"

    # Prepare: copy original dataset + checkpoint
    echo "  Preparing dataset and checkpoint..."
    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_lerobot "$POLICY_A_DIR_TEMP/lerobot_dataset" \
        --checkpoint "$CHECKPOINT_A" \
        --out "$POLICY_A_DIR_LAST" \
        --prepare_only \
        --include_obj_pose \
        --include_gripper

    # Copy wandb
    if [ -d "$POLICY_A_DIR_TEMP/checkpoints/wandb" ]; then
        cp -r "$POLICY_A_DIR_TEMP/checkpoints/wandb" "$POLICY_A_DIR_LAST/checkpoints/wandb"
        WANDB_LATEST="$POLICY_A_DIR_LAST/checkpoints/wandb/latest-run"
        if [ -d "$WANDB_LATEST" ]; then
            for mf in "$WANDB_LATEST"/files/wandb-metadata.json; do
                [ -f "$mf" ] && sed -i "s|$POLICY_A_DIR_TEMP|$POLICY_A_DIR_LAST|g" "$mf"
            done
        fi
    fi

    # Convert last → symlink
    CKPT_DIR="$POLICY_A_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        CUR_STEP=$(get_current_step "$POLICY_A_DIR_LAST")
        STEP_NAME=$(printf "%06d" $CUR_STEP)
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  ✓ Converted: last → $STEP_NAME (symlink)"
    fi

    # Train
    CUR_STEP_A=$(get_current_step "$POLICY_A_DIR_LAST")
    TARGET_A=$((CUR_STEP_A + STEPS_PER_ITER))
    echo "  Training: step $CUR_STEP_A → $TARGET_A ($NUM_TRAINING_GPUS GPUs)"

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

    echo "  ✓ Policy A trained to step $TARGET_A"

    # Rotate: last → temp
    rm -rf "$POLICY_A_DIR_TEMP"
    mv "$POLICY_A_DIR_LAST" "$POLICY_A_DIR_TEMP"
    echo "  ✓ Rotated A: last → temp"

    # =====================================================================
    # Train Policy B
    # =====================================================================
    print_section "Training Policy B ($STEPS_PER_ITER steps, original data only)"

    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")

    rm -rf "$POLICY_B_DIR_LAST"
    mkdir -p "$POLICY_B_DIR_LAST"

    # Prepare: copy original dataset + checkpoint
    echo "  Preparing dataset and checkpoint..."
    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_lerobot "$POLICY_B_DIR_TEMP/lerobot_dataset" \
        --checkpoint "$CHECKPOINT_B" \
        --out "$POLICY_B_DIR_LAST" \
        --prepare_only \
        --include_obj_pose \
        --include_gripper

    # Copy wandb
    if [ -d "$POLICY_B_DIR_TEMP/checkpoints/wandb" ]; then
        cp -r "$POLICY_B_DIR_TEMP/checkpoints/wandb" "$POLICY_B_DIR_LAST/checkpoints/wandb"
        WANDB_LATEST="$POLICY_B_DIR_LAST/checkpoints/wandb/latest-run"
        if [ -d "$WANDB_LATEST" ]; then
            for mf in "$WANDB_LATEST"/files/wandb-metadata.json; do
                [ -f "$mf" ] && sed -i "s|$POLICY_B_DIR_TEMP|$POLICY_B_DIR_LAST|g" "$mf"
            done
        fi
    fi

    # Convert last → symlink
    CKPT_DIR="$POLICY_B_DIR_LAST/checkpoints/checkpoints"
    LAST_DIR="$CKPT_DIR/last"
    if [ -d "$LAST_DIR" ] && [ ! -L "$LAST_DIR" ]; then
        CUR_STEP=$(get_current_step "$POLICY_B_DIR_LAST")
        STEP_NAME=$(printf "%06d" $CUR_STEP)
        mv "$LAST_DIR" "$CKPT_DIR/$STEP_NAME"
        (cd "$CKPT_DIR" && ln -s "$STEP_NAME" last)
        echo "  ✓ Converted: last → $STEP_NAME (symlink)"
    fi

    # Train
    CUR_STEP_B=$(get_current_step "$POLICY_B_DIR_LAST")
    TARGET_B=$((CUR_STEP_B + STEPS_PER_ITER))
    echo "  Training: step $CUR_STEP_B → $TARGET_B ($NUM_TRAINING_GPUS GPUs)"

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

    echo "  ✓ Policy B trained to step $TARGET_B"

    # Rotate: last → temp
    rm -rf "$POLICY_B_DIR_TEMP"
    mv "$POLICY_B_DIR_LAST" "$POLICY_B_DIR_TEMP"
    echo "  ✓ Rotated B: last → temp"

    # =====================================================================
    # Stage checkpoints for eval script
    # =====================================================================
    print_section "Staging checkpoints for Eval Iteration $EVAL_ITER"

    ITER_STAGING="$STAGING_DIR/iter_${EVAL_ITER}"
    rm -rf "$ITER_STAGING"
    mkdir -p "$ITER_STAGING/policy_A" "$ITER_STAGING/policy_B"

    # Copy pretrained_model dirs (the actual weights)
    CKPT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEMP")
    CKPT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEMP")

    cp -r "$CKPT_A" "$ITER_STAGING/policy_A/pretrained_model"
    cp -r "$CKPT_B" "$ITER_STAGING/policy_B/pretrained_model"

    # Record step info
    FINAL_STEP_A=$(get_current_step "$POLICY_A_DIR_TEMP")
    FINAL_STEP_B=$(get_current_step "$POLICY_B_DIR_TEMP")
    echo "{\"step_A\": $FINAL_STEP_A, \"step_B\": $FINAL_STEP_B}" > "$ITER_STAGING/info.json"

    # Signal readiness
    touch "$ITER_STAGING/ready"

    echo "  ✓ Staged: $ITER_STAGING"
    echo "  ✓ Steps: A=$FINAL_STEP_A, B=$FINAL_STEP_B"
    echo "  ✓ READY marker created"

    print_section "Training Iteration $train_iter complete! (Eval $EVAL_ITER ready)"
done

print_header "All Training Complete!"
echo "  Checkpoints staged in: $STAGING_DIR/iter_2/ through $STAGING_DIR/iter_10/"
echo "  Eval script should pick them up automatically."
