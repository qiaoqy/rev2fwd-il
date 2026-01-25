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
# Configuration
# =============================================================================
MAX_ITERATIONS=10           # Maximum number of test-finetune iterations
STEPS_PER_ITER=5000         # Training steps per finetuning iteration
MAX_CYCLES=50               # Maximum A→B cycles per alternating test
HORIZON=400                 # Maximum steps per task attempt
BATCH_SIZE=32               # Training batch size

# Thresholds for success detection
# Note: New success criteria requires object on table (z<0.05), at goal (XY distance), and gripper open
DISTANCE_THRESHOLD=0.05     # Maximum distance from target for success

# Action chunk settings
N_ACTION_STEPS=16           # Number of action steps to execute per inference

# Policy directories
POLICY_A_DIR="runs/diffusion_A_circle"
POLICY_B_DIR="runs/diffusion_B_circle"

# Original training data
DATA_A="data/A_circle.npz"
DATA_B="data/B_circle.npz"

# Goal position (plate center)
GOAL_X=0.5
GOAL_Y=0.0

# GPU configuration
# Set CUDA_VISIBLE_DEVICES before running this script, or modify here
# Example: CUDA_VISIBLE_DEVICES=0

# Additional flags
INCLUDE_OBJ_POSE="--include_obj_pose"  # Include object pose in observations
HEADLESS="--headless"                   # Run in headless mode

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

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks"

# Check that original data exists
check_file_exists "$DATA_A"
check_file_exists "$DATA_B"

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

echo "✓ Original data files found"
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
echo "  POLICY_A_DIR:       $POLICY_A_DIR"
echo "  POLICY_B_DIR:       $POLICY_B_DIR"
echo "  DATA_A:             $DATA_A"
echo "  DATA_B:             $DATA_B"
echo "  GOAL:               ($GOAL_X, $GOAL_Y)"

# =============================================================================
# Main Loop
# =============================================================================
for iter in $(seq 1 $MAX_ITERATIONS); do
    print_header "Iteration $iter / $MAX_ITERATIONS"
    
    # Define output paths for this iteration
    ROLLOUT_A="data/rollout_A_circle_iter${iter}.npz"
    ROLLOUT_B="data/rollout_B_circle_iter${iter}.npz"
    
    # Get current checkpoint paths
    CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR")
    CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR")
    
    # =========================================================================
    # Step 1: Alternating Test (with retry until data is collected)
    # =========================================================================
    print_section "[Step 1] Running alternating test..."
    
    echo "  Checkpoint A: $CHECKPOINT_A"
    echo "  Checkpoint B: $CHECKPOINT_B"
    echo "  Output A: $ROLLOUT_A"
    echo "  Output B: $ROLLOUT_B"
    
    MAX_RETRIES=10  # Maximum number of retries for data collection
    retry_count=0
    
    while true; do
        retry_count=$((retry_count + 1))
        echo ""
        echo "  [Attempt $retry_count/$MAX_RETRIES] Running alternating test..."
        
        # Remove any existing partial data files before retry
        rm -f "$ROLLOUT_A" "$ROLLOUT_B"
        
        python scripts/scripts_pick_place/6_test_alternating.py \
            --policy_A "$CHECKPOINT_A" \
            --policy_B "$CHECKPOINT_B" \
            --out_A "$ROLLOUT_A" \
            --out_B "$ROLLOUT_B" \
            --max_cycles $MAX_CYCLES \
            --horizon $HORIZON \
            --distance_threshold $DISTANCE_THRESHOLD \
            --n_action_steps $N_ACTION_STEPS \
            --goal_xy $GOAL_X $GOAL_Y \
            $HEADLESS
        
        # Check if BOTH rollout data files were collected
        if [ -f "$ROLLOUT_A" ] && [ -f "$ROLLOUT_B" ]; then
            echo "  ✓ Both Task A and Task B rollout data collected!"
            break
        elif [ -f "$ROLLOUT_A" ]; then
            echo "  ⚠ Only Task A data collected, Task B missing. Retrying..."
        elif [ -f "$ROLLOUT_B" ]; then
            echo "  ⚠ Only Task B data collected, Task A missing. Retrying..."
        else
            echo "  ✗ No rollout data collected. Retrying..."
        fi
        
        if [ $retry_count -ge $MAX_RETRIES ]; then
            echo "  ERROR: Failed to collect rollout data after $MAX_RETRIES attempts."
            echo "  This may indicate that the policies are not working properly."
            echo "  Stopping iteration loop."
            break 2  # Break out of both loops
        fi
        
        echo "  Waiting 2 seconds before retry..."
        sleep 2
    done
    
    # Final check if rollout data was collected
    if [ ! -f "$ROLLOUT_A" ] || [ ! -f "$ROLLOUT_B" ]; then
        echo "WARNING: Could not collect complete rollout data in iteration $iter"
        echo "Stopping iteration loop."
        break
    fi
    
    # =========================================================================
    # Step 2: Finetune Policy A (if rollout data exists)
    # =========================================================================
    if [ -f "$ROLLOUT_A" ]; then
        print_section "[Step 2] Finetuning Policy A..."
        
        echo "  Original data: $DATA_A"
        echo "  Rollout data: $ROLLOUT_A"
        echo "  Checkpoint: $CHECKPOINT_A"
        echo "  Output: $POLICY_A_DIR"
        echo "  Steps: $STEPS_PER_ITER"
        
        python scripts/scripts_pick_place/7_finetune_with_rollout.py \
            --original_data "$DATA_A" \
            --rollout_data "$ROLLOUT_A" \
            --checkpoint "$CHECKPOINT_A" \
            --out "$POLICY_A_DIR" \
            --steps $STEPS_PER_ITER \
            --batch_size $BATCH_SIZE \
            --n_action_steps $N_ACTION_STEPS \
            $INCLUDE_OBJ_POSE
        
        echo "✓ Policy A finetuning complete"
    else
        echo "Skipping Policy A finetuning (no rollout data)"
    fi
    
    # =========================================================================
    # Step 3: Finetune Policy B (if rollout data exists)
    # =========================================================================
    if [ -f "$ROLLOUT_B" ]; then
        print_section "[Step 3] Finetuning Policy B..."
        
        echo "  Original data: $DATA_B"
        echo "  Rollout data: $ROLLOUT_B"
        echo "  Checkpoint: $CHECKPOINT_B"
        echo "  Output: $POLICY_B_DIR"
        echo "  Steps: $STEPS_PER_ITER"
        
        python scripts/scripts_pick_place/7_finetune_with_rollout.py \
            --original_data "$DATA_B" \
            --rollout_data "$ROLLOUT_B" \
            --checkpoint "$CHECKPOINT_B" \
            --out "$POLICY_B_DIR" \
            --steps $STEPS_PER_ITER \
            --batch_size $BATCH_SIZE \
            --n_action_steps $N_ACTION_STEPS \
            $INCLUDE_OBJ_POSE
        
        echo "✓ Policy B finetuning complete"
    else
        echo "Skipping Policy B finetuning (no rollout data)"
    fi
    
    print_section "Iteration $iter complete!"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
print_header "Iterative Training Finished!"
echo ""
echo "Final checkpoints:"
echo "  Policy A: $(get_checkpoint_path "$POLICY_A_DIR")"
echo "  Policy B: $(get_checkpoint_path "$POLICY_B_DIR")"
echo ""
echo "Rollout data saved in data/ directory:"
ls -la data/rollout_*_iter*.npz 2>/dev/null || echo "  (no rollout data files found)"
echo ""
echo "To evaluate the final policies, run:"
echo "  python scripts/scripts_pick_place/5_eval_diffusion.py \\"
echo "      --checkpoint $(get_checkpoint_path "$POLICY_A_DIR") \\"
echo "      --out_dir ${POLICY_A_DIR}/videos_final \\"
echo "      --num_episodes 10 --headless"
echo ""
