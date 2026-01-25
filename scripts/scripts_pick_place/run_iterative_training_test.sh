#!/bin/bash
# =============================================================================
# TEST VERSION: Iterative Training Script for Rev2Fwd Imitation Learning
# =============================================================================
# This is a test version with reduced parameters for quick validation.
# After testing passes, use run_iterative_training.sh for full training.
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration (REDUCED FOR TESTING)
# =============================================================================
MAX_ITERATIONS=1            # Only 1 iteration for testing
STEPS_PER_ITER=200          # Only 200 training steps for testing
MAX_CYCLES=5                # 5 A→B cycles for testing (increased from 3)
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
POLICY_A_DIR_TEST="runs/PP_A_circle_finetune_test"
POLICY_B_DIR_TEST="runs/PP_B_circle_finetune_test"

# Original training data (MODIFY THESE FOR YOUR SETUP)
DATA_A="data/A_circle.npz"
DATA_B="data/B_circle.npz"
# Goal position (plate center)
GOAL_X=0.5
GOAL_Y=0.0

# GPU configuration
# Set CUDA_VISIBLE_DEVICES before running this script, or modify here
# Example: export CUDA_VISIBLE_DEVICES=0

# Additional flags
INCLUDE_OBJ_POSE="--include_obj_pose"  # Include object pose in observations
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

# =============================================================================
# Pre-flight Checks
# =============================================================================
print_header "Pre-flight Checks (TEST MODE)"

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
echo "  POLICY_A_DIR_TEST:  $POLICY_A_DIR_TEST (finetune output)"
echo "  POLICY_B_DIR_TEST:  $POLICY_B_DIR_TEST (finetune output)"
echo "  DATA_A:             $DATA_A"
echo "  DATA_B:             $DATA_B"
echo "  GOAL:               ($GOAL_X, $GOAL_Y)"

# Prompt user to confirm
echo ""
read -p "This is a TEST run. Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================================================
# Main Loop
# =============================================================================
for iter in $(seq 1 $MAX_ITERATIONS); do
    print_header "Iteration $iter / $MAX_ITERATIONS (TEST MODE)"
    
    # Define output paths for this iteration (use _test suffix to avoid overwriting real data)
    ROLLOUT_A="data/rollout_A_circle_iter${iter}_test.npz"
    ROLLOUT_B="data/rollout_B_circle_iter${iter}_test.npz"
    
    # Get current checkpoint paths (use finetuned checkpoints if they exist, otherwise use original)
    if [ $iter -eq 1 ]; then
        # First iteration: use original checkpoints
        CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR")
        CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR")
    else
        # Subsequent iterations: use finetuned checkpoints from test output
        CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR_TEST")
        CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR_TEST")
        
        # Fall back to original if finetuned doesn't exist
        if [ ! -d "$CHECKPOINT_A" ]; then
            CHECKPOINT_A=$(get_checkpoint_path "$POLICY_A_DIR")
        fi
        if [ ! -d "$CHECKPOINT_B" ]; then
            CHECKPOINT_B=$(get_checkpoint_path "$POLICY_B_DIR")
        fi
    fi
    
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
    
    echo "  Original data: $DATA_A"
    if [ -f "$ROLLOUT_A" ]; then
        echo "  Rollout data: $ROLLOUT_A"
        ROLLOUT_A_ARG="--rollout_data $ROLLOUT_A"
    else
        echo "  Rollout data: (none - using original data only)"
        ROLLOUT_A_ARG=""
    fi
    echo "  Checkpoint: $CHECKPOINT_A"
    echo "  Output: $POLICY_A_DIR_TEST (test output, original checkpoint preserved)"
    echo "  Steps: $STEPS_PER_ITER"
    
    # Finetuning - rollout_data is optional
    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_data "$DATA_A" \
        $ROLLOUT_A_ARG \
        --checkpoint "$CHECKPOINT_A" \
        --out "$POLICY_A_DIR_TEST" \
        --steps $STEPS_PER_ITER \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        $INCLUDE_OBJ_POSE
    
    echo "✓ Policy A finetuning complete (saved to $POLICY_A_DIR_TEST)"
    
    # =========================================================================
    # Step 3: Finetune Policy B (always finetune, with or without rollout data)
    # =========================================================================
    print_section "[Step 3] Finetuning Policy B (TEST MODE - $STEPS_PER_ITER steps)..."
    
    echo "  Original data: $DATA_B"
    if [ -f "$ROLLOUT_B" ]; then
        echo "  Rollout data: $ROLLOUT_B"
        ROLLOUT_B_ARG="--rollout_data $ROLLOUT_B"
    else
        echo "  Rollout data: (none - using original data only)"
        ROLLOUT_B_ARG=""
    fi
    echo "  Checkpoint: $CHECKPOINT_B"
    echo "  Output: $POLICY_B_DIR_TEST (test output, original checkpoint preserved)"
    echo "  Steps: $STEPS_PER_ITER"
    
    # Finetuning - rollout_data is optional
    python scripts/scripts_pick_place/7_finetune_with_rollout.py \
        --original_data "$DATA_B" \
        $ROLLOUT_B_ARG \
        --checkpoint "$CHECKPOINT_B" \
        --out "$POLICY_B_DIR_TEST" \
        --steps $STEPS_PER_ITER \
        --batch_size $BATCH_SIZE \
        --n_action_steps $N_ACTION_STEPS \
        $INCLUDE_OBJ_POSE
    
    echo "✓ Policy B finetuning complete (saved to $POLICY_B_DIR_TEST)"
    
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
echo "Finetuned test checkpoints:"
ls -la $POLICY_A_DIR_TEST/checkpoints/checkpoints/last/pretrained_model 2>/dev/null || echo "  Policy A: (not created)"
ls -la $POLICY_B_DIR_TEST/checkpoints/checkpoints/last/pretrained_model 2>/dev/null || echo "  Policy B: (not created)"
echo ""
echo "Original checkpoints preserved at:"
echo "  $POLICY_A_DIR"
echo "  $POLICY_B_DIR"
echo ""
echo "To run full iterative training, use:"
echo "  bash scripts/scripts_pick_place/run_iterative_training.sh"
echo ""
