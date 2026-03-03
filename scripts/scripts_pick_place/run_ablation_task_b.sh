#!/bin/bash
# =============================================================================
# Ablation Experiments for Task B Success Rate Decline
# =============================================================================
#
# Two experiments to diagnose why Task B success rate decreases:
#
# Experiment A  (--run_exp_a):
#   Hypothesis: The A→B chaining / reset mechanism causes B's decline.
#   Method:     Use iteration-10 weights. Evaluate Task A and Task B
#               INDEPENDENTLY (full env.reset before every single episode).
#               50 episodes each. No A→B coupling.
#
# Experiment B  (--run_exp_b):
#   Hypothesis: Insufficient convergence of new Task B data causes decline.
#   Method:     Use Task A iter-10 model + original Task B model (no new
#               data finetuning). Run 50 A→B cycles with the standard
#               recovery pipeline (reset only after each task).
#               If the un-finetuned B still performs well, the new data +
#               insufficient training is the culprit.
#
# Usage:
#   # Run both experiments
#   bash scripts/scripts_pick_place/run_ablation_task_b.sh
#
#   # Run only experiment A
#   bash scripts/scripts_pick_place/run_ablation_task_b.sh --run_exp_a
#
#   # Run only experiment B
#   bash scripts/scripts_pick_place/run_ablation_task_b.sh --run_exp_b
# =============================================================================

set -e

# =============================================================================
# Conda
# =============================================================================
eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================
NUM_EPISODES=50             # Episodes per task (Exp A) / cycles (Exp B)
HORIZON=400
DISTANCE_THRESHOLD=0.05
N_ACTION_STEPS=16
GOAL_X=0.5
GOAL_Y=0.0

# GPU
DATA_COLLECTION_GPU=0

# Checkpoint paths
# Iteration 10 final weights (after all 10 rounds of finetuning)
POLICY_A_ITER10="runs/PP_A_success_rate_temp/checkpoints/checkpoints/last/pretrained_model"
POLICY_B_ITER10="runs/PP_B_success_rate_temp/checkpoints/checkpoints/last/pretrained_model"

# Original weights (before any finetuning with new data)
POLICY_B_ORIGINAL="runs/PP_B_circle/checkpoints/checkpoints/last/pretrained_model"

HEADLESS="--headless"

# =============================================================================
# Parse arguments
# =============================================================================
RUN_EXP_A=false
RUN_EXP_B=false

for arg in "$@"; do
    case $arg in
        --run_exp_a) RUN_EXP_A=true ;;
        --run_exp_b) RUN_EXP_B=true ;;
    esac
done

# Default: run both
if ! $RUN_EXP_A && ! $RUN_EXP_B; then
    RUN_EXP_A=true
    RUN_EXP_B=true
fi

# =============================================================================
# Pre-flight checks
# =============================================================================
echo ""
echo "=============================================="
echo "Pre-flight Checks"
echo "=============================================="

for ckpt in "$POLICY_A_ITER10" "$POLICY_B_ITER10" "$POLICY_B_ORIGINAL"; do
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        exit 1
    fi
done
echo "✓ All checkpoints found"

echo ""
echo "  Policy A (iter 10): $POLICY_A_ITER10"
echo "  Policy B (iter 10): $POLICY_B_ITER10"
echo "  Policy B (original): $POLICY_B_ORIGINAL"

# =============================================================================
# Experiment A: Independent evaluation (always reset)
# =============================================================================
if $RUN_EXP_A; then
    echo ""
    echo "=============================================="
    echo "Experiment A: Independent Evaluation"
    echo "=============================================="
    echo "  Hypothesis: A→B chaining / reset mechanism causes B decline"
    echo "  Method:     Full env.reset() before EVERY episode, no chaining"
    echo "  Weights:    A=iter10, B=iter10"
    echo "  Episodes:   ${NUM_EPISODES} per task (independent)"
    echo ""

    EXP_A_OUT="data/ablation_exp_a_independent.stats.json"
    rm -f "$EXP_A_OUT"

    CUDA_VISIBLE_DEVICES=$DATA_COLLECTION_GPU python scripts/scripts_pick_place/10_eval_independent.py \
        --policy_A "$POLICY_A_ITER10" \
        --policy_B "$POLICY_B_ITER10" \
        --out "$EXP_A_OUT" \
        --num_episodes $NUM_EPISODES \
        --horizon $HORIZON \
        --distance_threshold $DISTANCE_THRESHOLD \
        --n_action_steps $N_ACTION_STEPS \
        --goal_xy $GOAL_X $GOAL_Y \
        $HEADLESS

    echo ""
    echo "----------------------------------------------"
    echo "Experiment A Results"
    echo "----------------------------------------------"
    if [ -f "$EXP_A_OUT" ]; then
        python3 -c "
import json
with open('$EXP_A_OUT', 'r') as f:
    s = json.load(f)['summary']
print(f'  Task A (independent): {s[\"task_A_success_count\"]}/{s[\"task_A_total_episodes\"]} = {s[\"task_A_success_rate\"]*100:.1f}%')
print(f'  Task B (independent): {s[\"task_B_success_count\"]}/{s[\"task_B_total_episodes\"]} = {s[\"task_B_success_rate\"]*100:.1f}%')
print()
print('  Compare with pipeline iter 10 (alternating):')
print('    Task A: 36.0%    Task B: 80.0%')
print()
print('  If Task B independent ≈ Task B alternating → reset mechanism is NOT the cause')
print('  If Task B independent >> Task B alternating → reset mechanism IS a factor')
"
    fi
fi

# =============================================================================
# Experiment B: Original B model with iter-10 A model (alternating)
# =============================================================================
if $RUN_EXP_B; then
    echo ""
    echo "=============================================="
    echo "Experiment B: Untrained B + Trained A (Alternating)"
    echo "=============================================="
    echo "  Hypothesis: Insufficient convergence of new B data causes decline"
    echo "  Method:     A=iter10 + B=original, 50 A→B cycles with recovery"
    echo "  If original B stays ~90%: finetuning with new data hurt convergence"
    echo ""

    EXP_B_OUT_A="data/ablation_exp_b_eval_A.npz"
    EXP_B_OUT_B="data/ablation_exp_b_eval_B.npz"
    EXP_B_STATS="${EXP_B_OUT_A%.npz}.stats.json"
    rm -f "$EXP_B_OUT_A" "$EXP_B_OUT_B" "$EXP_B_STATS"

    CUDA_VISIBLE_DEVICES=$DATA_COLLECTION_GPU python scripts/scripts_pick_place/9_eval_with_recovery.py \
        --policy_A "$POLICY_A_ITER10" \
        --policy_B "$POLICY_B_ORIGINAL" \
        --out_A "$EXP_B_OUT_A" \
        --out_B "$EXP_B_OUT_B" \
        --num_cycles $NUM_EPISODES \
        --horizon $HORIZON \
        --distance_threshold $DISTANCE_THRESHOLD \
        --n_action_steps $N_ACTION_STEPS \
        --goal_xy $GOAL_X $GOAL_Y \
        $HEADLESS

    echo ""
    echo "----------------------------------------------"
    echo "Experiment B Results"
    echo "----------------------------------------------"
    if [ -f "$EXP_B_STATS" ]; then
        python3 -c "
import json
with open('$EXP_B_STATS', 'r') as f:
    s = json.load(f)['summary']
print(f'  Task A (iter10):    {s[\"task_A_success_count\"]}/{s[\"total_task_A_episodes\"]} = {s[\"task_A_success_rate\"]*100:.1f}%')
print(f'  Task B (original):  {s[\"task_B_success_count\"]}/{s[\"total_task_B_episodes\"]} = {s[\"task_B_success_rate\"]*100:.1f}%')
print()
print('  Compare with pipeline results:')
print('    Iter 1  (B original):  B=90.0%')
print('    Iter 10 (B finetuned): B=80.0%')
print()
print('  If B(original) still ~90% here → finetuning introduced noise / insufficient convergence')
print('  If B(original) also drops → the decline is NOT due to finetuning')
"
    fi
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Ablation Experiments Complete"
echo "=============================================="

if $RUN_EXP_A && [ -f "data/ablation_exp_a_independent.stats.json" ]; then
    echo ""
    echo "  Experiment A (Independent eval):"
    python3 -c "
import json
with open('data/ablation_exp_a_independent.stats.json', 'r') as f:
    s = json.load(f)['summary']
print(f'    Task A: {s[\"task_A_success_rate\"]*100:.1f}%')
print(f'    Task B: {s[\"task_B_success_rate\"]*100:.1f}%')
"
fi

if $RUN_EXP_B && [ -f "data/ablation_exp_b_eval_A.stats.json" ]; then
    echo ""
    echo "  Experiment B (A-iter10 + B-original, alternating):"
    python3 -c "
import json
with open('data/ablation_exp_b_eval_A.stats.json', 'r') as f:
    s = json.load(f)['summary']
print(f'    Task A (iter10):    {s[\"task_A_success_rate\"]*100:.1f}%')
print(f'    Task B (original):  {s[\"task_B_success_rate\"]*100:.1f}%')
"
fi

echo ""
echo "  Pipeline reference (iter 10, alternating):"
echo "    Task A: 36.0%    Task B: 80.0%"
echo ""
