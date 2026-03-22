#!/bin/bash
# =============================================================================
# Rev2Fwd RL Fine-tuning Pipeline
# =============================================================================
#
# 概述:
#   RL 微调实验流程:
#     Phase 1: 确保 BC 预训练 checkpoint 存在
#     Phase 2: TD-Diffusion 训练 (方案 A)
#     Phase 3: DPPO 训练 (方案 B)
#     Phase 4: 公平评估 (两种方案 + baseline)
#
# 使用示例:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/scripts_rl/run_rl_pipeline.sh
#
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================

# --- BC checkpoint (pretrained, required) ---
BC_CKPT_A="${BC_CKPT_A:-data/pick_place_isaac_lab_simulation/exp_new/weights/PP_A/checkpoints/checkpoints/last/pretrained_model}"
BC_CKPT_B="${BC_CKPT_B:-data/pick_place_isaac_lab_simulation/exp_new/weights/PP_B/checkpoints/checkpoints/last/pretrained_model}"

# --- Output ---
EXP_NAME="${EXP_NAME:-rl_exp1}"
BASE_DIR="runs/rl_experiments/${EXP_NAME}"

# --- Environment ---
GOAL_X="0.5"
GOAL_Y="-0.2"
DISTANCE_THRESHOLD="0.03"
NUM_ENVS="${NUM_ENVS:-16}"
HORIZON=1500

# --- Training ---
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-500000}"
REWARD_TYPE="${REWARD_TYPE:-dense}"
SEED=42

# --- Eval ---
EVAL_EPISODES=50

# --- Misc ---
HEADLESS="--headless"
WANDB_PROJECT="${WANDB_PROJECT:-rev2fwd-rl-${EXP_NAME}}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# --- Script paths ---
SCRIPT_DIR="scripts/scripts_rl"

# =============================================================================
# Derived paths
# =============================================================================
TD_DIR="${BASE_DIR}/td_diffusion"
DPPO_DIR="${BASE_DIR}/dppo"
LOG_DIR="${BASE_DIR}/logs"

# =============================================================================
# Helpers
# =============================================================================
log() { echo "[$(date '+%H:%M:%S')] $*"; }
phase_done() { [ -f "$BASE_DIR/.done_${1}" ]; }
mark_done()  { touch "$BASE_DIR/.done_${1}"; }

# =============================================================================
# Initialize
# =============================================================================
mkdir -p "$BASE_DIR" "$LOG_DIR" "$TD_DIR" "$DPPO_DIR"

echo ""
echo "======================================================"
echo "  Rev2Fwd RL Fine-tuning Pipeline"
echo "======================================================"
echo "  BC checkpoint A: $BC_CKPT_A"
echo "  BC checkpoint B: $BC_CKPT_B"
echo "  Output:          $BASE_DIR"
echo "  GPU:             $GPU"
echo "  Num envs:        $NUM_ENVS"
echo "  Total steps:     $TOTAL_ENV_STEPS"
echo "  Reward type:     $REWARD_TYPE"
echo ""

# =============================================================================
# Phase 1: Verify BC checkpoints
# =============================================================================
echo "--- Phase 1: Verify BC Checkpoints ---"

if [ ! -f "$BC_CKPT_A/model.safetensors" ]; then
    echo "ERROR: BC checkpoint A not found: $BC_CKPT_A"
    echo "Run the BC training pipeline first (run_pipeline.sh)"
    exit 1
fi
log "BC checkpoint A verified: $BC_CKPT_A"

# =============================================================================
# Phase 2: TD-Diffusion Training (Plan A)
# =============================================================================
echo ""
echo "--- Phase 2: TD-Diffusion Training (Plan A) ---"

if ! phase_done phase2_td; then
    log "Starting TD-Diffusion training..."
    CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/train_td_diffusion.py" \
        --policy_A_ckpt "$BC_CKPT_A" \
        --out "$TD_DIR" \
        --task "Isaac-Lift-Cube-Franka-IK-Abs-v0" \
        --num_envs $NUM_ENVS \
        --total_env_steps $TOTAL_ENV_STEPS \
        --horizon $HORIZON \
        --goal_xy $GOAL_X $GOAL_Y \
        --distance_threshold $DISTANCE_THRESHOLD \
        --reward_type $REWARD_TYPE \
        --seed $SEED \
        --wandb --wandb_project "${WANDB_PROJECT}-td" \
        $HEADLESS \
        2>&1 | tee "${LOG_DIR}/train_td.log"
    mark_done phase2_td
    log "TD-Diffusion training complete"
else
    log "[Skip] TD-Diffusion training already done"
fi

# =============================================================================
# Phase 3: DPPO Training (Plan B)
# =============================================================================
echo ""
echo "--- Phase 3: DPPO Training (Plan B) ---"

if ! phase_done phase3_dppo; then
    log "Starting DPPO training..."
    CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/train_dppo.py" \
        --policy_A_ckpt "$BC_CKPT_A" \
        --out "$DPPO_DIR" \
        --task "Isaac-Lift-Cube-Franka-IK-Abs-v0" \
        --num_envs $NUM_ENVS \
        --total_env_steps $TOTAL_ENV_STEPS \
        --horizon $HORIZON \
        --goal_xy $GOAL_X $GOAL_Y \
        --distance_threshold $DISTANCE_THRESHOLD \
        --reward_type $REWARD_TYPE \
        --seed $SEED \
        --wandb --wandb_project "${WANDB_PROJECT}-dppo" \
        $HEADLESS \
        2>&1 | tee "${LOG_DIR}/train_dppo.log"
    mark_done phase3_dppo
    log "DPPO training complete"
else
    log "[Skip] DPPO training already done"
fi

# =============================================================================
# Phase 4: Fair Evaluation
# =============================================================================
echo ""
echo "--- Phase 4: Fair Evaluation ---"

# Evaluate TD-Diffusion
if ! phase_done phase4_eval_td; then
    log "Evaluating TD-Diffusion..."
    CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/eval_rl.py" \
        --checkpoint "$TD_DIR/latest_checkpoint.pt" \
        --bc_ckpt "$BC_CKPT_A" \
        --out "$TD_DIR/eval_fair.json" \
        --num_episodes $EVAL_EPISODES \
        --horizon $HORIZON \
        --goal_xy $GOAL_X $GOAL_Y \
        --distance_threshold $DISTANCE_THRESHOLD \
        --seed $SEED \
        $HEADLESS \
        2>&1 | tee "${LOG_DIR}/eval_td.log"
    mark_done phase4_eval_td
else
    log "[Skip] TD-Diffusion evaluation already done"
fi

# Evaluate DPPO
if ! phase_done phase4_eval_dppo; then
    log "Evaluating DPPO..."
    CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/eval_rl.py" \
        --checkpoint "$DPPO_DIR/latest_checkpoint.pt" \
        --bc_ckpt "$BC_CKPT_A" \
        --out "$DPPO_DIR/eval_fair.json" \
        --num_episodes $EVAL_EPISODES \
        --horizon $HORIZON \
        --goal_xy $GOAL_X $GOAL_Y \
        --distance_threshold $DISTANCE_THRESHOLD \
        --seed $SEED \
        $HEADLESS \
        2>&1 | tee "${LOG_DIR}/eval_dppo.log"
    mark_done phase4_eval_dppo
else
    log "[Skip] DPPO evaluation already done"
fi

# =============================================================================
# Print Results Summary
# =============================================================================
echo ""
echo "======================================================"
echo "  Results Summary"
echo "======================================================"

for result in "$TD_DIR/eval_fair.json" "$DPPO_DIR/eval_fair.json"; do
    if [ -f "$result" ]; then
        python3 -c "
import json
with open('$result') as f:
    d = json.load(f)
method = d.get('method', 'unknown')
sr = d['success_rate']
n = d['num_success']
total = d['num_total']
print(f'  {method}: {n}/{total} = {sr*100:.1f}%  (avg_reward={d[\"avg_reward\"]:.1f})')
"
    fi
done

echo ""
echo "  Results saved to: $BASE_DIR"
echo "  TD checkpoint:    $TD_DIR/latest_checkpoint.pt"
echo "  DPPO checkpoint:  $DPPO_DIR/latest_checkpoint.pt"
echo ""

touch "$BASE_DIR/.complete"
log "Pipeline complete!"
