#!/bin/bash
# =============================================================================
# Rev2Fwd SAC Fine-tuning Pipeline
# =============================================================================
#
# 概述:
#   SAC (Soft Actor-Critic) 微调实验流程:
#     Phase 1: 确保 BC 预训练 checkpoint 存在
#     Phase 2: SAC-Gaussian 训练 (独立 Gaussian Actor)
#     Phase 3: SAC-Diffusion 训练 (微调 Diffusion Policy)
#     Phase 4: 公平评估 (两种模式 + baseline)
#
# 使用示例:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/scripts_sac/run_sac_pipeline.sh
#
#   # 只跑 Gaussian actor
#   CUDA_VISIBLE_DEVICES=0 SKIP_DIFFUSION=1 bash scripts/scripts_sac/run_sac_pipeline.sh
#
#   # 只跑 Diffusion actor
#   CUDA_VISIBLE_DEVICES=0 SKIP_GAUSSIAN=1 bash scripts/scripts_sac/run_sac_pipeline.sh
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

# --- Output ---
EXP_NAME="${EXP_NAME:-sac_exp1}"
BASE_DIR="runs/sac_experiments/${EXP_NAME}"

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

# --- SAC-specific ---
INIT_ALPHA="${INIT_ALPHA:-0.2}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
LR_ACTOR="${LR_ACTOR:-3e-4}"
LR_ALPHA="${LR_ALPHA:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# --- Eval ---
EVAL_EPISODES=50

# --- Skip flags ---
SKIP_GAUSSIAN="${SKIP_GAUSSIAN:-0}"
SKIP_DIFFUSION="${SKIP_DIFFUSION:-0}"

# --- Misc ---
HEADLESS="--headless"
WANDB_PROJECT="${WANDB_PROJECT:-rev2fwd-sac-${EXP_NAME}}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# --- Script paths ---
SCRIPT_DIR="scripts/scripts_sac"

# =============================================================================
# Derived paths
# =============================================================================
GAUSS_DIR="${BASE_DIR}/sac_gaussian"
DIFF_DIR="${BASE_DIR}/sac_diffusion"
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
mkdir -p "$BASE_DIR" "$LOG_DIR" "$GAUSS_DIR" "$DIFF_DIR"

echo ""
echo "======================================================"
echo "  Rev2Fwd SAC Fine-tuning Pipeline"
echo "======================================================"
echo "  BC checkpoint A: $BC_CKPT_A"
echo "  Output:          $BASE_DIR"
echo "  GPU:             $GPU"
echo "  Num envs:        $NUM_ENVS"
echo "  Total steps:     $TOTAL_ENV_STEPS"
echo "  Reward type:     $REWARD_TYPE"
echo "  Init α:          $INIT_ALPHA"
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
# Phase 2: SAC-Gaussian Training
# =============================================================================
if [ "$SKIP_GAUSSIAN" != "1" ]; then
    echo ""
    echo "--- Phase 2: SAC-Gaussian Training ---"

    if ! phase_done phase2_sac_gaussian; then
        log "Starting SAC-Gaussian training..."
        CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/train_sac.py" \
            --policy_A_ckpt "$BC_CKPT_A" \
            --out "$GAUSS_DIR" \
            --actor_type gaussian \
            --task "Isaac-Lift-Cube-Franka-IK-Abs-v0" \
            --num_envs $NUM_ENVS \
            --total_env_steps $TOTAL_ENV_STEPS \
            --horizon $HORIZON \
            --goal_xy $GOAL_X $GOAL_Y \
            --distance_threshold $DISTANCE_THRESHOLD \
            --reward_type $REWARD_TYPE \
            --init_alpha $INIT_ALPHA \
            --lr_critic $LR_CRITIC \
            --lr_actor $LR_ACTOR \
            --lr_alpha $LR_ALPHA \
            --batch_size $BATCH_SIZE \
            --seed $SEED \
            --wandb --wandb_project "${WANDB_PROJECT}-gaussian" \
            $HEADLESS \
            2>&1 | tee "${LOG_DIR}/train_sac_gaussian.log"
        mark_done phase2_sac_gaussian
        log "SAC-Gaussian training complete"
    else
        log "[Skip] SAC-Gaussian training already done"
    fi
fi

# =============================================================================
# Phase 3: SAC-Diffusion Training
# =============================================================================
if [ "$SKIP_DIFFUSION" != "1" ]; then
    echo ""
    echo "--- Phase 3: SAC-Diffusion Training ---"

    if ! phase_done phase3_sac_diffusion; then
        log "Starting SAC-Diffusion training..."
        CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/train_sac.py" \
            --policy_A_ckpt "$BC_CKPT_A" \
            --out "$DIFF_DIR" \
            --actor_type diffusion \
            --task "Isaac-Lift-Cube-Franka-IK-Abs-v0" \
            --num_envs $NUM_ENVS \
            --total_env_steps $TOTAL_ENV_STEPS \
            --horizon $HORIZON \
            --goal_xy $GOAL_X $GOAL_Y \
            --distance_threshold $DISTANCE_THRESHOLD \
            --reward_type $REWARD_TYPE \
            --init_alpha $INIT_ALPHA \
            --lr_critic $LR_CRITIC \
            --lr_actor 1e-5 \
            --lr_alpha $LR_ALPHA \
            --batch_size $BATCH_SIZE \
            --seed $SEED \
            --wandb --wandb_project "${WANDB_PROJECT}-diffusion" \
            $HEADLESS \
            2>&1 | tee "${LOG_DIR}/train_sac_diffusion.log"
        mark_done phase3_sac_diffusion
        log "SAC-Diffusion training complete"
    else
        log "[Skip] SAC-Diffusion training already done"
    fi
fi

# =============================================================================
# Phase 4: Fair Evaluation
# =============================================================================
echo ""
echo "--- Phase 4: Fair Evaluation ---"

# Evaluate SAC-Gaussian
if [ "$SKIP_GAUSSIAN" != "1" ]; then
    if ! phase_done phase4_eval_gaussian; then
        log "Evaluating SAC-Gaussian..."
        CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/eval_sac.py" \
            --checkpoint "$GAUSS_DIR/latest_checkpoint.pt" \
            --bc_ckpt "$BC_CKPT_A" \
            --actor_type gaussian \
            --out "$GAUSS_DIR/eval_fair.json" \
            --num_episodes $EVAL_EPISODES \
            --horizon $HORIZON \
            --goal_xy $GOAL_X $GOAL_Y \
            --distance_threshold $DISTANCE_THRESHOLD \
            --seed $SEED \
            $HEADLESS \
            2>&1 | tee "${LOG_DIR}/eval_sac_gaussian.log"
        mark_done phase4_eval_gaussian
    else
        log "[Skip] SAC-Gaussian evaluation already done"
    fi
fi

# Evaluate SAC-Diffusion
if [ "$SKIP_DIFFUSION" != "1" ]; then
    if ! phase_done phase4_eval_diffusion; then
        log "Evaluating SAC-Diffusion..."
        CUDA_VISIBLE_DEVICES=$GPU python "${SCRIPT_DIR}/eval_sac.py" \
            --checkpoint "$DIFF_DIR/latest_checkpoint.pt" \
            --bc_ckpt "$BC_CKPT_A" \
            --actor_type diffusion \
            --out "$DIFF_DIR/eval_fair.json" \
            --num_episodes $EVAL_EPISODES \
            --horizon $HORIZON \
            --goal_xy $GOAL_X $GOAL_Y \
            --distance_threshold $DISTANCE_THRESHOLD \
            --seed $SEED \
            $HEADLESS \
            2>&1 | tee "${LOG_DIR}/eval_sac_diffusion.log"
        mark_done phase4_eval_diffusion
    else
        log "[Skip] SAC-Diffusion evaluation already done"
    fi
fi

# =============================================================================
# Print Results Summary
# =============================================================================
echo ""
echo "======================================================"
echo "  SAC Results Summary"
echo "======================================================"

for result in "$GAUSS_DIR/eval_fair.json" "$DIFF_DIR/eval_fair.json"; do
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
if [ "$SKIP_GAUSSIAN" != "1" ]; then
    echo "  SAC-Gaussian checkpoint:  $GAUSS_DIR/latest_checkpoint.pt"
fi
if [ "$SKIP_DIFFUSION" != "1" ]; then
    echo "  SAC-Diffusion checkpoint: $DIFF_DIR/latest_checkpoint.pt"
fi
echo ""

touch "$BASE_DIR/.complete"
log "SAC Pipeline complete!"
