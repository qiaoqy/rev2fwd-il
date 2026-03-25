#!/bin/bash
# =============================================================================
# RECAP RL Pipeline for Pick-and-Place Simulator (Task A + B)
# =============================================================================
#
# Implements one full RECAP iteration:
#   Step 1: Collect rollout data (all episodes, success + failure)
#   Step 2: Train distributional value function (offline, no simulator)
#   Step 3: Compute per-frame advantages and binary indicators (offline)
#   Step 4: Fine-tune DP with advantage conditioning (offline)
#   Step 5: Fair evaluation of RECAP policy (simulator)
#
# Auto-resumable via .done_{step} marker files.
#
# Usage:
#   # First iteration (requires pre-trained Policy A and B from run_pipeline.sh)
#   EXP_NAME=recap_iter1 \
#   POLICY_A=data/pick_place.../weights/PP_A/checkpoints/checkpoints/last/pretrained_model \
#   POLICY_B=data/pick_place.../weights/PP_B/checkpoints/checkpoints/last/pretrained_model \
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/scripts_recap_rl/run_recap_pipeline.sh
#
#   # Subsequent iterations (use RECAP policy from previous iteration)
#   EXP_NAME=recap_iter2 \
#   POLICY_A=data/recap_iter1/recap_A/checkpoints/.../pretrained_model \
#   POLICY_B=data/recap_iter1/recap_B/checkpoints/.../pretrained_model \
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/scripts_recap_rl/run_recap_pipeline.sh
# =============================================================================

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate rev2fwd_il
echo "Activated: $CONDA_DEFAULT_ENV"

# =============================================================================
# Configuration
# =============================================================================

EXP_NAME="${EXP_NAME:-recap_iter1}"
BASE_DIR="data/recap/${EXP_NAME}"

# Policy checkpoints (REQUIRED — set these via environment variables)
if [ -z "$POLICY_A" ] || [ -z "$POLICY_B" ]; then
    echo "ERROR: Set POLICY_A and POLICY_B environment variables before running."
    echo "  POLICY_A=<path_to_task_A_checkpoint> \\"
    echo "  POLICY_B=<path_to_task_B_checkpoint> \\"
    echo "  bash scripts/scripts_recap_rl/run_recap_pipeline.sh"
    exit 1
fi

# --- Demo data (from initial Rev2Fwd training) ---
DEMO_A="${DEMO_A:-data/pick_place_isaac_lab_simulation/exp_new/task_A_reversed_100.npz}"
DEMO_B="${DEMO_B:-data/pick_place_isaac_lab_simulation/exp_new/task_B_100.npz}"

# --- Rollout collection (parallel alternating A→B) ---
CYCLES_PER_GPU="${CYCLES_PER_GPU:-10}"  # A→B cycles per GPU
HORIZON="${HORIZON:-1200}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"

# --- Value function training ---
VF_EPOCHS="${VF_EPOCHS:-300}"
VF_BATCH_SIZE="${VF_BATCH_SIZE:-512}"
VF_LR="${VF_LR:-1e-3}"
C_FAIL="${C_FAIL:-1200}"
MAX_EP_LEN="${MAX_EP_LEN:-1200}"
PERCENTILE="${PERCENTILE:-30}"
NUM_BINS="${NUM_BINS:-32}"

# --- RECAP fine-tuning ---
RECAP_STEPS="${RECAP_STEPS:-15000}"
RECAP_BATCH_SIZE="${RECAP_BATCH_SIZE:-64}"
RECAP_LR="${RECAP_LR:-5e-5}"
NULL_PROB="${NULL_PROB:-0.2}"
ALPHA="${ALPHA:-1.0}"

# --- Evaluation ---
EVAL_EPISODES="${EVAL_EPISODES:-100}"

# --- GPU assignment ---
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_TRAIN_GPUS=$(echo "$TRAIN_GPUS" | tr ',' '\n' | wc -l)
IFS=',' read -ra _GPU_LIST <<< "$TRAIN_GPUS"
COLLECT_GPU="${_GPU_LIST[0]:-0}"
EVAL_GPU="${_GPU_LIST[0]:-0}"

# --- NCCL ---
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

WANDB_PROJECT="${WANDB_PROJECT:-rev2fwd-recap-${EXP_NAME}}"
SEED=42

SCRIPT_DIR="scripts/scripts_recap_rl"

# =============================================================================
# Derived paths
# =============================================================================
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$BASE_DIR" "$LOG_DIR"

ROLLOUT_A_NPZ="${BASE_DIR}/rollouts_A.npz"
ROLLOUT_B_NPZ="${BASE_DIR}/rollouts_B.npz"

VF_A_PT="${BASE_DIR}/vf_A.pt"
VF_B_PT="${BASE_DIR}/vf_B.pt"

ADV_A_NPZ="${BASE_DIR}/advantages_A.npz"
ADV_B_NPZ="${BASE_DIR}/advantages_B.npz"

ADV_A_STATS="${BASE_DIR}/advantage_stats_A.json"
ADV_B_STATS="${BASE_DIR}/advantage_stats_B.json"

RECAP_A_DIR="${BASE_DIR}/recap_A"
RECAP_B_DIR="${BASE_DIR}/recap_B"

EVAL_RECAP_A="${BASE_DIR}/eval_recap_A.stats.json"
EVAL_RECAP_B="${BASE_DIR}/eval_recap_B.stats.json"

# Logging helper
add_timestamps() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
    done
}
exec > >(add_timestamps | tee -a "${LOG_DIR}/recap_pipeline.log") 2>&1

print_header() {
    echo ""
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

phase_done() { [ -f "$BASE_DIR/.done_${1}" ]; }
mark_done()  { touch "$BASE_DIR/.done_${1}"; }

random_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}

get_ckpt() {
    echo "${1}/checkpoints/checkpoints/last/pretrained_model"
}

# =============================================================================
# Startup info
# =============================================================================
print_header "RECAP RL Pipeline"
echo "  Experiment:    $BASE_DIR"
echo "  Policy A:      $POLICY_A"
echo "  Policy B:      $POLICY_B"
echo "  Train GPUs:    $TRAIN_GPUS ($NUM_TRAIN_GPUS)"
echo "  Collection:    $NUM_TRAIN_GPUS GPUs x $CYCLES_PER_GPU A→B cycles = $((NUM_TRAIN_GPUS * CYCLES_PER_GPU)) episodes/task"
echo "  VF epochs:     $VF_EPOCHS"
echo "  RECAP steps:   $RECAP_STEPS"
echo "  null_prob:     $NULL_PROB  alpha: $ALPHA"
echo ""

# =============================================================================
# Step 1: Parallel alternating A→B collection (all GPUs)
# =============================================================================
print_header "Step 1: Parallel Alternating A→B Collection"
if ! phase_done step1_collect; then
    log "Launching $NUM_TRAIN_GPUS GPU workers x $CYCLES_PER_GPU A→B cycles..."

    COLLECT_PIDS=()
    for gpu_id in $(seq 0 $((NUM_TRAIN_GPUS-1))); do
        _real_gpu=${_GPU_LIST[$gpu_id]}
        log "  GPU ${_real_gpu}: ${CYCLES_PER_GPU} A→B cycles (seed=${gpu_id})..."
        CUDA_VISIBLE_DEVICES=$_real_gpu python "${SCRIPT_DIR}/1_collect_rollouts.py" \
            --policy_A "$POLICY_A" \
            --policy_B "$POLICY_B" \
            --out_A "${BASE_DIR}/rollouts_gpu${gpu_id}_A.npz" \
            --out_B "${BASE_DIR}/rollouts_gpu${gpu_id}_B.npz" \
            --num_cycles "$CYCLES_PER_GPU" \
            --horizon "$HORIZON" \
            --n_action_steps "$N_ACTION_STEPS" \
            --seed "$gpu_id" --headless \
            > "${LOG_DIR}/collect_gpu${gpu_id}.log" 2>&1 &
        COLLECT_PIDS+=($!)
    done

    log "Waiting for ${#COLLECT_PIDS[@]} collection workers..."
    FAILED=0
    for i in "${!COLLECT_PIDS[@]}"; do
        pid=${COLLECT_PIDS[$i]}
        if wait "$pid"; then
            log "  Worker ${i}: done (PID $pid)"
        else
            log "  Worker ${i}: FAILED (PID $pid, exit=$?)"
            FAILED=$((FAILED+1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        log "ERROR: ${FAILED}/${NUM_TRAIN_GPUS} workers failed. Check ${LOG_DIR}/collect_gpu*.log"
        exit 1
    fi

    # Merge per-GPU shards
    log "Merging per-GPU shards..."
    python3 << MERGE_EOF
import numpy as np
from pathlib import Path

base = Path("$BASE_DIR")
num = $NUM_TRAIN_GPUS

for task in ["A", "B"]:
    all_eps = []
    n_suc = 0
    for g in range(num):
        p = base / f"rollouts_gpu{g}_{task}.npz"
        if not p.exists():
            print(f"  WARNING: {p} not found")
            continue
        eps = np.load(str(p), allow_pickle=True)["episodes"]
        suc = sum(1 for e in eps if e.get("success", False))
        print(f"  GPU {g} Task {task}: {len(eps)} eps, {suc} success")
        all_eps.extend(eps)
        n_suc += suc
    out = base / f"rollouts_{task}.npz"
    np.savez_compressed(str(out), episodes=np.array(all_eps, dtype=object))
    print(f"  => Task {task}: {len(all_eps)} total, {n_suc} success "
          f"({n_suc/max(len(all_eps),1)*100:.1f}%) -> {out}")
MERGE_EOF

    mark_done step1_collect
    log "Collection + merge complete"
else
    log "[Skip] Rollout collection already done"
fi

# =============================================================================
# Step 2a: Train value function for Task A
# =============================================================================
print_header "Step 2a: Train Value Function (Task A)"
if ! phase_done step2a_vf_A; then
    log "Training distributional value function for Task A..."
    python "${SCRIPT_DIR}/2_train_value_function.py" \
        --policy "$POLICY_A" \
        --npz_paths "$DEMO_A" "$ROLLOUT_A_NPZ" \
        --out "$VF_A_PT" \
        --epochs "$VF_EPOCHS" \
        --batch_size "$VF_BATCH_SIZE" \
        --lr "$VF_LR" \
        --num_bins "$NUM_BINS" \
        --c_fail "$C_FAIL" \
        --max_ep_len "$MAX_EP_LEN" \
        2>&1 | tee "${LOG_DIR}/vf_A.log"
    mark_done step2a_vf_A
    log "Value function A trained: $VF_A_PT"
else
    log "[Skip] Value function A already trained"
fi

# =============================================================================
# Step 2b: Train value function for Task B
# =============================================================================
print_header "Step 2b: Train Value Function (Task B)"
if ! phase_done step2b_vf_B; then
    log "Training distributional value function for Task B..."
    python "${SCRIPT_DIR}/2_train_value_function.py" \
        --policy "$POLICY_B" \
        --npz_paths "$DEMO_B" "$ROLLOUT_B_NPZ" \
        --out "$VF_B_PT" \
        --epochs "$VF_EPOCHS" \
        --batch_size "$VF_BATCH_SIZE" \
        --lr "$VF_LR" \
        --num_bins "$NUM_BINS" \
        --c_fail "$C_FAIL" \
        --max_ep_len "$MAX_EP_LEN" \
        2>&1 | tee "${LOG_DIR}/vf_B.log"
    mark_done step2b_vf_B
    log "Value function B trained: $VF_B_PT"
else
    log "[Skip] Value function B already trained"
fi

# =============================================================================
# Step 3a: Compute advantages for Task A
# =============================================================================
print_header "Step 3a: Compute Advantages (Task A)"
if ! phase_done step3a_adv_A; then
    log "Computing advantages for Task A..."
    python "${SCRIPT_DIR}/3_compute_advantages.py" \
        --policy "$POLICY_A" \
        --vf_ckpt "$VF_A_PT" \
        --npz_paths "$DEMO_A" "$ROLLOUT_A_NPZ" \
        --out "$ADV_A_NPZ" \
        --stats_out "$ADV_A_STATS" \
        --percentile "$PERCENTILE" \
        --c_fail "$C_FAIL" \
        --max_ep_len "$MAX_EP_LEN" \
        2>&1 | tee "${LOG_DIR}/adv_A.log"
    mark_done step3a_adv_A
    log "Advantages computed for Task A: $ADV_A_NPZ"
else
    log "[Skip] Task A advantages already computed"
fi

# =============================================================================
# Step 3b: Compute advantages for Task B
# =============================================================================
print_header "Step 3b: Compute Advantages (Task B)"
if ! phase_done step3b_adv_B; then
    log "Computing advantages for Task B..."
    python "${SCRIPT_DIR}/3_compute_advantages.py" \
        --policy "$POLICY_B" \
        --vf_ckpt "$VF_B_PT" \
        --npz_paths "$DEMO_B" "$ROLLOUT_B_NPZ" \
        --out "$ADV_B_NPZ" \
        --stats_out "$ADV_B_STATS" \
        --percentile "$PERCENTILE" \
        --c_fail "$C_FAIL" \
        --max_ep_len "$MAX_EP_LEN" \
        2>&1 | tee "${LOG_DIR}/adv_B.log"
    mark_done step3b_adv_B
    log "Advantages computed for Task B: $ADV_B_NPZ"
else
    log "[Skip] Task B advantages already computed"
fi

# =============================================================================
# Step 4a: RECAP fine-tune Policy A
# =============================================================================
print_header "Step 4a: RECAP Fine-tuning (Task A)"
if ! phase_done step4a_recap_A; then
    log "Fine-tuning Policy A with advantage conditioning ($RECAP_STEPS steps)..."
    PORT=$(random_port)
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$NUM_TRAIN_GPUS \
        --master_port=$PORT \
        "${SCRIPT_DIR}/4_retrain_with_recap.py" \
        --npz_path "$ADV_A_NPZ" \
        --policy "$POLICY_A" \
        --out "$RECAP_A_DIR" \
        --steps "$RECAP_STEPS" \
        --batch_size "$RECAP_BATCH_SIZE" \
        --lr "$RECAP_LR" \
        --null_prob "$NULL_PROB" \
        --alpha "$ALPHA" \
        --n_action_steps $N_ACTION_STEPS \
        --noise_scheduler_type DDIM \
        --num_inference_steps 10 \
        --seed "$SEED" \
        ${WANDB_ENABLED:+--wandb --wandb_project "$WANDB_PROJECT"} \
        2>&1 | tee "${LOG_DIR}/recap_A.log"
    mark_done step4a_recap_A
    log "RECAP Policy A trained: $(get_ckpt $RECAP_A_DIR)"
else
    log "[Skip] RECAP Policy A already trained"
fi

# =============================================================================
# Step 4b: RECAP fine-tune Policy B
# =============================================================================
print_header "Step 4b: RECAP Fine-tuning (Task B)"
if ! phase_done step4b_recap_B; then
    log "Fine-tuning Policy B with advantage conditioning ($RECAP_STEPS steps)..."
    PORT=$(random_port)
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$NUM_TRAIN_GPUS \
        --master_port=$PORT \
        "${SCRIPT_DIR}/4_retrain_with_recap.py" \
        --npz_path "$ADV_B_NPZ" \
        --policy "$POLICY_B" \
        --out "$RECAP_B_DIR" \
        --steps "$RECAP_STEPS" \
        --batch_size "$RECAP_BATCH_SIZE" \
        --lr "$RECAP_LR" \
        --null_prob "$NULL_PROB" \
        --alpha "$ALPHA" \
        --n_action_steps $N_ACTION_STEPS \
        --noise_scheduler_type DDIM \
        --num_inference_steps 10 \
        --seed "$SEED" \
        ${WANDB_ENABLED:+--wandb --wandb_project "$WANDB_PROJECT"} \
        2>&1 | tee "${LOG_DIR}/recap_B.log"
    mark_done step4b_recap_B
    log "RECAP Policy B trained: $(get_ckpt $RECAP_B_DIR)"
else
    log "[Skip] RECAP Policy B already trained"
fi

# =============================================================================
# Step 5: Fair evaluation
# =============================================================================
print_header "Step 5: Fair Evaluation"

RECAP_A_CKPT=$(get_ckpt "$RECAP_A_DIR")
RECAP_B_CKPT=$(get_ckpt "$RECAP_B_DIR")

if ! phase_done step5_eval_A; then
    log "Evaluating RECAP Policy A ($EVAL_EPISODES episodes)..."
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python "${SCRIPT_DIR}/5_eval_fair.py" \
        --policy "$RECAP_A_CKPT" \
        --task A \
        --num_episodes "$EVAL_EPISODES" \
        --out "$EVAL_RECAP_A" \
        --horizon "$HORIZON" \
        --n_action_steps "$N_ACTION_STEPS" \
        --seed "$SEED" \
        --headless \
        2>&1 | tee "${LOG_DIR}/eval_recap_A.log"
    mark_done step5_eval_A
else
    log "[Skip] RECAP Policy A evaluation already done"
fi

if ! phase_done step5_eval_B; then
    log "Evaluating RECAP Policy B ($EVAL_EPISODES episodes)..."
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python "${SCRIPT_DIR}/5_eval_fair.py" \
        --policy "$RECAP_B_CKPT" \
        --task B \
        --num_episodes "$EVAL_EPISODES" \
        --out "$EVAL_RECAP_B" \
        --horizon "$HORIZON" \
        --n_action_steps "$N_ACTION_STEPS" \
        --seed "$SEED" \
        --headless \
        2>&1 | tee "${LOG_DIR}/eval_recap_B.log"
    mark_done step5_eval_B
else
    log "[Skip] RECAP Policy B evaluation already done"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "RECAP Pipeline Complete"

python3 << PYSUMMARY
import json
from pathlib import Path

results = {}
for task, path in [("A", "$EVAL_RECAP_A"), ("B", "$EVAL_RECAP_B")]:
    p = Path(path)
    if p.exists():
        d = json.load(open(p))
        results[task] = d
        print(f"  Task {task}: {d['num_success']}/{d['num_total']} = {d['success_rate']*100:.1f}%")
    else:
        print(f"  Task {task}: not found")

# Advantage stats
for task, path in [("A", "$ADV_A_STATS"), ("B", "$ADV_B_STATS")]:
    p = Path(path)
    if p.exists():
        d = json.load(open(p))
        print(f"  Adv {task}: threshold={d['threshold']:.4f}  "
              f"positive_ratio={d['positive_ratio']*100:.1f}%  "
              f"suc_pos={d['success_ep_positive_ratio']*100:.1f}%  "
              f"fail_pos={d['failure_ep_positive_ratio']*100:.1f}%")
PYSUMMARY

echo ""
echo "  Policy A checkpoint: $RECAP_A_CKPT"
echo "  Policy B checkpoint: $RECAP_B_CKPT"
echo "  Stats A: $EVAL_RECAP_A"
echo "  Stats B: $EVAL_RECAP_B"
echo ""
echo "  To run another iteration, set:"
echo "    EXP_NAME=recap_iter2  POLICY_A=$RECAP_A_CKPT  POLICY_B=$RECAP_B_CKPT"
echo ""

touch "$BASE_DIR/.complete"
