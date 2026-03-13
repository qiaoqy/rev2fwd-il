#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Recommended Commands (tmux background execution)
# =============================================================================
# 1) Create and enter tmux session:
#    tmux new -s exp9
#
# 2) Run exp9 pipeline in tmux (recommended GPU split: 0,1 for collect/fair;
#    0,1,2 for train A; 3,4,5 for train B):
#    cd /mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il
#    conda activate rev2fwd_il
#    BASE_DIR=data/pick_place_isaac_lab_simulation/exp9 \
#    COLLECT_GPU=0 FAIR_TEST_GPU=1 \
#    TRAIN_GPUS_A=0,1,2 TRAIN_GPUS_B=3,4,5 \
#    bash scripts/scripts_pick_place/run_exp9_fixed_start.sh
#
# 3) Detach tmux and keep running in background:
#    Ctrl+b, then press d
#
# 4) Re-attach later:
#    tmux attach -t exp9
#
# 5) Monitor logs without attaching:
#    tail -f data/pick_place_isaac_lab_simulation/exp9/logs/iter1_collect.log
#    tail -f data/pick_place_isaac_lab_simulation/exp9/logs/iter1_train_A.log
#
# 6) Stop the pipeline if needed (inside tmux):
#    Ctrl+C
#
# 7) Kill tmux session after completion:
#    tmux kill-session -t exp9
# =============================================================================

# exp9 fixed-start ablation pipeline
# - Iter0: collect B20 -> reverse to A -> train A/B to 200 epochs
# - Iter1..10: parallel collect+fair-test -> prepare finetune -> parallel resume train
# - All artifacts are written under data/pick_place_isaac_lab_simulation/exp9/

# =========================
# Environment
# =========================
eval "$(conda shell.bash hook)"
conda activate rev2fwd_il

# =========================
# Config (override by env)
# =========================
BASE_DIR="${BASE_DIR:-data/pick_place_isaac_lab_simulation/exp9}"
LOG_DIR="${BASE_DIR}/logs"

FIXED_START_X="${FIXED_START_X:-0.45}"
FIXED_START_Y="${FIXED_START_Y:-0.15}"
GOAL_X="${GOAL_X:-0.5}"
GOAL_Y="${GOAL_Y:-0.0}"
DISTANCE_THRESHOLD="${DISTANCE_THRESHOLD:-0.03}"

NUM_ITER0_EPISODES="${NUM_ITER0_EPISODES:-20}"
HORIZON="${HORIZON:-400}"
ITER_ROUNDS="${ITER_ROUNDS:-10}"
NUM_CYCLES="${NUM_CYCLES:-50}"
NUM_FAIR_TEST_EPISODES="${NUM_FAIR_TEST_EPISODES:-100}"

BATCH_SIZE="${BATCH_SIZE:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
ITER0_EPOCHS="${ITER0_EPOCHS:-200}"
STEPS_PER_ITER="${STEPS_PER_ITER:-5000}"

# GPU placement (default uses 0..5)
COLLECT_GPU="${COLLECT_GPU:-0}"
FAIR_TEST_GPU="${FAIR_TEST_GPU:-1}"
TRAIN_GPUS_A="${TRAIN_GPUS_A:-0,1,2}"
TRAIN_GPUS_B="${TRAIN_GPUS_B:-3,4,5}"

HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

# NCCL robustness
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

# =========================
# Paths
# =========================
ITER0_B_NPZ="${BASE_DIR}/iter0_taskB_20.npz"
ITER0_A_NPZ="${BASE_DIR}/iter0_taskA_from_reverse.npz"

ITER0_A_OUT="${BASE_DIR}/weights/exp9_PP_A_iter0"
ITER0_B_OUT="${BASE_DIR}/weights/exp9_PP_B_iter0"

ITER0_A_LEROBOT="${BASE_DIR}/lerobot/exp9_A_iter0"
ITER0_B_LEROBOT="${BASE_DIR}/lerobot/exp9_B_iter0"

WORK_A_TEMP="${BASE_DIR}/work/PP_A_temp"
WORK_B_TEMP="${BASE_DIR}/work/PP_B_temp"
WORK_A_LAST="${BASE_DIR}/work/PP_A_last"
WORK_B_LAST="${BASE_DIR}/work/PP_B_last"

RECORD_JSON="${BASE_DIR}/record.json"
CONFIG_JSON="${BASE_DIR}/config.json"

# =========================
# Helpers
# =========================
mkdir -p "${BASE_DIR}" "${LOG_DIR}" "${BASE_DIR}/weights" "${BASE_DIR}/lerobot" "${BASE_DIR}/work"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

gpu_count() {
  awk -F',' '{print NF}' <<< "$1"
}

random_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

get_ckpt() {
  echo "$1/checkpoints/checkpoints/last/pretrained_model"
}

get_step() {
  local step_file="$1/checkpoints/checkpoints/last/training_state/training_step.json"
  if [[ -f "$step_file" ]]; then
    python3 - <<PY
import json
p = "$step_file"
with open(p) as f:
    print(int(json.load(f).get("step", 0)))
PY
  else
    echo 0
  fi
}

calc_steps_from_npz() {
  local npz="$1"
  local batch="$2"
  local epochs="$3"
  python3 - <<PY
import math
import numpy as np
npz = "$npz"
batch = int("$batch")
epochs = int("$epochs")
with np.load(npz, allow_pickle=True) as data:
    eps = list(data["episodes"])
frames = 0
for ep in eps:
    if "action" in ep:
        frames += int(len(ep["action"]))
steps = math.ceil(frames * epochs / max(batch, 1))
print(max(steps, 1))
PY
}

append_record() {
  local iter="$1"
  local collect_stats="$2"
  local fair_stats="$3"
  local step_a="$4"
  local step_b="$5"
  python3 - <<PY
import json
from pathlib import Path
record_path = Path("$RECORD_JSON")
if record_path.exists():
    rec = json.loads(record_path.read_text())
else:
    rec = {"description": "exp9 fixed-start pipeline", "iterations": []}

entry = None
for e in rec["iterations"]:
    if e.get("iteration") == int("$iter"):
        entry = e
        break
if entry is None:
    entry = {"iteration": int("$iter")}
    rec["iterations"].append(entry)

entry["checkpoint_info"] = {"policy_A_step": int("$step_a"), "policy_B_step": int("$step_b")}

cpath = Path("$collect_stats")
fpath = Path("$fair_stats")

if cpath.exists():
    c = json.loads(cpath.read_text())
    entry["collection_metrics"] = c.get("summary", {})
if fpath.exists():
    f = json.loads(fpath.read_text())
    entry["fair_test_metrics"] = f.get("summary", {})

rec["iterations"].sort(key=lambda x: x.get("iteration", 0))
record_path.write_text(json.dumps(rec, indent=2))
print(f"record updated: iter={int('$iter')}")
PY
}

run_train_from_npz() {
  local gpus="$1"
  local dataset="$2"
  local out_dir="$3"
  local lerobot_dir="$4"
  local steps="$5"
  local log_file="$6"

  local nproc
  nproc="$(gpu_count "$gpus")"

  if [[ "$nproc" -gt 1 ]]; then
    local port
    port="$(random_port)"
    CUDA_VISIBLE_DEVICES="$gpus" torchrun --nproc_per_node="$nproc" --master_port="$port" \
      scripts/scripts_pick_place/4_train_diffusion.py \
      --dataset "$dataset" \
      --out "$out_dir" \
      --lerobot_dataset_dir "$lerobot_dir" \
      --steps "$steps" \
      --batch_size "$BATCH_SIZE" \
      --n_action_steps "$N_ACTION_STEPS" \
      --include_obj_pose --include_gripper \
      > "$log_file" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpus" python scripts/scripts_pick_place/4_train_diffusion.py \
      --dataset "$dataset" \
      --out "$out_dir" \
      --lerobot_dataset_dir "$lerobot_dir" \
      --steps "$steps" \
      --batch_size "$BATCH_SIZE" \
      --n_action_steps "$N_ACTION_STEPS" \
      --include_obj_pose --include_gripper \
      > "$log_file" 2>&1
  fi
}

run_resume_train() {
  local gpus="$1"
  local last_dir="$2"
  local target_steps="$3"
  local save_freq="$4"
  local log_file="$5"

  local sample_weights="${last_dir}/lerobot_dataset/meta/sampling_weights.json"
  local extra_weights=()
  if [[ -f "$sample_weights" ]]; then
    extra_weights=(--sample_weights "$sample_weights")
  fi

  local nproc
  nproc="$(gpu_count "$gpus")"

  if [[ "$nproc" -gt 1 ]]; then
    local port
    port="$(random_port)"
    CUDA_VISIBLE_DEVICES="$gpus" torchrun --nproc_per_node="$nproc" --master_port="$port" \
      scripts/scripts_pick_place/4_train_diffusion.py \
      --dataset dummy.npz \
      --lerobot_dataset_dir "${last_dir}/lerobot_dataset" \
      --out "$last_dir" \
      --steps "$target_steps" \
      --batch_size "$BATCH_SIZE" \
      --n_action_steps "$N_ACTION_STEPS" \
      --save_freq "$save_freq" \
      --skip_convert --resume \
      --include_obj_pose --include_gripper \
      "${extra_weights[@]}" \
      > "$log_file" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpus" python scripts/scripts_pick_place/4_train_diffusion.py \
      --dataset dummy.npz \
      --lerobot_dataset_dir "${last_dir}/lerobot_dataset" \
      --out "$last_dir" \
      --steps "$target_steps" \
      --batch_size "$BATCH_SIZE" \
      --n_action_steps "$N_ACTION_STEPS" \
      --save_freq "$save_freq" \
      --skip_convert --resume \
      --include_obj_pose --include_gripper \
      "${extra_weights[@]}" \
      > "$log_file" 2>&1
  fi
}

# =========================
# Preflight
# =========================
log "Writing config to ${CONFIG_JSON}"
python3 - <<PY
import json, datetime
cfg = {
  "created_at": datetime.datetime.now().isoformat(),
  "base_dir": "$BASE_DIR",
  "fixed_start_xy": [float("$FIXED_START_X"), float("$FIXED_START_Y")],
  "goal_xy": [float("$GOAL_X"), float("$GOAL_Y")],
  "distance_threshold": float("$DISTANCE_THRESHOLD"),
  "num_iter0_episodes": int("$NUM_ITER0_EPISODES"),
  "iter_rounds": int("$ITER_ROUNDS"),
  "num_cycles": int("$NUM_CYCLES"),
  "num_fair_test_episodes": int("$NUM_FAIR_TEST_EPISODES"),
  "horizon": int("$HORIZON"),
  "batch_size": int("$BATCH_SIZE"),
  "n_action_steps": int("$N_ACTION_STEPS"),
  "iter0_epochs": int("$ITER0_EPOCHS"),
  "steps_per_iter": int("$STEPS_PER_ITER"),
  "gpus": {
    "collect": "$COLLECT_GPU",
    "fair_test": "$FAIR_TEST_GPU",
    "train_A": "$TRAIN_GPUS_A",
    "train_B": "$TRAIN_GPUS_B",
  },
}
with open("$CONFIG_JSON", "w") as f:
    json.dump(cfg, f, indent=2)
PY

if [[ ! -f "$RECORD_JSON" ]]; then
  printf '{"description":"exp9 fixed-start pipeline","iterations":[]}' > "$RECORD_JSON"
fi

# =========================
# Phase A: Iter0 data build
# =========================
if [[ ! -f "$ITER0_B_NPZ" ]]; then
  log "Iter0: collect Task B (${NUM_ITER0_EPISODES} successful episodes)"
  CUDA_VISIBLE_DEVICES="$COLLECT_GPU" python scripts/scripts_pick_place/1_collect_data_pick_place.py \
    --num_episodes "$NUM_ITER0_EPISODES" \
    --horizon "$HORIZON" \
    --out "$ITER0_B_NPZ" \
    --fixed_start_xy "$FIXED_START_X" "$FIXED_START_Y" \
    $HEADLESS_FLAG \
    > "$LOG_DIR/iter0_collect_B.log" 2>&1
else
  log "Iter0 Task B dataset exists, skip: $ITER0_B_NPZ"
fi

if [[ ! -f "$ITER0_A_NPZ" ]]; then
  log "Iter0: reverse Task B -> Task A"
  python scripts/scripts_pick_place/3_make_forward_data.py \
    --input "$ITER0_B_NPZ" \
    --out "$ITER0_A_NPZ" \
    --success_only 1 \
    > "$LOG_DIR/iter0_make_forward.log" 2>&1
else
  log "Iter0 Task A reverse dataset exists, skip: $ITER0_A_NPZ"
fi

# =========================
# Phase B: Iter0 training
# =========================
if [[ ! -d "$(get_ckpt "$ITER0_A_OUT")" || ! -d "$(get_ckpt "$ITER0_B_OUT")" ]]; then
  A_STEPS_FOR_200_EPOCH="$(calc_steps_from_npz "$ITER0_A_NPZ" "$BATCH_SIZE" "$ITER0_EPOCHS")"
  B_STEPS_FOR_200_EPOCH="$(calc_steps_from_npz "$ITER0_B_NPZ" "$BATCH_SIZE" "$ITER0_EPOCHS")"

  log "Iter0 train steps: A=${A_STEPS_FOR_200_EPOCH}, B=${B_STEPS_FOR_200_EPOCH}"
  log "Iter0: parallel training Policy A/B"

  run_train_from_npz "$TRAIN_GPUS_A" "$ITER0_A_NPZ" "$ITER0_A_OUT" "$ITER0_A_LEROBOT" "$A_STEPS_FOR_200_EPOCH" "$LOG_DIR/iter0_train_A.log" &
  PID_A=$!
  run_train_from_npz "$TRAIN_GPUS_B" "$ITER0_B_NPZ" "$ITER0_B_OUT" "$ITER0_B_LEROBOT" "$B_STEPS_FOR_200_EPOCH" "$LOG_DIR/iter0_train_B.log" &
  PID_B=$!

  wait "$PID_A"
  wait "$PID_B"
  log "Iter0 A/B training completed"
else
  log "Iter0 checkpoints already exist, skip training"
fi

# Initialize working temps from iter0 once
if [[ ! -d "$WORK_A_TEMP" ]]; then
  log "Initialize work temp from iter0 A"
  cp -r "$ITER0_A_OUT" "$WORK_A_TEMP"
fi
if [[ ! -d "$WORK_B_TEMP" ]]; then
  log "Initialize work temp from iter0 B"
  cp -r "$ITER0_B_OUT" "$WORK_B_TEMP"
fi

# =========================
# Phase C: Iterative loop
# =========================
for iter in $(seq 1 "$ITER_ROUNDS"); do
  log "================ Iteration ${iter}/${ITER_ROUNDS} ================"

  CKPT_A="$(get_ckpt "$WORK_A_TEMP")"
  CKPT_B="$(get_ckpt "$WORK_B_TEMP")"
  STEP_A="$(get_step "$WORK_A_TEMP")"
  STEP_B="$(get_step "$WORK_B_TEMP")"

  ROLLOUT_A="${BASE_DIR}/iter${iter}_collect_A.npz"
  ROLLOUT_B="${BASE_DIR}/iter${iter}_collect_B.npz"
  COLLECT_STATS="${BASE_DIR}/iter${iter}_collect_A.stats.json"
  FAIR_STATS="${BASE_DIR}/iter${iter}_fair_test.stats.json"

  # 1) Parallel collect + fair-test
  log "Iter${iter}: launch collect (GPU ${COLLECT_GPU}) and fair-test (GPU ${FAIR_TEST_GPU})"
  CUDA_VISIBLE_DEVICES="$COLLECT_GPU" python scripts/scripts_pick_place/9_eval_with_recovery.py \
    --policy_A "$CKPT_A" \
    --policy_B "$CKPT_B" \
    --out_A "$ROLLOUT_A" \
    --out_B "$ROLLOUT_B" \
    --num_cycles "$NUM_CYCLES" \
    --horizon "$HORIZON" \
    --distance_threshold "$DISTANCE_THRESHOLD" \
    --n_action_steps "$N_ACTION_STEPS" \
    --goal_xy "$GOAL_X" "$GOAL_Y" \
    --fixed_start_xy "$FIXED_START_X" "$FIXED_START_Y" \
    $HEADLESS_FLAG \
    > "$LOG_DIR/iter${iter}_collect.log" 2>&1 &
  PID_COLLECT=$!

  CUDA_VISIBLE_DEVICES="$FAIR_TEST_GPU" python scripts/scripts_pick_place/10_eval_independent.py \
    --policy_A "$CKPT_A" \
    --policy_B "$CKPT_B" \
    --out "$FAIR_STATS" \
    --num_episodes "$NUM_FAIR_TEST_EPISODES" \
    --horizon "$HORIZON" \
    --distance_threshold "$DISTANCE_THRESHOLD" \
    --n_action_steps "$N_ACTION_STEPS" \
    --goal_xy "$GOAL_X" "$GOAL_Y" \
    --fixed_start_xy "$FIXED_START_X" "$FIXED_START_Y" \
    $HEADLESS_FLAG \
    > "$LOG_DIR/iter${iter}_fair_test.log" 2>&1 &
  PID_FAIR=$!

  wait "$PID_COLLECT"
  wait "$PID_FAIR"

  # 2) record
  append_record "$iter" "$COLLECT_STATS" "$FAIR_STATS" "$STEP_A" "$STEP_B"

  # 3) prepare finetune data A/B in parallel
  rm -rf "$WORK_A_LAST" "$WORK_B_LAST"
  log "Iter${iter}: prepare finetune datasets"
  python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_lerobot "$WORK_A_TEMP/lerobot_dataset" \
    --rollout_data "$ROLLOUT_A" \
    --checkpoint "$CKPT_A" \
    --out "$WORK_A_LAST" \
    --prepare_only --include_obj_pose --include_gripper --n_action_steps "$N_ACTION_STEPS" \
    > "$LOG_DIR/iter${iter}_prepare_A.log" 2>&1 &
  PID_PREP_A=$!

  python scripts/scripts_pick_place/7_finetune_with_rollout.py \
    --original_lerobot "$WORK_B_TEMP/lerobot_dataset" \
    --rollout_data "$ROLLOUT_B" \
    --checkpoint "$CKPT_B" \
    --out "$WORK_B_LAST" \
    --prepare_only --include_obj_pose --include_gripper --n_action_steps "$N_ACTION_STEPS" \
    > "$LOG_DIR/iter${iter}_prepare_B.log" 2>&1 &
  PID_PREP_B=$!

  wait "$PID_PREP_A"
  wait "$PID_PREP_B"

  # 4) parallel resume train A/B
  CUR_STEP_A="$(get_step "$WORK_A_LAST")"
  CUR_STEP_B="$(get_step "$WORK_B_LAST")"
  TARGET_A=$((CUR_STEP_A + STEPS_PER_ITER))
  TARGET_B=$((CUR_STEP_B + STEPS_PER_ITER))

  log "Iter${iter}: parallel resume train A(${CUR_STEP_A}->${TARGET_A}) B(${CUR_STEP_B}->${TARGET_B})"
  run_resume_train "$TRAIN_GPUS_A" "$WORK_A_LAST" "$TARGET_A" "$STEPS_PER_ITER" "$LOG_DIR/iter${iter}_train_A.log" &
  PID_TRAIN_A=$!
  run_resume_train "$TRAIN_GPUS_B" "$WORK_B_LAST" "$TARGET_B" "$STEPS_PER_ITER" "$LOG_DIR/iter${iter}_train_B.log" &
  PID_TRAIN_B=$!

  wait "$PID_TRAIN_A"
  wait "$PID_TRAIN_B"

  # 5) rotate and save checkpoint snapshot
  rm -rf "$WORK_A_TEMP" "$WORK_B_TEMP"
  mv "$WORK_A_LAST" "$WORK_A_TEMP"
  mv "$WORK_B_LAST" "$WORK_B_TEMP"

  CKPT_A_NEW="$(get_ckpt "$WORK_A_TEMP")"
  CKPT_B_NEW="$(get_ckpt "$WORK_B_TEMP")"
  rm -rf "${BASE_DIR}/weights/iter${iter}_ckpt_A" "${BASE_DIR}/weights/iter${iter}_ckpt_B"
  cp -r "$CKPT_A_NEW" "${BASE_DIR}/weights/iter${iter}_ckpt_A"
  cp -r "$CKPT_B_NEW" "${BASE_DIR}/weights/iter${iter}_ckpt_B"

  # 6) plotting (best effort)
  python scripts/scripts_pick_place/plot_success_rate.py \
    --record "$RECORD_JSON" \
    --out "${BASE_DIR}/fair_test_curve.png" \
    --metrics_key fair_test_metrics \
    > "$LOG_DIR/iter${iter}_plot_fair.log" 2>&1 || true

  python scripts/scripts_pick_place/plot_success_rate.py \
    --record "$RECORD_JSON" \
    --out "${BASE_DIR}/collection_curve.png" \
    --metrics_key collection_metrics \
    > "$LOG_DIR/iter${iter}_plot_collect.log" 2>&1 || true

  log "Iter${iter} done"
done

log "Pipeline complete"
log "Record: ${RECORD_JSON}"
log "Latest A ckpt: $(get_ckpt "$WORK_A_TEMP")"
log "Latest B ckpt: $(get_ckpt "$WORK_B_TEMP")"
