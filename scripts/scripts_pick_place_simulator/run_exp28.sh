#!/usr/bin/env bash
# =============================================================================
# Exp28: Multi-object pick & place data collection pipeline
# =============================================================================
# This script collects Task B data for each object type, reverses it to get
# Task A, and runs inspection/visualization.
#
# All outputs go to: data/pick_place_isaac_lab_simulation/exp28/
#
# Usage:
#   conda activate rev2fwd_il
#
#   # Collect a single object type:
#   bash scripts/scripts_pick_place_simulator/run_exp28.sh cylinder
#
#   # Collect all object types:
#   bash scripts/scripts_pick_place_simulator/run_exp28.sh all
#
#   # Dry-run (10 episodes) for testing:
#   bash scripts/scripts_pick_place_simulator/run_exp28.sh cylinder 10
#
#   # Dry-run without headless (with GUI):
#   bash scripts/scripts_pick_place_simulator/run_exp28.sh cylinder 10 gui
# =============================================================================

set -euo pipefail

# ---- Configuration ----
EXP_DIR="data/pick_place_isaac_lab_simulation/exp28"
SCRIPT_DIR="scripts/scripts_pick_place_simulator"
COLLECT_SCRIPT="${SCRIPT_DIR}/1_collect_task_B_multi_obj.py"
REVERSE_SCRIPT="${SCRIPT_DIR}/2_reverse_to_task_A.py"
INSPECT_SCRIPT="${SCRIPT_DIR}/3_inspect_data.py"

# Default parameters
NUM_EPISODES="${2:-100}"
RENDER_MODE="${3:-headless}"  # "headless" or "gui"
HEADLESS_FLAG=""
if [[ "$RENDER_MODE" == "headless" ]]; then
    HEADLESS_FLAG="--headless"
fi

ALL_OBJECTS=(cube cylinder sphere bottle gear)

# ---- Parse object type argument ----
OBJECT_ARG="${1:-}"
if [[ -z "$OBJECT_ARG" ]]; then
    echo "Usage: $0 <object_type|all> [num_episodes] [headless|gui]"
    echo "  object_type: one of {${ALL_OBJECTS[*]}} or 'all'"
    echo "  num_episodes: default 100"
    echo "  headless|gui: default headless"
    exit 1
fi

if [[ "$OBJECT_ARG" == "all" ]]; then
    OBJECTS=("${ALL_OBJECTS[@]}")
else
    OBJECTS=("$OBJECT_ARG")
fi

# ---- Create output directories ----
mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/videos"

echo "========================================================"
echo "Exp28: Multi-object Pick & Place"
echo "  Objects:      ${OBJECTS[*]}"
echo "  Episodes:     ${NUM_EPISODES}"
echo "  Render mode:  ${RENDER_MODE}"
echo "  Output dir:   ${EXP_DIR}"
echo "========================================================"

# ---- Run pipeline for each object type ----
for OBJ in "${OBJECTS[@]}"; do
    echo ""
    echo "========================================================"
    echo "  Processing: ${OBJ}"
    echo "========================================================"

    TASK_B_PATH="${EXP_DIR}/task_B_${OBJ}_${NUM_EPISODES}.npz"
    TASK_A_PATH="${EXP_DIR}/task_A_${OBJ}_${NUM_EPISODES}.npz"
    LOG_PATH="${EXP_DIR}/logs/collect_${OBJ}.log"

    # --- Step 1: Collect Task B data ---
    if [[ -f "$TASK_B_PATH" ]]; then
        echo "[SKIP] Task B data already exists: ${TASK_B_PATH}"
    else
        echo "[1/3] Collecting Task B data for ${OBJ} (${NUM_EPISODES} episodes)..."
        python "${COLLECT_SCRIPT}" \
            --object_type "${OBJ}" \
            --out "${TASK_B_PATH}" \
            --num_episodes "${NUM_EPISODES}" \
            ${HEADLESS_FLAG} \
            2>&1 | tee "${LOG_PATH}"
        echo "[1/3] Done. Output: ${TASK_B_PATH}"
    fi

    # --- Step 2: Reverse to get Task A ---
    if [[ -f "$TASK_A_PATH" ]]; then
        echo "[SKIP] Task A data already exists: ${TASK_A_PATH}"
    else
        if [[ ! -f "$TASK_B_PATH" ]]; then
            echo "[ERROR] Cannot reverse: Task B data missing at ${TASK_B_PATH}"
            continue
        fi
        echo "[2/3] Reversing Task B -> Task A for ${OBJ}..."
        python "${REVERSE_SCRIPT}" \
            --input "${TASK_B_PATH}" \
            --out "${TASK_A_PATH}" \
            --verify \
            2>&1 | tee -a "${LOG_PATH}"
        echo "[2/3] Done. Output: ${TASK_A_PATH}"
    fi

    # --- Step 3: Inspect & visualize (Task B episode 0) ---
    echo "[3/3] Inspecting Task B data for ${OBJ}..."
    python "${INSPECT_SCRIPT}" \
        --dataset "${TASK_B_PATH}" \
        --episode 0 --frame 30 \
        --out_dir "${EXP_DIR}/videos" \
        --name "task_B_${OBJ}" \
        --fps 30 \
        2>&1 | tee -a "${LOG_PATH}"

    # Also inspect Task A if it exists
    if [[ -f "$TASK_A_PATH" ]]; then
        echo "      Inspecting Task A data for ${OBJ}..."
        python "${INSPECT_SCRIPT}" \
            --dataset "${TASK_A_PATH}" \
            --episode 0 --frame 30 \
            --out_dir "${EXP_DIR}/videos" \
            --name "task_A_${OBJ}" \
            --fps 30 \
            2>&1 | tee -a "${LOG_PATH}"
    fi

    echo "[DONE] ${OBJ} pipeline complete."
done

# ---- Summary ----
echo ""
echo "========================================================"
echo "Exp28 pipeline complete!"
echo "========================================================"
echo "Data files:"
for OBJ in "${OBJECTS[@]}"; do
    B="${EXP_DIR}/task_B_${OBJ}_${NUM_EPISODES}.npz"
    A="${EXP_DIR}/task_A_${OBJ}_${NUM_EPISODES}.npz"
    if [[ -f "$B" ]]; then
        B_SIZE=$(du -h "$B" | cut -f1)
        echo "  [B] ${B}  (${B_SIZE})"
    fi
    if [[ -f "$A" ]]; then
        A_SIZE=$(du -h "$A" | cut -f1)
        echo "  [A] ${A}  (${A_SIZE})"
    fi
done
echo ""
echo "Logs:   ${EXP_DIR}/logs/"
echo "Videos: ${EXP_DIR}/videos/"
