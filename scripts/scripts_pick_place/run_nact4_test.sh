#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate rev2fwd_il

export PYTHONUNBUFFERED=1

CKPT_A="runs/exp5_PP_A_temp/checkpoints/checkpoints/last/pretrained_model"
CKPT_B="runs/exp5_PP_B_temp/checkpoints/checkpoints/last/pretrained_model"
OUT_DIR="data/pick_place_isaac_lab_simulation/exp5"
DIST=0.05
NACT=8
TAG="nact${NACT}_d${DIST//.}"

echo "Launching 4 parallel Task A evals (n_action_steps=$NACT, dist=$DIST, 100 total)..."

CUDA_VISIBLE_DEVICES=1 python scripts/scripts_pick_place/10_eval_independent.py \
  --policy_A "$CKPT_A" --policy_B "$CKPT_B" \
  --out "$OUT_DIR/${TAG}_gpu1.stats.json" \
  --num_episodes 25 --horizon 400 --distance_threshold $DIST \
  --n_action_steps $NACT --goal_xy 0.5 0.0 --tasks A --seed 0 --headless \
  > "$OUT_DIR/${TAG}_gpu1.log" 2>&1 &
PID1=$!
echo "GPU 1: PID $PID1 (25 episodes, seed 0)"

CUDA_VISIBLE_DEVICES=2 python scripts/scripts_pick_place/10_eval_independent.py \
  --policy_A "$CKPT_A" --policy_B "$CKPT_B" \
  --out "$OUT_DIR/${TAG}_gpu2.stats.json" \
  --num_episodes 25 --horizon 400 --distance_threshold $DIST \
  --n_action_steps $NACT --goal_xy 0.5 0.0 --tasks A --seed 100 --headless \
  > "$OUT_DIR/${TAG}_gpu2.log" 2>&1 &
PID2=$!
echo "GPU 2: PID $PID2 (25 episodes, seed 100)"

CUDA_VISIBLE_DEVICES=3 python scripts/scripts_pick_place/10_eval_independent.py \
  --policy_A "$CKPT_A" --policy_B "$CKPT_B" \
  --out "$OUT_DIR/${TAG}_gpu3.stats.json" \
  --num_episodes 25 --horizon 400 --distance_threshold $DIST \
  --n_action_steps $NACT --goal_xy 0.5 0.0 --tasks A --seed 200 --headless \
  > "$OUT_DIR/${TAG}_gpu3.log" 2>&1 &
PID3=$!
echo "GPU 3: PID $PID3 (25 episodes, seed 200)"

CUDA_VISIBLE_DEVICES=4 python scripts/scripts_pick_place/10_eval_independent.py \
  --policy_A "$CKPT_A" --policy_B "$CKPT_B" \
  --out "$OUT_DIR/${TAG}_gpu4.stats.json" \
  --num_episodes 25 --horizon 400 --distance_threshold $DIST \
  --n_action_steps $NACT --goal_xy 0.5 0.0 --tasks A --seed 300 --headless \
  > "$OUT_DIR/${TAG}_gpu4.log" 2>&1 &
PID4=$!
echo "GPU 4: PID $PID4 (25 episodes, seed 300)"

echo "All launched. Waiting for completion..."
set +e
wait $PID1; RC1=$?
wait $PID2; RC2=$?
wait $PID3; RC3=$?
wait $PID4; RC4=$?
set -e

echo ""
echo "Exit codes: GPU1=$RC1  GPU2=$RC2  GPU3=$RC3  GPU4=$RC4"

if [ $RC1 -ne 0 ] || [ $RC2 -ne 0 ] || [ $RC3 -ne 0 ] || [ $RC4 -ne 0 ]; then
    echo "ERROR: Some processes failed. Check logs:"
    [ $RC1 -ne 0 ] && echo "  $OUT_DIR/${TAG}_gpu1.log"
    [ $RC2 -ne 0 ] && echo "  $OUT_DIR/${TAG}_gpu2.log"
    [ $RC3 -ne 0 ] && echo "  $OUT_DIR/${TAG}_gpu3.log"
    [ $RC4 -ne 0 ] && echo "  $OUT_DIR/${TAG}_gpu4.log"
    exit 1
fi

echo "All processes completed. Aggregating results..."

python3 -c "
import json, glob

files = sorted(glob.glob('$OUT_DIR/${TAG}_gpu*.stats.json'))
total_success = 0
total_episodes = 0
all_details = []

for f in files:
    with open(f) as fh:
        data = json.load(fh)
    s = data['summary']
    total_success += s['task_A_success_count']
    total_episodes += s['task_A_total_episodes']
    all_details.extend(data.get('episodes_A', []))
    print(f'  {f}: {s[\"task_A_success_count\"]}/{s[\"task_A_total_episodes\"]} = {s[\"task_A_success_rate\"]*100:.1f}%')

rate = total_success / total_episodes if total_episodes > 0 else 0
print(f'')
print(f'=== AGGREGATE ===')
print(f'Task A (n_action_steps=$NACT, dist=$DIST): {total_success}/{total_episodes} = {rate*100:.1f}%')

# Save merged stats
steps = [d['success_step'] for d in all_details if d.get('success') and d.get('success_step')]
merged = {
    'experiment': 'n_action_steps_${NACT}_dist_${DIST}_ablation',
    'config': {'n_action_steps': $NACT, 'distance_threshold': $DIST, 'horizon': 400, 'num_episodes': total_episodes},
    'summary': {
        'task_A_success_count': total_success,
        'task_A_total_episodes': total_episodes,
        'task_A_success_rate': rate,
        'avg_success_step_A': sum(steps)/len(steps) if steps else None,
    },
    'episodes_A': all_details,
}
out = '$OUT_DIR/iter10_${TAG}_merged.stats.json'
with open(out, 'w') as fh:
    json.dump(merged, fh, indent=2)
print(f'Saved merged stats to: {out}')
"
