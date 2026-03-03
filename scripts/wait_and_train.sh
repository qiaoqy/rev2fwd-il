#!/bin/bash
# Wait for current training jobs to finish, then launch the remaining two trainings
# Current jobs: A_ps5collected_euler (GPU 0,1) and B_euler (GPU 2,3)
# Remaining: A_euler and B_reversed_euler

cd /mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/rev2fwd-il

echo "$(date): Waiting for current training jobs to finish..."
echo "Monitoring GPU 0-3 for training completion..."

# Wait until no 7_train_ditflow processes are running
while true; do
    TRAIN_PROCS=$(ps aux | grep "7_train_ditflow" | grep -v grep | wc -l)
    if [ "$TRAIN_PROCS" -eq 0 ]; then
        echo "$(date): No training processes detected. GPUs should be free."
        break
    fi
    echo "$(date): Still $TRAIN_PROCS training processes running. Checking again in 60s..."
    sleep 60
done

# Double check GPU memory is freed (wait a bit for cleanup)
sleep 15
echo "$(date): Checking GPU availability..."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Find which GPUs are free (less than 5GB used)
FREE_GPUS=()
for gpu_id in 0 1 2 3; do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
    if [ "$MEM_USED" -lt 5000 ]; then
        FREE_GPUS+=($gpu_id)
    fi
done

echo "$(date): Free GPUs: ${FREE_GPUS[@]}"

if [ ${#FREE_GPUS[@]} -lt 4 ]; then
    echo "ERROR: Expected at least 4 free GPUs but only found ${#FREE_GPUS[@]}. Aborting."
    exit 1
fi

# Use first 2 free GPUs for job 1, next 2 for job 2
GPU_SET1="${FREE_GPUS[0]},${FREE_GPUS[1]}"
GPU_SET2="${FREE_GPUS[2]},${FREE_GPUS[3]}"

echo "$(date): Starting training A_euler on GPUs $GPU_SET1..."
CUDA_VISIBLE_DEVICES=$GPU_SET1 torchrun --nproc_per_node=2 --master_port=29521 \
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0226_A_euler \
    --out runs/pickplace_piper_0226_A_euler \
    --save_freq 10000 --lr 5e-4 --batch_size 128 --steps 40000 \
    --wandb --include_gripper --skip_convert &
PID1=$!

echo "$(date): Starting training B_reversed_euler on GPUs $GPU_SET2..."
CUDA_VISIBLE_DEVICES=$GPU_SET2 torchrun --nproc_per_node=2 --master_port=29522 \
    scripts/scripts_piper_local/7_train_ditflow.py \
    --dataset data/pickplace_piper_0226_B_reversed_euler \
    --out runs/pickplace_piper_0226_B_reversed_euler \
    --save_freq 10000 --lr 5e-4 --batch_size 128 --steps 40000 \
    --wandb --include_gripper --skip_convert &
PID2=$!

echo "$(date): Both trainings launched. PIDs: $PID1, $PID2"
echo "Waiting for both to complete..."

wait $PID1
EXIT1=$?
echo "$(date): A_euler training finished with exit code $EXIT1"

wait $PID2
EXIT2=$?
echo "$(date): B_reversed_euler training finished with exit code $EXIT2"

echo "$(date): All trainings complete!"
