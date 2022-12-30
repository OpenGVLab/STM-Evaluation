#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC

MODEL=$1
CKPT="/mnt/petrelfs/wangweiyun/m2odel_ckpt/${MODEL}.pth"


DESC="invariance_eval" 

# key hyperparameters
TOTAL_BATCH_SIZE="256"
VARIANCE_TYPE="translation"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_${VARIANCE_TYPE}_1k_${DESC}"

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE="auto"

CPUS_PER_TASK=${CPUS_PER_TASK:-12}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    --async \
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
    python -u invariance_eval_all.py \
    --model ${MODEL} \
    --resume ${CKPT} \
    --variance_type ${VARIANCE_TYPE} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
    --input_size 224 \
    --crop_pct 0.875 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp true \
    --output_dir "/mnt/petrelfs/wangweiyun/model_evaluation/invariance/${PROJECT_NAME}"
