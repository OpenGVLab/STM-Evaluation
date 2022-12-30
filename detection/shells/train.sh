#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL=$1
SCALE=$2
PROJECT_NAME="${MODEL}_${SCALE}"

GPUS=8
QUOTA_TYPE="spot"

srun -p ${PARTITION} \
    --job-name="${MODEL}_${SCALE}" \
    --gres=gpu:${GPUS} \
    --quotatype=${QUOTA_TYPE} \
    --async \
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
    sh shells/dist_train.sh \
    "configs/unified_models/unified_${MODEL}/mask_rcnn_${MODEL}_${SCALE}.py" \
    ${GPUS} \
    --work-dir "/mnt/petrelfs/wangweiyun/mmdet_logs/${PROJECT_NAME}" \
    --auto-scale-lr --auto-resume
