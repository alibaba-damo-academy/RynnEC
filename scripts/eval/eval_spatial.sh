#!/bin/bash
export PYTHONWARNINGS="ignore"

MODEL_PATH=${1:-"Alibaba-DAMO-Academy/RynnEC-2B"}
QUESTION_FILE=${2:-"data/RynnECBench/spatial_cognition.json"}
NPROC_PER_NODE=4

echo "=================================================="
echo "Starting distributed evaluation with torchrun..."
echo "Total Processes: ${NPROC_PER_NODE}"
echo "Model Path: ${MODEL_PATH}"
echo "Question File: ${QUESTION_FILE}"
echo "=================================================="

DATA_ROOT=data/RynnECBench
SAVE_DIR=evaluation_results

torchrun  --nproc_per_node $NPROC_PER_NODE \
    --master_port=16666 \
    -m evaluation.eval_cognition \
    --model_path ${MODEL_PATH} \
    --video_folder ${DATA_ROOT} \
    --question_file ${QUESTION_FILE} \
    --output_file ""${SAVE_DIR}/RynnEC-2B/$(basename "$QUESTION_FILE" .json).json"" \
    --task_type spatial