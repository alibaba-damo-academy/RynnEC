#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=256
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS

# Log Arguments
export TRANSFORMERS_OFFLINE=1
RUN_NAME=rynn_stage3
DATA_DIR=./data
CKPT_DIR=./checkpoints

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    rynnec/train.py \
    --deepspeed scripts/zero1.json \
    --model_type rynnec_qwen2 \
    --model_path ./checkpoints/rynnec_stage2 \
    --vision_encoder DAMO-NLP-SG/VL3-SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path stage3.json \
    --data_folder ${DATA_DIR} \
    --dataset_cache_dir ./.cache \
    --group_by_modality_length True \
    --has_mask False \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio qwen2vl \
    --spatial_merge_size 2 \
    --mm_max_length 8192 \
    --use_token_compression True \
    --fps 1 \
    --max_frames 80 \
    --lora_enable False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${CKPT_DIR}/stage3_newpath_split_add_vl3_data_a800 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --llm_lr 4e-5 \
    --mm_projector_lr 1e-5 \
    --region_encoder_lr 1e-5 \
    --vision_encoder_lr 0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --save_safetensors False \
    --run_name $RUN_NAME 2>&1 | tee -a logs/REGION_${RUN_NAME}_${RANK}.log

