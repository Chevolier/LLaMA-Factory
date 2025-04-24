#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_DISABLED="true"
DIR=`pwd`

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

#The related information about multi-nodes cluster.
MASTER_HOST=localhost # $SM_MASTER
MASTER_ADDR=127.0.0.1 # $SM_MASTER_ADDR
MASTER_PORT="23456"
NNODES=1 # "$NODE_NUMBER"
NODE_RANK=0 # "$NODE_INDEX"
GPUS_PER_NODE=8 # "$SM_NUM_GPUS"

echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "job_id: ${job_id}"

# Number of GPUs per GPU worker
# GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
MODEL="/home/ec2-user/SageMaker/efs/Models/DeepSeek-R1-Distill-Qwen-7B"
DATASET_DIR="/home/ec2-user/SageMaker/efs/Projects/slm-assist-llm/LIMO/eval/data/s1"
# OUTPUT_DIR="/home/ec2-user/SageMaker/efs/Projects/virtual-boyfriend/checkpoints"
OUTPUT_DIR="/home/ec2-user/SageMaker/efs/Projects/slm-assist-llm/checkpoints/DS-Qwen-7B"

# s5cmd sync $MODEL_S3_PATH $MODEL

#!/bin/bash
# deepspeed --num_gpus $GPUS_PER_NODE src/train.py \
torchrun --nproc_per_node="${GPUS_PER_NODE}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset s1 \
    --dataset_dir $DATASET_DIR \
    --template llama3 \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 32768 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy epoch \
    --eval_strategy no \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --warmup_ratio 0.1 \
    --val_size 0.0 \
    --plot_loss \
    --bf16 \
    --save_only_model True \
#     --lora_target q_proj,v_proj
    
if [ $? -eq 1 ]; then
    echo "Training script error, please check CloudWatch logs"
    exit 1
fi