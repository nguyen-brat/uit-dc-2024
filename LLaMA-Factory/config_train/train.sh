export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
DIR=`pwd`

NPROC_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DS_CONFIG_PATH="examples/deepspeed/ds_z2_offload_config.json"
OUTPUT_PATH="sarcasm_model_trained"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "


torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --do_eval \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --dataset sarcasm_detection_ds \
    --val_size 0.05 \
    -load_best_model_at_end \
    --metric_for_best_model "eval_loss" \
    --dataloader_num_workers 36 \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 6 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 10000 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16 \
    --export_hub_model_id "nguyen-brat/Qwen2-vl-sarcasm" \
    --use_liger_kernel \
    --hf_hub_token "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" \
    --hub_private_repo \
    --gradient_checkpointing \
    --freeze_vision_tower \
    --report_to "wandb" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_safetensors \
    --save_total_limit 5 \
    --resume_from_checkpoint "sarcasm_model_trained/checkpoint-600" 
    
# --neftune_noise_alpha 5
# --optim adafactor / paged_lion_32bit / paged_lion_8bit / adamw_8bit / paged_adamw_32bit