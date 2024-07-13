#!/bin/zsh
DATASET='NExT-QA'
SCRATCH=/home/shreyasjena/BTP
VIDEO_FOLDER="${SCRATCH}/datasets/NExT-QA/data/videos"
CKPT_FOLDER="${SCRATCH}/models/STIC/checkpoints/videollava_stic_stage1_2024-07-11_20-55-54"
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${SCRATCH}/models/STIC/checkpoints/videollava_stic_stage2_${TIME}"


# export HF_HOME="/data1/yihedeng"
deepspeed --include=localhost:0 Video-LLaVA/videollava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path LanguageBind/Video-LLaVA-7B \
    --load_peft $CKPT_FOLDER \
    --version v1 \
    --data_path outputs/sft_data_desc_ft_NExT-QA.json \
    --video_folder $VIDEO_FOLDER \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --tf32 True \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

    # True
    # True
    # $CKPT_FOLDER
    # --tf32 False \
