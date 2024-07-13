# conda env 
# export HF_HOME="/data1/yihedeng"

SCRATCH="/home/shreyasjena/BTP"
VIDEO_FOLDER="${SCRATCH}/datasets/NExT-QA/data/videos"
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="${SCRATCH}/models/STIC/checkpoints/videollava_stic_stage1_${TIME}"

deepspeed --master_port=25641 --include=localhost:0 videollava/train/train_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path LanguageBind/Video-LLaVA-7B \
    --version v1 \
    --data_path ./outputs/data_pref_NExT-QA_sample.json \
    --video_folder $VIDEO_FOLDER \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb \
    --model_max_length 1090 \

        # --gradient_checkpointing True \
        # --bits 8 \
