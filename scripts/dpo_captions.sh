export OMP_NUM_THREADS=16
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# data args

SCRATCH="/home/shreyasj/BTP"
TEAM_NAME="Video-LMs"
PROJECT_NAME="LLaVA-NeXT-Video"
DATA_PATH="data/data_pref_merged.jsonl"
IMAGE_FOLDER="${SCRATCH}/datasets/COCO/pref_images"
VIDEO_FOLDER="${SCRATCH}/datasets/WebVid/videos"

# model args

PROMPT_VERSION="v1"
MODEL_NAME="lmms-lab/LLaVA-NeXT-Video-7B"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"

# training args

LR=5e-6
BATCH_SIZE=1
GRAD_ACCUM=16        # change this
NUM_EPOCHS=1
NUM_NODES=1
NUM_GPUS=2
BITS=8
MASTER_PORT=29500

# wandb login
export WANDB_API_KEY=`cat .wandb_api_key`
wandb login $WANDB_API_KEY
export WANDB_NAME=$PROJECT_NAME--$MODEL_NAME
export WANDB_ENTITY=$TEAM_NAME
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_MODE=online
MID_RUN_NAME="llavanextvideo7b_dpo_finetune_mixed"
wandb online


#torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
ACCELERATE_CPU_AFFINITY=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node ${NUM_GPUS} --nnodes ${NUM_NODES} --master_port ${MASTER_PORT} \
    LLaVA-NeXT/llava/train/train_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --version $PROMPT_VERSION \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0 \
    --data_path=$DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type "spatial_pool" \
    --mm_spatial_pool_out_channels 1024 \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --bits $BITS \
    --run_name $MID_RUN_NAME \
    --output_dir "checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --save_full_model True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --verbose_logging True \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 \
    --frames_upbound 30