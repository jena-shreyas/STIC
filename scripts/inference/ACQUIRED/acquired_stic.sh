SCRATCH="/home/shreyasjena/BTP"
model_path="LanguageBind/Video-LLaVA-7B"
DATASET="ACQUIRED"
MODEL="Video-LLaVA"
peft_path="${SCRATCH}/models/STIC/checkpoints/videollava_stic_stage2_2024-07-12_13-19-49"
cache_dir="./cache_dir"
video_dir="${SCRATCH}/datasets/ACQUIRED/acquired_dataset"
gt_file_question="./outputs/ACQUIRED/val_ACQUIRED.json"
output_dir="./results/inference/${DATASET}/${MODEL}"
output_name="acquired_videollava_stic"
CHUNKS=1
IDX=0


# --load_peft ${peft_path} \

CUDA_VISIBLE_DEVICES=0 python3 VideoLLaVA/videollava/eval/video/run_inference_acquired.py \
    --model_path ${model_path} \
    --load_peft ${peft_path} \
    --cache_dir ${cache_dir} \
    --video_dir ${video_dir} \
    --gt_file_question ${gt_file_question} \
    --output_dir ${output_dir} \
    --output_name ${output_name} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX
    