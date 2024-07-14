SCRATCH="/home/shreyasjena/BTP"
model_path="LanguageBind/Video-LLaVA-7B"
peft_path="${SCRATCH}/models/STIC/checkpoints/videollava_stic_stage2_2024-07-12_13-19-49"
cache_dir="./cache_dir"
video_dir="${SCRATCH}/datasets/NExT-QA/data/videos"
gt_file_question="./outputs/val_NExT-QA.json"
output_dir="./results/inference"
output_name="nextqa_videollava_stic_baseline"
CHUNKS=1
IDX=0


# --load_peft ${peft_path} \

CUDA_VISIBLE_DEVICES=0 python3 VideoLLaVA/videollava/eval/video/run_inference_nextqa.py \
    --model_path ${model_path} \
    --load_peft ${peft_path} \
    --cache_dir ${cache_dir} \
    --video_dir ${video_dir} \
    --gt_file_question ${gt_file_question} \
    --output_dir ${output_dir} \
    --output_name ${output_name} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX
    