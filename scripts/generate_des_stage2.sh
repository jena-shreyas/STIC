SCRATCH="/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb"
VIDEO_FOLDER="${SCRATCH}/shreyas/STIC/data/NExT-QA/videos"
CKPT_FOLDER="${SCRATCH}/shreyas/STIC/checkpoints/videollava_stic_stage1_2024-06-28_19-38-56"
OUTPUT_DIR="${SCRATCH}/shreyas/STIC/outputs/"

CUDA_VISIBLE_DEVICES=1 python ./stic/generate_des_stage2.py \
    --model-path LanguageBind/Video-LLaVA-7B \
    --video-dir $VIDEO_FOLDER \
    --save-dir $OUTPUT_DIR/video_description.jsonl \
    --adapter-path $CKPT_FOLDER