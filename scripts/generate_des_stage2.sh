SCRATCH="/home/shreyasjena/BTP"
VIDEO_FOLDER="${SCRATCH}/datasets/NExT-QA/data/videos"
CKPT_FOLDER="${SCRATCH}/models/STIC/checkpoints/videollava_stic_stage1_2024-07-11_20-55-54"
OUTPUT_DIR="${SCRATCH}/models/STIC/outputs/"

CUDA_VISIBLE_DEVICES=0 python ./stic/generate_des_stage2.py \
    --model-path LanguageBind/Video-LLaVA-7B \
    --video-dir $VIDEO_FOLDER \
    --save-dir $OUTPUT_DIR/video_description.jsonl \
    --adapter-path $CKPT_FOLDER