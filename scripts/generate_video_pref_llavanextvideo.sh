DATASET='WebVid'
MODEL_ID="llava-hf/LLaVA-NeXT-Video-7B-hf"
VIDEO_PERT_PROB=0.3
VIDEO_DIR="rem_videos"
SCRATCH=/home/shreyasjena/BTP/

python stic/generate_video_pref_llavanextvideo.py \
    --model_path $MODEL_ID \
    --video_dir $SCRATCH/datasets/$DATASET/$VIDEO_DIR \
    --corrupt_dir $SCRATCH/models/STIC/pert_vids/$DATASET \
    --output_dir  $SCRATCH/models/STIC/outputs/$DATASET \
    --output_name "data_pref_${DATASET}_${VIDEO_DIR}" \
    --video_pert_prob $VIDEO_PERT_PROB
