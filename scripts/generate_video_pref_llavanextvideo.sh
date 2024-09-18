DATASET='finevideo'
MODEL_ID="llava-hf/LLaVA-NeXT-Video-7B-hf"
VIDEO_PERT_PROB=0.3
VIDEO_DIR="data"
SCRATCH=/scratch/svani/data

python stic/generate_video_pref_llavanextvideo.py \
    --model_path $MODEL_ID \
    --video_dir $SCRATCH/$DATASET/$VIDEO_DIR \
    --corrupt_dir $SCRATCH/$DATASET/perturbed_videos \
    --output_dir  $SCRATCH/$DATASET/ \
    --output_name "data_pref_${DATASET}" \
    --video_pert_prob $VIDEO_PERT_PROB
