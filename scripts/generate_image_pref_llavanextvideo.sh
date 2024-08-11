DATASET='COCO'
MODEL_ID="llava-hf/LLaVA-NeXT-Video-7B-hf"
IMAGE_DIR="pref_images"
SCRATCH=/home/shreyasjena/BTP

python stic/generate_image_pref_llavanextvideo.py \
    --model_path $MODEL_ID \
    --img_dir $SCRATCH/datasets/$DATASET/$IMAGE_DIR \
    --corrupt_dir $SCRATCH/models/STIC/pert_imgs/$DATASET \
    --output_dir  $SCRATCH/models/STIC/outputs/$DATASET \
    --output_name "data_pref_${DATASET}_rem_images_p1"
