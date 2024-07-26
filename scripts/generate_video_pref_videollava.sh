DATASET='NExT-QA'
CUDA_IDX=$1

SCRATCH=/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb
export TMPDIR=$SCRATCH

cd $SCRATCH/shreyas/STIC/stic

CUDA_VISIBLE_DEVICES=$CUDA_IDX python generate_video_pref.py \
    --model_path LanguageBind/Video-LLaVA-7B \
    --cache_dir $HF_HOME \
    --video_dir $SCRATCH/shreyas/STIC/data/$DATASET/videos \
    --corrupt_dir $SCRATCH/shreyas/STIC/pert_vids/ \
    --output_dir  $SCRATCH/shreyas/STIC/outputs \
    --output_name "data_pref_${DATASET}_${CUDA_IDX}" \
    --device_id $CUDA_IDX
