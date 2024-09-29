#!/bin/sh
PARTS=4
DEVICE_MAP="cuda:0"
CKPT_PATH="checkpoints/llavanextvideo7b_dpo_finetune_mixed/checkpoint-421"

# python data_prep/split_data_files.py $PARTS

# $(seq 1 $((PARTS)))
for n in $(seq 1 2); do
    echo "Running part $n"
    nohup python experiments/PerceptionTest/llavanextvideo_perceptiontest_mcqa.py $n $DEVICE_MAP $CKPT_PATH > \
        logs/inference/inference_perceptiontest_mcqa_llavanextvideo_ft_p$n.log 2>&1 &
done
