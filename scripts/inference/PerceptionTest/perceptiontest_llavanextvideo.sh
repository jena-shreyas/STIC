#!/bin/sh
PARTS=4
DEVICE_MAP="cuda:0"

# python data_prep/split_data_files.py $PARTS

for n in 4; do
    python experiments/llavanextvideo_perceptiontest_mcqa.py $n $DEVICE_MAP > \
        logs/inference/inference_perceptiontest_mcqa_llavanextvideo_p$n.log 2>&1 &
done
