#!/bin/sh
PARTS=4

python data_prep/split_data_files.py $PARTS

for n in $(seq 1 $PARTS); do
    python experiments/llavanextvideo_perceptiontest_mcqa.py $n > \
        logs/inference/inference_perceptiontest_mcqa_llavanextvideo_p$n.log 2>&1 &
done