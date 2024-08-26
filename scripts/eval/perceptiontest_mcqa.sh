SCRATCH="/home/shreyasjena/BTP"
DATASET="PerceptionTest"
MODEL="LLaVA-NExT-Video"
TYPE="mcqa"

output_dir="${SCRATCH}/models/STIC/results/eval/${DATASET}/${MODEL}"
pred_path="${SCRATCH}/models/STIC/results/inference/${DATASET}/perceptiontest_${TYPE}_llavanextvideo_responses.jsonl"
output_json="${SCRATCH}/models/STIC/results/eval/results_perceptiontest_${TYPE}_llavanextvideo.json"

api_type="azure"
api_base="https://gpt35newdec23.openai.azure.com/"
api_version="2023-09-15-preview"
api_key="45a56bedd7d54f30ab4a622cdce4803d"

num_tasks=2


python3 VideoLLaVA/videollava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_type ${api_type} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --api_version ${api_version} \
    --num_tasks ${num_tasks}
