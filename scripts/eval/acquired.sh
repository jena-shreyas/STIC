SCRATCH="/home/shreyasjena/BTP"
DATASET="ACQUIRED"
MODEL="LLaVA-NExT-Video"
type="sample"
output_dir="${SCRATCH}/models/STIC/results/eval/${DATASET}/${MODEL}"
pred_path="${SCRATCH}/models/STIC/results/acquired_llavanextvideo_responses.jsonl"
# output_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/gpt"
output_json="${SCRATCH}/models/STIC/results/eval/results_acquired_llavanextvideo_${type}.json"

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
