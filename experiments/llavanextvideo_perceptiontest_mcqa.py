import os
import sys
from os.path import join as osj
import pandas as pd
import json
import shutil
import random
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import numpy as np
import av
from tqdm import tqdm

# SEED=42
# random.seed(SEED)

# SAMPLES_PER_DATASET = 200

ROOT="/home/shreyasjena/BTP/datasets/PerceptionTest"
PART_IDX=int(sys.argv[1])
DEVICE_MAP=sys.argv[2]

def create_random_string(length: int = 3):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# model loading
# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", 
                                                               torch_dtype=torch.float16,       # 
                                                               attn_implementation="flash_attention_2",
                                                               device_map=DEVICE_MAP
                                                               )
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")


# ACQUIRED eval

VID_DIR="videos"
DATA_DIR="data"
SPLIT="validation"

with open(osj(ROOT,DATA_DIR,f"all_valid_p{PART_IDX}.json")) as f:
    data = json.load(f)

# print("ACQUIRED | Total test samples : ", len(acq_data))

# acq_sample = random.sample(acq_data, SAMPLES_PER_DATASET)
area_stats = {}
reasoning_stats = {}
total_qns = 0

for vid, d in data.items():
    if d["mc_question"]:
        for qn_dict in d["mc_question"]:
            total_qns += 1
            if qn_dict["area"] not in area_stats:
                area_stats[qn_dict["area"]] = 0
            
            area_stats[qn_dict["area"]] += 1

            if qn_dict["reasoning"] not in reasoning_stats:
                reasoning_stats[qn_dict["reasoning"]] = 0

            reasoning_stats[qn_dict["reasoning"]] += 1

print("Total questions : ", total_qns)
for area in area_stats:
    print(f"{area} : {area_stats[area]}")

for reasoning in reasoning_stats:
    print(f"{reasoning} : {reasoning_stats[reasoning]}")    

OUTPUT_DIR='results/inference/PerceptionTest'
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = osj(OUTPUT_DIR, f"perceptiontest_mcqa_llavanextvideo_responses_p{PART_IDX}.jsonl")

completed_vids = set()
total_vids = list(data.keys())
vids = list()

if os.path.exists(output_path):
    # load jsonl file
    f = open(output_path, "r")
    for line in f.readlines():
        d = json.loads(line)
        vid = d["video"]
        completed_vids.add(vid)
    completed_vids = list(completed_vids)
    last_vid = completed_vids[-1]
    last_vid_idx = total_vids.index(last_vid)
    vids = total_vids[last_vid_idx+1:]
    f = open(output_path, "a")
else:
    f = open(output_path, "w")
    vids = total_vids

for vid in tqdm(vids, total=len(vids)):
    d = data[vid]
    if(d["mc_question"]):
        video_path = osj(ROOT,VID_DIR,SPLIT,vid+".mp4")
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            num_frames = 16
            indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
            clip = read_video_pyav(container, indices)
            
            pre_instruction = "Imagine you are an expert at answering questions based on videos and you are being asked to answer the following question based on the video clip provided. The question is as follows:"
            instruction = "Based on the video clip provided, answer the question by choosing the most appropriate option from the following:"

            for qn_dict in d["mc_question"]:
                area = qn_dict["area"]
                reasoning = qn_dict["reasoning"]
                qs = qn_dict["question"]
                num_options = len(qn_dict["options"])
                options = '\n'.join([f"Option {i+1}: {qn_dict['options'][i]}" for i in range(num_options)])

                prompt = f"{pre_instruction}\n\n{qs}\n\n{instruction}\n\n{options}"
                # print(prompt)

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "video"},
                            ],
                    },
                ]

                processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs_video = processor(text=processed_prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
                output = model.generate(**inputs_video, max_new_tokens=1024, do_sample=False)
                output_conv = processor.decode(output[0][2:], skip_special_tokens=True)

                # parse output_conv so that the text after "ASSISTANT:" is returned
                response = output_conv.split("ASSISTANT:")[1].strip()
                correct_option = int(qn_dict['answer_id'])
                correct_answer = qn_dict['options'][correct_option]

                response_dict = {
                    "id": f"{vid}_{area}_{reasoning}_{create_random_string(3)}",
                    "video": vid,
                    "type": f"{area}_{reasoning}",
                    "area": area,
                    "reasoning": reasoning,
                    "question": prompt,
                    "answer": correct_answer,
                    "pred": response
                }

                f.write(json.dumps(response_dict) + "\n")

        except Exception as e:
            print(f"Error in processing file {video_path} : ", e)

f.close()
