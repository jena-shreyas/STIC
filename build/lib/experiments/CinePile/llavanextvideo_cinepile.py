import os
import sys
from os.path import join as osj
from os.path import dirname as osd
import pandas as pd
import json
import shutil
import random
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import numpy as np
import av
from tqdm import tqdm

from datasets import load_dataset
from data_utils import process_video, get_prompt

ROOT=osj(osd(osd(os.getcwd())), "datasets/CinePile")
DEVICE_MAP=sys.argv[1]
# PART_IDX=int(sys.argv[2])


def create_random_string(length: int = 3):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

def parse_vid_from_yt_link(data):
    yt_link = data['yt_clip_link']
    data['vid'] = yt_link.split('=')[-1]
    return data

# def read_video_pyav(container, indices):
#     '''
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# model loading
# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", 
                                                               torch_dtype=torch.float16,       # 
                                                               attn_implementation="flash_attention_2",
                                                               device_map=DEVICE_MAP
                                                               )
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# Eval

VID_DIR="yt_videos"
SPLIT="test"

cinepile = load_dataset("tomg-group-umd/cinepile")
data = cinepile[SPLIT]

print("CinePile | Total test samples : ", len(data))

OUTPUT_DIR='results/inference/CinePile'
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = osj(OUTPUT_DIR, f"cinepile_mcqa_llavanextvideo_responses.jsonl")

map_vid_data = data.map(parse_vid_from_yt_link)
total_vids = set(map_vid_data['vid'])

# prev_vid = ""
# for i in range(len(map_vid_data)):
#     cur_vid = map_vid_data['vid']
#     if cur_vid != prev_vid:
#         total_vids.append(cur_vid)
#     prev_vid = cur_vid

print("Total no. of unique videos : ", len(total_vids))
start_idx = 0

# if os.path.exists(output_path):
#     # load jsonl file
#     f = open(output_path, "r")
#     start_idx = len(f.readlines())
#     f = open(output_path, "a")
# else:
f = open(output_path, "w")
    
for i in tqdm(range(start_idx, len(data))):
    d = data[i]
    vid = d['yt_clip_link'].split('=')[-1]
    video_path = osj(ROOT,VID_DIR,SPLIT,vid+".mp4")
    category = d['question_category']

    try:
        # container = av.open(video_path)
        # total_frames = container.streams.video[0].frames
        # num_frames = 16
        # indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
        # clip = read_video_pyav(container, indices)
        
        prompt = get_prompt(d)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                    ],
            },
        ]

        clip = process_video(data, osj(ROOT, VID_DIR), SPLIT, 10, False, i)

        processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs_video = processor(text=processed_prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)
        output = model.generate(**inputs_video, max_new_tokens=1024, do_sample=False)
        output_conv = processor.decode(output[0][2:], skip_special_tokens=True)

        # parse output_conv so that the text after "ASSISTANT:" is returned
        response = output_conv.split("ASSISTANT:")[1].strip()
        correct_option = int(d['answer_key_position'])
        correct_answer = d['answer_key'] 

        response_dict = {
            "id": f"{vid}_{create_random_string(3)}",
            "video": vid,
            "type": category,
            "question": prompt,
            "correct_option": correct_option,
            "answer": correct_answer,
            "pred": response
        }

        f.write(json.dumps(response_dict) + "\n")

    except Exception as e:
        print(f"Error in processing file {video_path} : ", e)

f.close()
