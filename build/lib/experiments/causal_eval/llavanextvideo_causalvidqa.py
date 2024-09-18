import os
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
import traceback

SEED=42
random.seed(SEED)

ROOT="/home/shreyasjena/BTP/datasets"

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
                                                               device_map="auto"
                                                               )
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")


# Causal-VidQA eval

CVQA_ROOT=osj(ROOT,"Causal-VidQA/data/orig_data")
CVQA_VID="sample_videos"

with open(osj(CVQA_ROOT,"test_sample.json"), 'r') as f:
    cvqa_data = json.load(f)

print("Causal-VidQA | Total test samples : ", len(cvqa_data))

f = open("results/cvqa_llavanextvideo_responses.jsonl", "w")

for d in tqdm(cvqa_data, total=len(cvqa_data)):
    video_path = osj(CVQA_ROOT,CVQA_VID,str(d["video_id"])+".mp4")
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        num_frames = 16
        indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
        clip = read_video_pyav(container, indices)
        
        pre_instruction = "Imagine you are an expert at answering questions based on videos and you are being asked to answer the following question based on the video clip provided. The question is as follows:"
        qs = d["question"]
        instruction = "Based on the video clip provided, answer the question by choosing the most appropriate option from the following:"
        options = '\n'.join([f"Option {i+1}: {d['options'][i]}" for i in range(5)])

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
        correct_answer = d[f"answer"]

        response_dict = {
            "id": d["qid"],
            "type": d["qtype"],
            "question": prompt,
            "answer": correct_answer,
            "pred": response
        }

        f.write(json.dumps(response_dict) + "\n")

    except Exception as e:
        print(f"Error in processing file {video_path} : ", traceback.format_exc())

f.close()
