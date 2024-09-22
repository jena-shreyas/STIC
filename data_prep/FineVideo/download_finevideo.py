import os
from datasets import load_dataset   
import json
import numpy as np

cache_dir = "/home/shreyasj/BTP/datasets/FineVideo"
os.makedirs(cache_dir, exist_ok=True)
dataset = load_dataset("HuggingFaceFV/finevideo", split="train", cache_dir=cache_dir)

# dataset = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True, cache_dir=cache_dir)
# sample = next(iter(dataset))

# print(type(sample))
# print(sample.keys())        # mp4, json
# array = np.frombuffer(sample['mp4'], dtype=np.uint8)
# print(array.shape)
# # print(array)

# output_dir = "data_prep/FineVideo/"
# with open(output_dir + "finevideo_sample.json", 'w') as f:
#     json.dump(sample['json'], f)

# with open(output_dir + 'sample.mp4', 'wb') as video_file:
#     video_file.write(sample['mp4'])
