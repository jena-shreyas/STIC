import json
import pandas as pd
from tqdm import tqdm

with open("outputs/data_pref_NExT-QA.json", 'r') as f:
    data = json.load(f)
    
data_dict = {video_dir['video']:video_dir for video_dir in data}
train = pd.read_csv("data/NExT-QA/nextqa/train.csv")
train_ids = [str(idx) for idx in list(set(train['video']))]
sample_vids = train_ids[:2000]
data_sample = {}

for id in tqdm(sample_vids):
    if id in data_dict:
        data_sample[id] = data_dict[id]
        
data_sample = list(data_sample.values())

with open("outputs/data_pref_NExT-QA_sample.json", 'w') as f:
    json.dump(data_sample, f, indent=4)
    