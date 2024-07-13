import os
import pandas as pd
import json
import random
from tqdm import tqdm

# Load the data
ROOT_DIR = '/home/shreyasjena/BTP/datasets/NExT-QA/data'
train = pd.read_csv(ROOT_DIR + "/nextqa/train.csv")

with open("outputs/data_pref_NExT-QA_sample.json", 'r') as f:
    pref = json.load(f)
    
pref_vids = [int(d['video']) for d in pref]
train_vids = [idx for idx in list(set(train['video']))]
reqd_vids = sorted(list(set(train_vids) - set(pref_vids)))
print("No. of allowed vids : ", len(reqd_vids))

NUM_QUESTIONS = 3000
QUESTIONS_PER_VIDEO = 2
VIDEO_FOLDER = ROOT_DIR + '/videos'

pref_dict = {}
# convert list to dict
for vid_dir in pref:
    vid = vid_dir['video']
    pref_dict[vid] = vid_dir
    
idx2opt = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

sft_data = {}

def bsearch_reqd_ids(vid):
    lo, hi = 0, len(reqd_vids)-1
    while lo<=hi:
        mid = (lo+hi)//2
        if reqd_vids[mid] == vid:
            return mid
        elif reqd_vids[mid] < vid:
            lo = mid+1
        else:
            hi = mid-1
    return None
    
def prepare_conversation(vid, row):
    conversation = []
    
    description_token = "<description>"
    question = row['question']
    ans = row['answer']
    options = [row[f'a{i}'] for i in range(len(idx2opt))]
    answer = row[f'a{ans}']
    
    # Prepare human query
    human_query = "<video>\nVideo description:\n" + description_token + "\n\n" + "Instruction: Answer the following question by choosing the most appropriate option (out of A, B, C, E).\n\nQuestion:\n" + question \
    + "?\n\nOptions:\n" \
    + '\n'.join([f'{idx2opt[i]}. {options[i]}' for i in range(len(idx2opt))]) \
    + "\n\nAnswer: "
    
    human_conv = {
        "from": "human",
        "value": human_query
    }
    
    # Prepare model response
    model_response = {
        "from": "gpt",
        "value": answer
    }
    
    conversation.extend([human_conv, model_response])
    return conversation

idx = 0

# iterate over dataframe
for (_, row) in tqdm(train.iterrows(), total=len(train)):
    vid = row['video']
    if bsearch_reqd_ids(vid) is not None and (str(vid) not in sft_data or len(sft_data[str(vid)]) < QUESTIONS_PER_VIDEO):
        # Add the data to the sft_data
        conversation = prepare_conversation(vid, row)
        video_path = os.path.join(VIDEO_FOLDER, f'{vid}.mp4')
    
        qtype = row['type']
        video_conv = {
            "id": str(idx),
            "type": qtype,
            "video": video_path,
            "conversations": conversation
        }
        if str(vid) not in sft_data:
            sft_data[str(vid)] = []

        sft_data[str(vid)].append(video_conv)
            
        idx+=1
        
        if idx == NUM_QUESTIONS:
            break
        
        
sft_data_list = []
for vid in sft_data:
    sft_data_list.extend(sft_data[vid])
    
with open("outputs/sft_data_desc_ft_NExT-QA.json", 'w') as f:
    json.dump(sft_data_list, f, indent=4)
