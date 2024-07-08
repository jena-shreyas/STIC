import pandas as pd
import json
from tqdm import tqdm

# Load the data
train = pd.read_csv("data/NExT-QA/nextqa/train.csv")

with open("outputs/video_description.json", 'r') as f:
    desc = json.load(f)
    
NUM_QUESTIONS = 6000
QUESTIONS_PER_VIDEO = NUM_QUESTIONS // len(desc)        # 3

desc_dict = {}
# convert list to dict
for vid_dir in desc:
    vid = vid_dir['video'].split('/')[-1].split('.')[0]
    desc_dict[vid] = vid_dir
    
# Get the common video ids
reqd_vids = sorted([int(vid) for vid in list(desc_dict.keys())])
assert len(reqd_vids) == 2000

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
    
    description = desc_dict[str(vid)]['description']
    question = row['question']
    ans = row['answer']
    options = [row[f'a{i}'] for i in range(len(idx2opt))]
    answer = row[f'a{ans}']
    
    # Prepare human query
    human_query = "<video>\nVideo description:\n" + description + "\n\n" + "Instruction: Answer the following question by choosing the most appropriate option (out of A, B, C, E).\n\nQuestion:\n" + question \
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
        video_path = desc_dict[str(vid)]['video']
    
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
    
with open("outputs/sft_data_NExT-QA.json", 'w') as f:
    json.dump(sft_data_list, f, indent=4)
