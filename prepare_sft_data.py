import pandas as pd
import json

# Load the data
train = pd.read_csv("data/NExT-QA/nextqa/train.csv")
train_vids = list(train['video'])

with open("outputs/video_description_dict.json", 'r') as f:
    desc = json.load(f)
    
desc_vids = [int(id) for id in list(desc.keys())]
common_vids = sorted(list(set(train_vids).intersection(set(desc_vids))))

idx2opt = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

sft_data = {
    'id': [],
    'conversations': [],
    'video': []
}

def bsearch_common_ids(vid):
    lo, hi = 0, len(common_vids)-1
    while lo<=hi:
        mid = (lo+hi)//2
        if common_vids[mid] == vid:
            return mid
        elif common_vids[mid] < vid:
            lo = mid+1
        else:
            hi = mid-1
    return None
    
def prepare_conversation(vid, row):
    conversation = []
    
    description = desc[str(vid)]['description']
    question = row['question']
    ans = row['answer']
    qid = row['qid']
    qtype = row['type']
    a0 = row['a0']
    a1 = row['a1']
    a2 = row['a2']
    a3 = row['a3']
    a4 = row['a4']
    answer = row[f'a{ans}']
    
    # Prepare human query
    human_query = "Video description:\n" + description + "\n\n" + sft_data[i]["conversations"][0]["value"]
    human_conv = {
        "from": "human",
        "value": human_query
    }
    
    # Prepare model response
    model_response = {
        "from": "gpt",
        "value": f"{idx2opt[ans]}. {answer}"
    }
    
    conversation.extend([human_conv, model_response])
    return conversation

# iterate over dataframe
for _, row in train.iterrows():
    vid = row['video']
    if bsearch_common_ids(vid) is not None:
        # Add the data to the sft_data
        conversation = prepare_conversation(vid, row)
    
        sft_data['video'].append(str(vid))