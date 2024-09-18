import os
import pandas as pd
import json
import random
from tqdm import tqdm

# Load the data
ROOT_DIR = '/home/shreyasjena/BTP/datasets/NExT-QA/data'
val = pd.read_csv(ROOT_DIR + "/nextqa/val.csv")

idx2opt = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

test = []
val_vids = set()

# iterate over dataframe
for (_, row) in tqdm(val.iterrows(), total=len(val)):
    vid = row['video']
    val_vids.add(vid)
    question = row['question']
    ans = row['answer']
    qid = row['qid']
    qtype = row['type']
    question_id = f'{vid}_{qid}'
    options = [row[f'a{i}'] for i in range(len(idx2opt))]
    answer = row[f'a{ans}']

    # Prepare human query
    query = "<video>\n\n" + "Instruction: Answer the following question by choosing the most appropriate option (out of A, B, C, D, E).\n\nQuestion:\n" + question \
    + "?\n\nOptions:\n" \
    + '\n'.join([f'{idx2opt[i]}. {options[i]}' for i in range(len(idx2opt))]) \
    + "\n\nAnswer: "

    d = {
        "video": str(vid),
        "question_id": question_id,
        "type": qtype,
        "question": query,
        "answer": answer
    }

    test.append(d)

with open("outputs/val_NExT-QA.json", 'w') as f:
    json.dump(test, f)

print("Total no. of questions for evaluation : ", len(test))
print("Total no. of videos used : ", len(val_vids))
