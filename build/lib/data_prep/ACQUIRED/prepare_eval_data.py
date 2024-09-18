import os
import json
import random
from tqdm import tqdm

# Load the data
ROOT_DIR = '/home/shreyasjena/BTP/datasets/ACQUIRED'
val = json.load(open(os.path.join(ROOT_DIR, "Dataset/test.json"), 'r'))      # 5275 questions

print(len(val))

idx2opt = {
    0: 'A',
    1: 'B'
}

val_vids = set()
test = []

# iterate over dataframe
for d in tqdm(val):
    vid = d['video_id']
    qtype = d['domain']
    val_vids.add(vid)
    question = d['question']
    ans = d['correct_answer_key']
    video_path = d['video_path']
    question_id = f'{d["video_id"]}_{d["domain"]}'
    options = [d[f'answer{i}'] for i in range(1, len(idx2opt)+1)]
    answer = d[ans]

    # Prepare human query
    query = "<video>\n\n" + "Instruction: Answer the following question by choosing the most appropriate option (out of A, B).\n\nQuestion:\n" + question \
    + "?\n\nOptions:\n" \
    + '\n'.join([f'{idx2opt[i]}. {options[i]}' for i in range(len(idx2opt))]) \
    + "\n\nAnswer: "

    d = {
        "video": str(vid),
        "video_path": video_path,
        "question_id": question_id,
        "type": qtype,
        "question": query,
        "answer": answer
    }

    test.append(d)

with open("outputs/ACQUIRED/val_ACQUIRED.json", 'w') as f:
    json.dump(test, f)

print("Total no. of questions for evaluation : ", len(test))
print("Total no. of videos used : ", len(val_vids))
