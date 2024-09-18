import os
import shutil
import json
import random
from tqdm import tqdm

random.seed(42)

SAMPLES_PER_DATASET = 800

ROOT="/home/shreyasjena/BTP/datasets/Causal-VidQA/data/orig_data"
VIDEO_DIR="dataset"
OUTPUT_VIDEO_DIR="sample_videos"
QA_DIR="QA"

categories = ["descriptive", "explanatory", "predictive", "counterfactual"]
sample_videos = random.sample(os.listdir(os.path.join(ROOT, VIDEO_DIR)), SAMPLES_PER_DATASET // len(categories))

os.makedirs(os.path.join(ROOT, OUTPUT_VIDEO_DIR), exist_ok=True)

cvqa_data = []

for video_name in tqdm(sample_videos):
    video_path = os.path.join(ROOT, VIDEO_DIR, video_name, video_name + ".mp4")
    shutil.copy(video_path, os.path.join(ROOT, OUTPUT_VIDEO_DIR, video_name + ".mp4"))
    question_path = os.path.join(ROOT, QA_DIR, video_name, 'text.json')
    answer_path = os.path.join(ROOT, QA_DIR, video_name, 'answer.json')

    with open(question_path) as f:
        questions = json.load(f)

    with open(answer_path) as f:
        answers = json.load(f)

    for cat in categories:
        d = questions[cat]
        question = d["question"]
        options = d["answer"]
        ans = int(answers[cat]["answer"])
        answer = options[ans]

        d_ = {
            "video_id": video_name,
            "qid": f"{video_name}_{cat}",
            "qtype": cat,
            "question": question,
            "options": options,
            "answer": answer,
        }

        cvqa_data.append(d_)

random.shuffle(cvqa_data)  # To keep questions from same videos away

with open(os.path.join(ROOT, "test_sample.json"), "w") as f:
    json.dump(cvqa_data, f)
