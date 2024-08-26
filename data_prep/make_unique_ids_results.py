import random
import json
import os

def create_random_string(length: int = 5):
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))

data_path = "results/inference/PerceptionTest/perceptiontest_mcqa_llavanextvideo_responses_p4.jsonl"
f = open(data_path, "r")

fnew = open("results/inference/PerceptionTest/perceptiontest_mcqa_llavanextvideo_responses_p4_new.jsonl", "w")

for line in f.readlines():
    d = json.loads(line)
    area = d["area"]
    reasoning = d["reasoning"]
    vid = d["video"]

    d_ = {
        "id": f"{vid}_{area}_{reasoning}_{create_random_string(3)}",
        "video": vid,
        "type": f"{area}_{reasoning}",
        "area": area,
        "reasoning": reasoning,
        "question": d["question"],
        "answer": d["answer"],
        "pred": d["pred"]
    }

    fnew.write(json.dumps(d_) + "\n")

f.close()
fnew.close()