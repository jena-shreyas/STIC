import json
import os

f = open("results/inference/PerceptionTest/perceptiontest_mcqa_llavanextvideo_responses_p4.jsonl", 'r')
fnew = open("results/inference/PerceptionTest/perceptiontest_mcqa_llavanextvideo_responses_p4_new.jsonl", 'w')

for line in f.readlines():
    data = json.loads(line)
    video = data['video']
    area = data['area']
    reasoning = data['reasoning']
    id = f"{video}_{area}_{reasoning}"
    data["id"] = id
    fnew.write(json.dumps(data) + "\n")

f.close()
fnew.close()

    