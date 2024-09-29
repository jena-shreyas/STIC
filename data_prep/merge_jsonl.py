import json
import os

dir_path = "results/inference/PerceptionTest"
output_filename = "perceptiontest_mcqa_llavanextvideo_responses.jsonl"

data = list()
for filename in os.listdir(dir_path):
    if filename.endswith(".jsonl"):
        f = open("{}/{}".format(dir_path, filename), "r")
        for line in f.readlines():
            data.append(json.loads(line))
        f.close()

f = open(os.path.join(dir_path, output_filename), "w")
for d in data:
    f.write(json.dumps(d) + "\n")

f = open(os.path.join(dir_path, output_filename), "w")
for d in data:
    f.write(json.dumps(d) + "\n")

f.close()

