import json
import os

dataset = "COCO"
idx=1
with open(f"outputs/{dataset}/data_pref_{dataset}_p{idx}.json", 'r') as f:
    data = json.load(f)

f = open(f"outputs/{dataset}/data_pref_{dataset}_p{idx}.jsonl", 'w')

for d in data:
    f.write(json.dumps(d) + '\n')

f.close()