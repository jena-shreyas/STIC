import json

with open("results/nextqa_videollava_stic_baseline.jsonl", 'r') as f:
    res = list(f)

val = json.load(open("outputs/val_NExT-QA.json", "r"))
qid2type = {d["question_id"]: d["type"] for d in val}

data_full = []

for res_str in res:
    data = json.loads(res_str)
    data['type'] = qid2type[data['id']]
    data_full.append(data)

with open("results/nextqa_videollava_stic_baseline_new.jsonl", 'w') as f:
    for data in data_full:
        f.write(json.dumps(data) + "\n")
