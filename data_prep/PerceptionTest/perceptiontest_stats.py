import json

data_path = "/home/shreyasjena/BTP/datasets/PerceptionTest/data/all_valid.json"

with open(data_path, 'r') as f:
    data = json.load(f)

area_stats = {}
reasoning_stats = {}
total_qns = 0

for vid, d in data.items():
    if d["mc_question"]:
        for qn_dict in d["mc_question"]:
            total_qns += 1
            if qn_dict["area"] not in area_stats:
                area_stats[qn_dict["area"]] = 0
            
            area_stats[qn_dict["area"]] += 1

            if qn_dict["reasoning"] not in reasoning_stats:
                reasoning_stats[qn_dict["reasoning"]] = 0

            reasoning_stats[qn_dict["reasoning"]] += 1

print("Total questions : ", total_qns)
for area in area_stats:
    print(f"{area} : {area_stats[area]}")

for reasoning in reasoning_stats:
    print(f"{reasoning} : {reasoning_stats[reasoning]}") 