import json

des = []

def path2id(path):
    return path.split('/')[-1].split('.')[0]

with open(f"outputs/video_description.json", 'r') as json_file:
    des = json.load(json_file)

with open("outputs/sft_data_desc_ft_NExT-QA.json", "r") as f:
    sft_data = json.load(f)

# convert list to dict
des_dict = {path2id(d['video']):d for d in des}

for i in range(len(sft_data)):
    vid = path2id(sft_data[i]["video"])
    cap = des_dict[vid]["description"]

    if "<description>" in sft_data[i]["conversations"][0]["value"]:
        sft_data[i]["conversations"][0]["value"] = sft_data[i]["conversations"][0]["value"].replace("<description>", cap)
    
print(sft_data[0])
with open("outputs/sft_data_desc_ft_NExT-QA_new.json", "w") as f:
    json.dump(sft_data, f, indent=4)
