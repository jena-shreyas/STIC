import json

with open("outputs/data_pref_NExT-QA.json", 'r') as f:
    data = json.load(f)
    
data_sample = data[:2000]

with open("outputs/data_pref_NExT-QA_sample.json", 'w') as f:
    json.dump(data_sample, f, indent=4)