import json

with open('outputs/data_pref_NExT-QA_0.json', 'r') as f:
    data0 = json.load(f)
    
with open('outputs/data_pref_NExT-QA_1.json', 'r') as f:
    data1 = json.load(f)
    
data = data0 + data1

with open('outputs/data_pref_NExT-QA.json', 'w') as f:
    json.dump(data, f)