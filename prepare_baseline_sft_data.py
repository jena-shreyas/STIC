import re
import json
from tqdm import tqdm

with open("outputs/sft_data_desc_ft_NExT-QA.json", 'r') as f:
    data = json.load(f)

baseline_data = []

for d in tqdm(data):
    instr = d['conversations'][0]['value']
    # Replace a given regex string in instruction with another string
    instr = re.sub(r'Video description:.*Instruction', r'Instruction', instr)
    pattern = r"Video description:\n.*\n\nInstruction"
    replacement = r"Instruction"
    assert re.search(pattern, instr) is not None, print(instr)
    d['conversations'][0]['value'] = re.sub(pattern, replacement, instr)
    baseline_data.append(d)

with open("outputs/sft_data_desc_ft_NExT-QA_baseline.json", 'w') as f:
    json.dump(baseline_data, f, indent=4)


