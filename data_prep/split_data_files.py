import json
import os
import sys

DATA_ROOT='/home/shreyasj/BTP/datasets/PerceptionTest'
with open(os.path.join(DATA_ROOT, 'data/all_valid.json'), 'r') as f:
    data = json.load(f)

PARTS=int(sys.argv[1])
total_vids = list(data.keys())
total_len = len(total_vids)

print("Total questions : ", total_len)

for i in range(PARTS):
    start, end = int(i/PARTS * total_len), int((i+1)/PARTS * total_len)
    vids = total_vids[start:end]
    data_chunk = {vid:data[vid] for vid in vids}
    print(f"Size of chunk {i+1} : ", end-start+1)
    with open(os.path.join(DATA_ROOT, f'data/all_valid_p{i+1}.json'), 'w') as f:
        json.dump(data_chunk, f)
