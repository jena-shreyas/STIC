import json
import random

with open("outputs/COCO/data_pref_COCO_p1.jsonl", 'r') as f:
    img_data = [json.loads(line) for line in f]

with open("outputs/WebVid/data_pref_WebVid_p1.jsonl", 'r') as f:
    vid_data = [json.loads(line) for line in f]

# Add file extensions, if absent
for d in img_data:
    if not d['image'].endswith('.jpg'):
        d['image'] += '.jpg'

for d in vid_data:
    if not d['video'].endswith('.mp4'):
        d['video'] += '.mp4'

# Merge the two lists
data = img_data + vid_data
random.shuffle(data)

f = open("data/data_pref_merged.jsonl", 'w')
for d in data:
    # convert data to correct format
    d['prompt'] = d['chosen'][0]['content']
    chosen = d['chosen'][1]['content']
    rejected = d['rejected'][1]['content']
    d['chosen_conv'] = d['chosen']
    d['rejected_conv'] = d['rejected']
    d['chosen'] = chosen
    d['rejected'] = rejected
    f.write(json.dumps(d) + '\n')

f.close()