import os
import json

with open("outputs/WebVid/data_pref_WebVid_rem_videos.json", 'r') as f:
    desc_data = json.load(f)

total_vids = [vid.split('.')[0] for vid in os.listdir('/home/shreyasjena/BTP/datasets/WebVid/rem_videos')]
compl_vids = [d['video'] for d in desc_data]

with open("/home/shreyasjena/BTP/datasets/WebVid/inf_left_vids.txt", 'w') as f:
    left_vids = list(set(total_vids) - set(compl_vids))
    print("Total videos left for inference: ", len(left_vids))
    for vid in left_vids:
        f.write(vid + '\n')
