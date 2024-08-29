'''
    Fix file names of the form [movie_name][vid][fmt][ext] 
    (i.e., Oppenheimer_vsclkjsoi34.f130.mp4 to Oppenheimer_vsclkjsoi34.mp4) 
    [DOWNLOAD ISSUE IN ORIGINAL SCRIPT]
'''


import os
import shutil
import json

root_dir = "/home/shreyasjena/BTP/datasets/CinePile/yt_videos"
split="test"

vid_dir = os.path.join(root_dir, split)

for filename in os.listdir(vid_dir):
    parts = filename.split('.')
    if len(parts) == 3:
        movie_name_vid, fmt, ext = parts
        vid = movie_name_vid[-11:]
        shutil.move(os.path.join(vid_dir, filename), os.path.join(vid_dir, f"{vid}.{ext}"))
    