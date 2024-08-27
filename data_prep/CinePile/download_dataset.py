import os
from os.path import dirname as osd, join as osj
import random
import subprocess
import json

random.seed(42)

from tqdm import tqdm

from datasets import load_dataset

def download_video(video_url, filename, root):
    """
    Download and convert a video from a URL and save it to a specified directory.

    Parameters:
    - video_url (str): The URL of the video to be downloaded.
    - filename (str): The base name for the output file, without file extension.
    - root (str): The root directory where the 'yt_videos' folder will be created.

    Returns:
    - tuple: A tuple containing the video URL and a boolean. The boolean is True if the
      download and conversion was successful, and False otherwise.
    """

    dir_path=root

    try:
        vid_prefix = os.path.join(dir_path, filename)
        full_command = [
            "yt-dlp",
            "-S",
            "height:224,ext:mp4:m4a",
            "--recode",
            "mp4",
            "-o",
            f"{vid_prefix}.mp4",
            video_url
        ]

        print(f'saving path: {vid_prefix}.mp4')

        result = subprocess.run(full_command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Downloaded: {vid_prefix}; {video_url}")
            return video_url, True
        else:
            print(f"Failed to download or convert {video_url}. Error: {result.stderr}")
            return video_url, False

    except Exception as e:
        print(f"Exception during download or conversion of {video_url}: {e}")
        return video_url, False
    
## load the dataset
DATA_ROOT=osj(osd(osd(os.getcwd())),"datasets/CinePile")
cinepile = load_dataset("tomg-group-umd/cinepile", cache_dir=DATA_ROOT)

# DATASET SPLIT VIDEO STATISTICS
for split in ['train', 'test']:
    cinepile_data = cinepile[split]
    vid2movie = {}
    vid2cliptitle = {}

    for i in tqdm(range(len(cinepile_data))):
        data = cinepile_data[i]
        clip_title, yt_link, movie_name = data['yt_clip_title'], data['yt_clip_link'], data['movie_name']
        vid = yt_link.split('=')[-1]
        vid2movie[vid] = movie_name
        vid2cliptitle[vid] = clip_title

    with open(f"data/cinepile_vid2cliptitle_{split}.json", 'w') as f:
        json.dump(vid2cliptitle, f, indent=4)

    print(f"Split: {split}")
    print("Total unique videos : ", len(vid2movie))
    
    if split == "train":
        continue    

    root_dir = osj(DATA_ROOT, f"yt_videos/{split}")
    os.makedirs(root_dir, exist_ok=True)
    
    print("Total videos to be downloaded : ", len(vid2movie))

    for vid in tqdm(vid2movie):
        yt_link = f"https://www.youtube.com/watch?v={vid}"
        vid_path = f"{vid2movie[vid]}_{vid}"
        while not os.path.exists(os.path.join(root_dir, vid_path + '.mp4')):
            print(f"Downloading {vid_path}...")
            _,  status = download_video(yt_link, vid_path, root=root_dir)
            if status:
                break
        print(f"Downloaded {vid_path}")
