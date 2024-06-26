import os
import cv2
import random
from PIL import Image
import numpy as np
from torchvision.transforms import ColorJitter


def flip(frame):
    flipped_frame = cv2.flip(frame, 1)
    return flipped_frame

def blur(frame):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    return blurred_frame

def jitter(frame):
    transform = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    pil_frame = Image.fromarray(frame)
    jittered_frame = np.array(transform(pil_frame))
    return jittered_frame

video_perturbations = {
    "frame_flip": flip,
    "frame_blur": blur,
    "frame_jitter": jitter
}
    
def perturb_video(sample_perturbation, video_path, video_corruption_dir):

    # load the video
    video_filename, ext = video_path.split("/")[-1].split(".")
    
    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dir = f'{video_corruption_dir}/{sample_perturbation}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{video_filename}.{ext}' 
    fps = 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 * 2)     # need to make sure the width and height are even numbers to avoid issues with the video writer
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2 * 2)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    # get the number of frames in the video
    # iterate over all frames in the video
    while(1):
        ret, frame = capture.read()
        if not ret:
            break
        writer.write(video_perturbations[sample_perturbation](frame))
    writer.release()
    return out_path
        

if __name__ == "__main__":
    video_path = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/shreyas/STIC/data/NExT-QA/videos/2399794335.mp4"
    video_filename = video_path.split("/")[-1]
    video_corruption_dir = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/shreyas/STIC/pert_vids"
    sample_perturbation = random.choice(list(video_perturbations.keys()))
    # print(sample_perturbation)
    sample_perturbation = "frame_jitter"
    corrupted_video_path = perturb_video(sample_perturbation, video_path, video_corruption_dir)