import os
import av
import random
import skimage
from PIL import Image
import numpy as np
from torchvision.transforms.functional import adjust_hue


def noise(frame_pil, amount=0.4):
    # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_norm = np.array(frame_pil)/255.0
    noise_frame = skimage.util.random_noise(frame_norm, "s&p", amount=amount)
    # noise_frame_bgr = cv2.cvtColor((noise_frame_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    output = (noise_frame * 255.0).astype(np.uint8)
    return output

def jitter(frame_pil, hue_factor=0.5):
    jittered_frame_pil = adjust_hue(frame_pil, hue_factor).convert("RGB")
    output = np.array(jittered_frame_pil)
    return output

video_perturbations = {
    "frame_noise": noise,
    "frame_jitter": jitter
}
    
def perturb_video(sample_perturbation, video_path, video_corruption_dir):

    # load the video
    out_dir = f'{video_corruption_dir}/{sample_perturbation}'
    os.makedirs(out_dir, exist_ok=True)
    video_filename = video_path.split("/")[-1]
    out_path = f'{out_dir}/{video_filename}' 
    fps = 30

    hue_factor = 0.
    amount = 0.

    if sample_perturbation == "frame_jitter":
        a, b, c, d = -0.5, -0.2, 0.2, 0.5

        prob = np.array([b-a, d-c])
        prob = prob/prob.sum() # Normalize to sum up to one

        hue_factor = np.random.choice([np.random.uniform(a, b), np.random.uniform(c, d)], p=prob)
        # hue_factor = -0.5
        print("Hue : ", hue_factor)
    elif sample_perturbation == "frame_noise":
        amount = np.random.uniform(0.4, 0.6)    # Strong noise needed since videos are high-quality and model is good at handling noise
        # amount = 0.6
        print("Noise : ", amount)

    container = av.open(video_path)

    out_container = av.open(out_path, mode="w")
    out_stream = out_container.add_stream("mpeg4", rate=fps)

    for i, frame in enumerate(container.decode(video=0)):
        # Convert the frame to an image (numpy array)
        img = frame.to_image().convert('RGB')       # PIL image
        width, height = img.size

        if i==0:
            out_stream.width = width
            out_stream.height = height

        if sample_perturbation == "frame_jitter":
            pert_frame = video_perturbations[sample_perturbation](img, hue_factor)        # np.array
        elif sample_perturbation == "frame_noise":
            pert_frame = video_perturbations[sample_perturbation](img, amount)        # np.array

        pert_frame = av.VideoFrame.from_ndarray(pert_frame, format="rgb24")

        for packet in out_stream.encode(pert_frame):
            out_container.mux(packet)

    # Flush output stream
    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()
    
    # capture = cv2.VideoCapture(video_path)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2 * 2)     # need to make sure the width and height are even numbers to avoid issues with the video writer
    # height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2 * 2)
    # writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    # # get the number of frames in the video
    # # iterate over all frames in the video

    # while(1):
    #     ret, frame = capture.read()
    #     if not ret:
    #         break
    #     if sample_perturbation == "frame_jitter":
    #         writer.write(video_perturbations[sample_perturbation](frame, hue_factor))
    #     else:
    #         writer.write(video_perturbations[sample_perturbation](frame))
    # writer.release()
    return out_path
        

if __name__ == "__main__":
    video_path = "/home/shreyasjena/BTP/datasets/WebVid/videos/stock-footage-a-cute-dog-lies-on-grass-and-dozes.mp4"
    video_filename = video_path.split("/")[-1]
    video_corruption_dir = "/home/shreyasjena/BTP/models/STIC/pert_vids"
    sample_perturbation = random.choice(list(video_perturbations.keys()))
    sample_perturbation = "frame_noise"
    corrupted_video_path = perturb_video(sample_perturbation, video_path, video_corruption_dir)
