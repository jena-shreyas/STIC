import os
import av
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from huggingface_hub import hf_hub_download
import time

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 

start = time.time()

# Can I make this the color_jitter specific prompt?
jitter_query = "Describe the video by mentioning the colors of various consituent objects and characters in the scene, exactly as they appear. Do not instead report the expected colors of the objects or characters if they seem to be contrary to the expected real-life colors for those objects."
noise_query = "Illustrate the details of the video. However, do not mention the presence of any noise or distortion in the video."

mod_query = "Augment the video description by inventing and incorporating details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."
full_query = """Please provide a detailed description of the video, focusing on the following. 
    Identify the main subjects (people, animals, objects) in the video and describe what they are doing.
    Describe the setting of the video. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the video convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the video's impact?
    Point out any details or symbols that might be relevant to understanding the video's meaning or context.
    If applicable, provide interpretations of what the video might represent or communicate."""

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": noise_query},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# video_dir = "/home/shreyasjena/BTP/datasets/WebVid/videos"
# video_name = "stock-footage-husband-and-wife-admire-the-view-from-their-terrace-balcony-of-new-house-apartment-and-the"
video_dir = "/home/shreyasjena/BTP/models/STIC/pert_vids/frame_noise"
# "stock-footage-hyperlaspe-pov-driving-on-off-road-tracks-in-the-puntas-de-calnegre-natural-park-on-the-coast-in"
video_name = "stock-footage-hyperlaspe-pov-driving-on-off-road-tracks-in-the-puntas-de-calnegre-natural-park-on-the-coast-in"
video_path = os.path.join(video_dir, video_name + ".mp4")
container = av.open(video_path)     # INPUT FRAME RATE : 30 fps

# sample uniformly 8 frames from the video, can sample more for longer videos  
total_frames = container.streams.video[0].frames
num_frames = 16

# input_frame_rate = 30 
# duration = total_frames // input_frame_rate
# num_frames = int(math.pow(2, math.ceil(math.log2(duration))))

# print("Total frames : ", total_frames)
# print("Duration : ", duration)
# print("Num frames sampled : ", num_frames)

indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
clip = read_video_pyav(container, indices)
inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

output = model.generate(**inputs_video, max_new_tokens=1024, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

end = time.time()
print("Inference time : ", end-start)
