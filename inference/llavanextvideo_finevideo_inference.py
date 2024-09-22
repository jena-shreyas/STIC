import random
import numpy as np
import os
import argparse
import json
import av
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from datasets import Dataset
from tqdm import tqdm
import io
import time

model_path = "llava-hf/LLaVA-NeXT-Video-7B-hf" 
parquet_dir = "/home/shreyasj/BTP/datasets/FineVideo"
device_id = 0

start = time.time()
# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_path, 
                                                               torch_dtype=torch.float16,       # 
                                                               attn_implementation="flash_attention_2",
                                                               low_cpu_mem_usage=True,
                                                               
                                                               ).to(device_id)
processor = LlavaNextVideoProcessor.from_pretrained(model_path)

load_time = time.time()
print("Model load time : ", load_time - start)

def resilient_generate(model, *args, **kwargs):

    oom = False
    try:
        return model.generate(*args, **kwargs)

    except torch.cuda.OutOfMemoryError as e:
        print(e)
        print("retrying with cache_implementation='offloaded'")
        oom = True

    if oom:
        torch.cuda.empty_cache()
        kwargs["cache_implementation"] = "offloaded"
        return model.generate(*args, **kwargs)
    
def read_video_pyav(video_bytes, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        video_bytes (bytes): Byte stream of the video.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''

    video_buffer = io.BytesIO(video_bytes)
    container = av.open(video_buffer)
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


def get_model_output(model, processor, video_bytes, qs, video_corruption=False):
    # define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": qs},
                {"type": "video"},
                ],
        },
    ]

    processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    video_buffer = io.BytesIO(video_bytes)
    container = av.open(video_buffer)

    # sample uniformly 16 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    num_frames = 32
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(video_bytes, indices)
    inputs_video = processor(text=processed_prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = resilient_generate(model, **inputs_video, max_length=1024)
    output_conv = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    # parse output_conv so that the text after "ASSISTANT:" is returned
    return output_conv.split("ASSISTANT:")[1].strip()


def inference():

    model.eval()

    prompt_list = ["Illustrate the details of the video.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the video.",
                   "Outline the key elements captured in the video.",
                   "Detail the composition and subjects within the various video frames.",
                   "Interpret the scene shown in the video.",
                   "Identify and describe the main focal points in the video."]
    
    jitter_prompt = "Describe the video by mentioning the colors of various consituent objects and characters in the scene, exactly as they appear. Do not instead report the expected colors of the objects or characters if they seem to be contrary to the expected real-life colors for those objects."
    noise_extra_prompt = " However, do not mention the presence of any noise or distortion in the video."
    
    full_prompt =   """
                        Please provide a detailed description of the entire video, considering both spatial and temporal elements as they evolve over time. Focus on the following:
                        Identify the main subjects (people, animals, objects) throughout the video and describe their actions as they unfold over different moments. How do their movements and interactions change from the beginning to the end of the video?
                        Based on the evolving sequence of events and visuals, what might the video represent or communicate as a whole? Consider how the video's progression contributes to its meaning, rather than focusing solely on a static frame.
                        What are the various events in the video and their resulting consequences in the video?
                        If applicable, specifically locate movements of objects as they evolve over time and how they interact in the video?
                    """
    
    hallu_prompt_list = [
                         "Modify the description by replacing objects or characters in the video with hypothetical objects or characters that could be part of the scene.",
                         "Enrich the video description by incorporating elements that, though absent, would seamlessly fit into the context of the video.",
                         "Enhance the video description by adding and describing additional everyday objects or activities that could be taking place just out of frame of the video being shown.",
                         "Augment the video description by inventing and incorporating details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."]
    
    # NOTE : Try to keep the hallu_prompt list small, as variance in prompts would adversely vary the output descriptions.

    # 1. Horrible, gives bullet points.
    # 2. Modified version of 2 adds "hypothetical" objects very well.
    # 3. Pretty good, performs as expected.
    # 4. Performs well, adds subtle wrong details.
    # 6. Description too similar to preferred, not much to be reject-worthy
    
    cache_dir = "/home/shreyasj/BTP/datasets/FineVideo/"
    arrow_dir = os.path.join(cache_dir, "HuggingFaceFV___finevideo/default/0.0.0/0d751f5e6563946f310c8a8e77a118e1f78af437")
    arrow_file = "finevideo-train-00902-of-00903.arrow"
    dataset = Dataset.from_file(os.path.join(arrow_dir, arrow_file))
    sample_idx = 0
    video_bytes = dataset['mp4'][sample_idx]

    with open("data_prep/FineVideo/sample_video.mp4", "wb") as f:
        f.write(video_bytes)

    query = full_prompt
    output = get_model_output(model, processor, video_bytes, query)
    print(output)


if __name__ == "__main__":
    inference()
    print("Inference time : ", time.time() - load_time)