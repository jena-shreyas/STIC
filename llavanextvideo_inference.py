from PIL import Image
import requests
import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import time

# Load the model in half-precision
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", 
                                                               torch_dtype=torch.float16,       # 
                                                               attn_implementation="flash_attention_2",
                                                               device_map="auto"
                                                               )
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

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

# Generate from image and video mixed inputs
# Load and image and write a new prompt
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

full_prompt = """Please provide a detailed description of the image, focusing on the following. 
    Describe the image and vividly detail the natural elements in the scene.
    Identify the main subjects (people, animals, objects) in the image and describe their appearance and interactions with the world around them.
    Describe the setting of the image. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the image convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the image's impact?
    Point out any details or symbols that might be relevant to understanding the image's meaning or context.
    If applicable, provide interpretations of what the image might represent or communicate."""
hallu_prompt = ""

img_path = "samples/sample_images/interlaken.jpeg"
image = Image.open(img_path)

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image"},
            ],
    }
]

# video_path = "samples/sample_videos/friends_ross_ben.mp4"
# container = av.open(video_path)
# total_frames = container.streams.video[0].frames
# num_frames = 16
# indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
# clip = read_video_pyav(container, indices)

# conversation = [
#     {

#         "role": "user",
#         "content": [
#             {"type": "text", "text": full_prompt},
#             {"type": "video"},
#             ],
#     }
# ]

# ,
#     {

#         "role": "assistant",
#         "content": [{"type": "text", "text": "There are two cats"}],
#     },
#     {

#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Why is this video funny?"},
#             {"type": "video"},
#             ],
#     },

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, padding=True, return_tensors="pt").to(model.device)        # In case of video inputs, Add argument "videos=clip"

# Generate
start = time.time()
generate_ids = model.generate(**inputs, max_length=800)
print("Time taken : ", (time.time() - start))
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output)
