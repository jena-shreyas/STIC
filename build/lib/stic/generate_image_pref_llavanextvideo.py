import random
import numpy as np
import os
import argparse
import json
import av
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image
import torchvision.transforms as T



def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True, default='LanguageBind/Video-LLaVA-7B')
    parser.add_argument('--img_dir', help='Directory containing image files.', required=True)
    parser.add_argument('--corrupt_dir', help='Directory containing corrupted image files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--device_id", type=int, required=False, default=0)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()


def load_image(image_file, image_corruption=None):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    if image_corruption is not None:
        if image_corruption == "blur":
            image = T.Resize(size=20)(image)
        elif image_corruption == "jitter":
            jitter = T.ColorJitter(brightness=.5, hue=.3)
            image = jitter(image)
    return image

def perturb_image(image_file, image_corruption_dir, image_corruption=None):
    if image_corruption is not None:
        corrupted_image = load_image(image_file, image_corruption)
        corrupted_image_path = os.path.join(image_corruption_dir, image_corruption, image_file.split("/")[-1])
        corrupted_image.save(corrupted_image_path)
        return corrupted_image_path
    return image_file

def get_model_output(model, processor, img_path, qs, image_corruption=None):
    # define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": qs},
                {"type": "image"},
                ],
        },
    ]

    processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    image = load_image(img_path, image_corruption)

    inputs_image = processor(text=processed_prompt, images=image, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_image, max_new_tokens=1024, do_sample=False)
    output_conv = processor.decode(output[0][2:], skip_special_tokens=True)

    # parse output_conv so that the text after "ASSISTANT:" is returned
    return output_conv.split("ASSISTANT:")[1].strip()


def generate_pref(args):
    
    model_id = args.model_path
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    prompt_list = ["Illustrate the details of the image.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the image.",
                   "Outline the key elements captured in the image.",
                   "Detail the composition and subjects within the image.",
                   "Interpret the scene shown in the image.",
                   "Identify and describe the main focal points in the image."]
    
    jitter_prompt = "Describe the image by mentioning the colors of various consituent objects and characters in the scene, exactly as they appear. Do not instead report the expected colors of the objects or characters if they seem to be contrary to the expected real-life colors for those objects."
    noise_extra_prompt = " However, do not mention the presence of any blur, noise or distortion in the image."

    full_prompt = """Please provide a detailed description of the image, focusing on the following. 
    Describe the image and vividly detail the natural elements in the scene.
    Identify the main subjects (people, animals, objects) in the image and describe their appearance and interactions with the world around them.
    Describe the setting of the image. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the image convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the image's impact?
    Point out any details or symbols that might be relevant to understanding the image's meaning or context.
    If applicable, provide interpretations of what the image might represent or communicate."""
    
    hallu_prompt_list = [
                         "Modify the description by replacing objects or characters in the image with hypothetical objects or characters that do not exist in the scene, but could be part of it.",
                         "Enhance the image description by adding and describing additional everyday objects or activities that could be taking place just out of frame of the image being shown.",
                         "Enhance the image description by suggesting and detailing practical items or people that could logically inhabit the image's setting."
                         "Augment the image description by inventing and incorporating details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."]
    

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    image_filenames = os.listdir(args.img_dir)

    ### CHANGE FILENAME HERE ###
    with open("rem_imgs_p1.txt", 'r') as f:     # 2300 videos
        image_filenames = [filename.strip() for filename in f.readlines()]
    ############################

    print("Total number of images: ", len(image_filenames))
    
    # # Split the video files between the two GPUs
    # device_id = int(args.device_id)
    # num_videos = len(video_filenames) // 2
    # video_filenames = video_filenames[:num_videos] if device_id == 0 else video_filenames[num_videos:]
    
    image_corruptions = [
        "blur",
        "jitter"
    ]

    for filename in tqdm(image_filenames):
        img_path = os.path.join(args.img_dir, filename)
        img_name = filename.split(".")[0]
        image_corruption = None
        
        hallu_prompt = ""
        prompt = random.choice(prompt_list)

        args.query = full_prompt
        try: 
            preferred_output = get_model_output(model, processor, img_path, args.query, image_corruption)

            # random sample a number between 0 and 1
            if random.random() > 0.5:
                hallu_prompt = random.choice(hallu_prompt_list)
                args.query = hallu_prompt
                image_corruption = None
                corrupted_output = get_model_output(model, processor, img_path, args.query, image_corruption)
            else:
                image_corruption = random.choice(image_corruptions)
                hallu_prompt = ""

                # color jitter needs a separate prompt to force model to mention the changed color in the perturbed image
                if image_corruption == "jitter":
                    args.query = jitter_prompt
                elif image_corruption == "blur":
                    args.query = prompt + noise_extra_prompt
                
                image_corruption_dir = args.corrupt_dir
                os.makedirs(os.path.join(image_corruption_dir, image_corruption), exist_ok=True)
                corrupted_image_path = perturb_image(img_path, image_corruption_dir, image_corruption)
                corrupted_output = get_model_output(model, processor, corrupted_image_path, args.query, image_corruption)

            d = {"image": img_name, 
                "image_corruption": (image_corruption != None),
                "corruption_type": image_corruption,
                "hallu_prompt": hallu_prompt,
                "chosen": [{"role":"user","content":prompt},{"role":"assistant","content":preferred_output}],
                "rejected": [{"role":"user","content":prompt},{"role":"assistant","content":corrupted_output}]}
            
            answers_file = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
            
            with open(answers_file, "a") as f:
                f.write(json.dumps(d))
                f.write("\n")

        except Exception as e:
            print(f"Error with image {img_path}: {e}")


if __name__ == "__main__":
    args = parse_args()
    generate_pref(args)