import math
import random
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from videollava.train.train import smart_tokenizer_and_embedding_resize
from videollava.utils import disable_torch_init

from video_perturbations import perturb_video


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True, default='LanguageBind/Video-LLaVA-7B')
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--corrupt_dir', help='Directory containing corrupted video files.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--device_id", type=int, required=False, default=0)
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args, video_corruption=False):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    video_tensor = video_processor(video, return_tensors='pt')['pixel_values'][0].half().to("cuda")
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


def generate_pref(args):
    
    # Initialize the model
    disable_torch_init()
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to("cuda")
    video_processor = processor['video']

    prompt_list = ["Illustrate the details of the video.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the video.",
                   "Outline the key elements captured in the video.",
                   "Detail the composition and subjects within the various video frames.",
                   "Convey the atmosphere and mood represented in the snapshot.",
                   "Interpret the scene shown in the video.",
                   "Identify and describe the main focal points in the visual."]
    
    full_prompt = """Please provide a detailed description of the video, focusing on the following. 
    Identify the main subjects (people, animals, objects) in the video and describe what they are doing.
    Describe the setting of the video. Is it indoors or outdoors? What kind of environment or location does it depict? 
    What mood does the video convey? Are there any specific elements (such as lighting, weather, expressions) that contribute to this atmosphere? 
    Describe the dominant colors and the overall composition. How do these elements affect the video's impact?
    Point out any details or symbols that might be relevant to understanding the video's meaning or context.
    If applicable, provide interpretations of what the video might represent or communicate."""
    
    hallu_prompt_list = ["Describe the video with imaginative objects that may exist in the scene.",
                         "Enrich the description by adding hypothetical objects or characters that could be part of the scene.",
                         "Suggest and detail practical items or people that could logically inhabit the video's setting.",
                         "Incorporate elements that, though absent, would seamlessly fit into the context of the video.",
                         "Imagine and describe additional everyday objects or activities taking place just out of frame.",
                         "Augment the scene with details of potential events or items that are plausible.",
                         "Conceive of and detail natural elements, such as weather or animals, that could realistically enter the video scene. Make the description affirmative.",
                         "Invent and incorporate details of practical tools, vehicles, or gadgets that could be expected in a similar scenario."]
    

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    video_filenames = os.listdir(args.video_dir)
    
    # Split the video files between the two GPUs
    device_id = int(args.device_id)
    num_videos = len(video_filenames) // 2
    video_filenames = video_filenames[:num_videos] if device_id == 0 else video_filenames[num_videos:]
    
    video_corruptions = [
        "frame_flip",
        "frame_blur",
        "frame_jitter"
    ]

    for filename in tqdm(video_filenames):
        video_path = os.path.join(args.video_dir, filename)
        video_name = filename.split(".")[0]
        args.query = full_prompt
        video_corruption = False

        preferred_output = get_model_output(model, video_processor, tokenizer, video_path, args.query, args)
        
        hallu_prompt = ""
        sample_corruption = ""
        prompt = random.choice(prompt_list)

        # random sample a number between 0 and 1
        if random.random() > 0.5:
            hallu_prompt = random.choice(hallu_prompt_list)
            args.query = hallu_prompt
            video_corruption = False
            corrupted_output = get_model_output(model, video_processor, tokenizer, video_path, args.query, args, video_corruption)
        else:
            args.query = prompt
            video_corruption = True
            sample_corruption = random.choice(video_corruptions)
            video_corruption_dir = os.path.join(args.corrupt_dir, sample_corruption)
            corrupted_video_path = perturb_video(sample_corruption, video_path, video_corruption_dir)
            corrupted_output = get_model_output(model, video_processor, tokenizer, corrupted_video_path, args.query, args, video_corruption)

        d = {"video": video_name, 
             "video_corruption": video_corruption,
             "corruption_type": sample_corruption,
             "hallu_prompt": hallu_prompt,
             "chosen": [{"role":"user","content":prompt},{"role":"assistant","content":preferred_output}],
             "rejected": [{"role":"user","content":prompt},{"role":"assistant","content":corrupted_output}]}
        
        answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
        
        with open(answers_file, "a") as f:
            f.write(json.dumps(d))
            f.write(",\n")


if __name__ == "__main__":
    args = parse_args()
    generate_pref(args)