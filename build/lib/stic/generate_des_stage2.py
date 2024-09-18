import random
import json

from videollava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN
)
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)

from tqdm import tqdm 
from peft import PeftModel

import argparse
import torch
import warnings
warnings.filterwarnings("ignore")


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_model_output(args):

    qs = args.query
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

    video_tensor = video_processor(args.video_file, return_tensors='pt')['pixel_values'][0].half().to("cuda")
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
    return outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="LanguageBind/Video-LLaVA-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-file", type=str, default="/data/yihe/COCO/val2014/COCO_val2014_000000033958.jpg")
    parser.add_argument("--query", type=str, default="Describe the image.")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--video-dir", type=str, default="/data1/yihedeng/image_data/")
    parser.add_argument("--save-dir", type=str, default="image_description.jsonl")
    parser.add_argument("--adapter-path", type=str, default=None)
    args = parser.parse_args()

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    video_processor = processor['video']
    
    if args.adapter_path is not None:
        
        adapter = PeftModel.from_pretrained(
            model, 
            args.adapter_path,
            device_map="cpu",
            offload_folder="offload"
            )
        
        model = adapter.merge_and_unload()
        # model.load_adapter(args.adapter_path)
        print("adapter loaded!")
    
    prompt_list = ["Illustrate the details of the video.",
                   "Summarize the visual content presented.",
                   "Explain what is depicted in the video.",
                   "Outline the key elements captured in the video.",
                   "Detail the composition and subjects within the various video frames.",
                   "Convey the atmosphere and mood represented in the snapshot.",
                   "Interpret the scene shown in the video.",
                   "Identify and describe the main focal points in the visual."]

    with open('outputs/sft_data_desc_ft_NExT-QA.json', 'r') as f:
        data_sft = json.load(f)

    directory = args.video_dir
    video_names = list(set([x['video'].split('/')[-1].split('.')[0] for x in data_sft]))
    print(len(video_names), video_names[0])

    # parallelize the task on multiple gpus to speed up the process
    for i in tqdm(range(len(video_names))):
        args.video_file = f"{directory}/{video_names[i]}.mp4"
        args.query = random.choice(prompt_list)
        output = get_model_output(args)

        d = {"video": args.video_file, 
            "prompt": args.query,
            "description": output}
        
        with open(args.save_dir,"a") as f:
            f.write(json.dumps(d))
            f.write(",\n")