import os
import torch
from videollava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN
)
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)
from peft import PeftModel

model_path = "LanguageBind/Video-LLaVA-7B"
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name)
root_dir = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/shreyas/STIC"
adapter_path = os.path.join(root_dir, 'checkpoints/videollava_stic_stage1_2024-06-28_19-38-56')

if adapter_path is not None:
    adapter = PeftModel.from_pretrained(
            model, 
            adapter_path,
            device_map="cpu",
            offload_folder="offload"
            )
        
    model = adapter.merge_and_unload()
    print("adapter loaded!")
        
video_processor = processor['video']

video_dir = os.path.join(root_dir, 'data/NExT-QA/videos')
video_name = "4346954399.mp4"
video_path = os.path.join(video_dir, video_name)

description = "The video features a man and a child sitting at a table with a laptop and a keyboard. The man is seen playing with the child and holding him in his lap. The child is also seen playing with a toy keyboard and a toy laptop. The video captures the playful interaction between the man and the child, with the man holding the child in his lap and the child playing with the toys."
question = "why is the man and child looking at the screen"
options = ["listening to music", \
            "see what s there", \
            "to see if it works", \
            "interacting with baby", \
            "zoom focus"
]

ans = 1

idx2opt = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

qs = "Video description:\n" + description + "\n\n" + "Instruction: Answer the following question by choosing the most appropriate option (out of A, B, C, E).\n\nQuestion:\n" + question \
    + "?\n\nOptions:\n" \
    + '\n'.join([f'{idx2opt[i]}. {options[i]}' for i in range(len(idx2opt))]) \
    + "\n\nAnswer: "
        
if model.config.mm_use_im_start_end:
    qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
else:
    qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values'][0].half().to("cuda")
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda")

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

# print(input_ids.get_device(), video_tensor.get_device(), model.device)

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

print("OUTPUT : \n\n", outputs)     # Answers a single option: B
