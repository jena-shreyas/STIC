import torch
import json
from tqdm import tqdm
from videollava.model.builder import load_pretrained_model
from videollava.mm_utils import get_model_name_from_path


model_path = "LanguageBind/Video-LLaVA-7B"
model_base = None
model_name = get_model_name_from_path(model_path)

tokenizer, pretrained_model, processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)
tokenizer.model_max_length = 1048

del(pretrained_model)
del(processor)

torch.cuda.empty_cache()

with open("outputs/video_description.json", 'r') as f:
    desc = json.load(f)

with open("outputs/video_description_pretrained.json", 'r') as f:
    desc_pt = json.load(f)
    
# len_pref, len_desc = len(data_pref), len(desc)
len_pref = len(desc)
# avg_len_pref, avg_len_desc = 0, 0
avg_len_pref = 0
avg_len_pref_pt = 0

for i in tqdm(len(desc)):
    pref_text = data["chosen"][1]["content"]
    rej_text = data["rejected"][1]["content"]
    avg_len_pref += (len(tokenizer(pref_text, padding=False).input_ids))/len_pref
    avg_len_rej += (len(tokenizer(rej_text, padding=False).input_ids))/len_pref
    
# for data in tqdm(desc):
#     text = data["description"]
#     avg_len_desc += (len(tokenizer(text, padding=False).input_ids))/len_desc
    
print(f"Average length of preferred descriptions: {avg_len_pref}")
print(f"Average length of rejected descriptions: {avg_len_rej}")