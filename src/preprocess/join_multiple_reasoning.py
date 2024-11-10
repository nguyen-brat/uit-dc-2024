import json
from tqdm import tqdm
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])

with open("data/public_train/ocr_llm_reasoning_v2_train_text_upsample_x10.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

with open("data/public_train/image_text_reasoning/reasoning_vi_intern_image_reasoning_01.json", 'r', encoding='utf-8') as f:
    image_reasoning_01 = json.load(f)
with open("data/public_train/image_text_reasoning/reasoning_vi_intern_image_reasoning_02.json", 'r', encoding='utf-8') as f:
    image_reasoning_02 = json.load(f)

with open("data/public_train/image_text_reasoning/text_Arcee_reasoning_01.json", "r", encoding='utf-8') as f:
    text_reasoning_01 = json.load(f)
with open("data/public_train/image_text_reasoning/text_Arcee_reasoning_02.json", "r", encoding='utf-8') as f:
    text_reasoning_02 = json.load(f)

image_reasoning = {**image_reasoning_01, **image_reasoning_02}
text_reasoning = {**text_reasoning_01, **text_reasoning_02}

for key, value in data.items():
    data[key]["text_reasoning"] = text_reasoning[key]["text_reasoning"]
    data[key]["image_reasoning"] = image_reasoning[key]["image_reasoning"]


with open("data/public_train/image_text_reasoning/train_text_umsaple_x10_image_text_ocr_llm_reasoning_v2.json", "w", encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)