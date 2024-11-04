import json


with open("data/public_train/ocr_llm_reasoning_v2.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
with open("data/public_train/reasoning_vi_intern_image_reasoning_01.json", 'r', encoding='utf-8') as f:
    image_reasoning_01 = json.load(f)
with open("data/public_train/reasoning_vi_intern_image_reasoning_02.json", 'r', encoding='utf-8') as f:
    image_reasoning_02 = json.load(f)

image_reasoning = {**image_reasoning_01, **image_reasoning_02}



with open("data/public_train/image_text_ocr_llm_reasoning_v2.json", "w", encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)