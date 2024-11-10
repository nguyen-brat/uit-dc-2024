import json
from tqdm import tqdm

with open("data/private_test/vimmsd-private-test.json", "r", encoding='utf-8') as f:
    data = json.load(f)
with open("data/private_test/text_reasoning.json", "r", encoding='utf-8') as f:
    text_reasoning = json.load(f)
with open("data/private_test/image_reasoning.json", "r", encoding='utf-8') as f:
    image_reasoning = json.load(f)
with open("data/private_test/reasoning.json", "r", encoding='utf-8') as f:
    reasoning = json.load(f)
with open("data/private_test/ocr_llm.json", "r", encoding='utf-8') as f:
    ocr = json.load(f)

for key, value in data.items():
    value["ocr"] = ocr[key]["ocr"]
    value["reasoning"] = reasoning[key]["reasoning"]
    value["text_reasoning"] = text_reasoning[key]["text_reasoning"]
    value["image_reasoning"] = image_reasoning[key]["image_reasoning"]

with open("data/private_test/processed_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)