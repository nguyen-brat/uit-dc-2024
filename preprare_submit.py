import json

with open(r"submit/public_test_reasoning.json", "r", encoding='utf-8') as f:
    data = json.load(f)

result = {
    "results": {},
    "phase": "dev"
}
label_types = ["multi-sarcasm", "not-sarcasm", "text-sarcasm", "image-sarcasm"]

for key, item in data.items():
    label = item["reasoning"].split(" ")[-1]
    if label[-1] == '.':
        label = label[:-1]
    if label not in label_types:
        last_sentence = item["reasoning"].split("\n")[-1]
        for label_type in label_types:
            if label_type in last_sentence:
                label = label_type
    if label not in label_types:
        label = "multi-sarcasm"
    result["results"][key] = label

with open("submit/results.json", "w", encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)