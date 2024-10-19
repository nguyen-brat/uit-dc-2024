import os
from os.path import join as osp
import json
import sys
from clean import clean_reasoning

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])
from agent.prompt import TRAIN_SYS_RESPONSE, TRAIN_USER_INS

def create_prompt_sharegpt_format(reasoning_file_path, image_path, output_path):
    with open(reasoning_file_path, 'r', encoding='utf-8') as f:
        reasoning_data = json.load(f)

    results = []
    for _, item in reasoning_data.items():
        if item != {}:
            
            result = {
                "messages":[
                    {
                        "content": TRAIN_USER_INS.format(ocr=item["ocr"], caption=item["caption"]),
                        "role": "user"
                    },
                    {
                        "content": TRAIN_SYS_RESPONSE.format(reason=clean_reasoning(item["reasoning"]), label=item["label"]),
                        "role": "assistant"
                    }
                ],
                "images": [osp("../", osp(image_path, item["image"]))]
            }

            results.append(result)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results

if __name__ == "__main__":
    _ = create_prompt_sharegpt_format(
        reasoning_file_path="data/public_train/reasoning_vlm_qwen2.json",
        image_path="data/public_train/train-images",
        output_path="LLaMA-Factory/data/sarcasm_detection_ds.json"
    )