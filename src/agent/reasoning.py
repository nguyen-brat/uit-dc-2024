from .prompt import create_reas_prompt
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import json
from os.path import join as osp

def reasoning(processor, input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            reasoning = create_reas_prompt(
                processor=processor,
                image_path=osp(input_path, value["image"]),
                caption=value["caption"],
                label=value["label"],
                ocr=value["ocr"],
            )
            data[key]["reasoning"] = reasoning
            json.dump(data, f)
        print("#############")
        print(value["caption"])
        print("--------------")
        print(reasoning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/public_train/ocr_llm.json")
    parser.add_argument("--output_path", type=str, default="data/public_train/reasoning_vlm.json")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    reasoning(processor, args.input_path, args.output_path)