import json
from tqdm import tqdm
import os
import sys
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
grand_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.extend([parent_dir, grand_dir])

def main(args):
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(args.image_reasoning_path, 'r', encoding='utf-8') as f:
        image_reasoning = json.load(f)

    with open(args.text_reasoning_path, "r", encoding='utf-8') as f:
        text_reasoning = json.load(f)
    with open(args.reasoning_path, "r", encoding='utf-8') as f:
        reasoning = json.load(f)

    for key, _ in data.items():
        data[key]["text_reasoning"] = text_reasoning[key]["text_reasoning"]
        data[key]["image_reasoning"] = image_reasoning[key]["image_reasoning"]
        data[key]["reasoning"] = reasoning[key]["reasoning"]


    with open(args.output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_reasoning_path", type=str)
    parser.add_argument("--image_reasoning_path", type=str)
    parser.add_argument("--reasoning_path", type=str)
    parser.add_argument("--ocr_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    main(args=args)