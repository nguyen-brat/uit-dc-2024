import argparse
import json
import itertools
from tqdm import tqdm

from src.model import MSD, MSDConfig
import torch
from peft import PeftModel
from accelerate import dispatch_model, infer_auto_device_map

def get_n_items_at_a_time(dictionary, n=4):
    dict_items = iter(dictionary.items())
    while True:
        items = dict(itertools.islice(dict_items, n))
        if not items:
            break
        yield items


def group_subitems_by_key(batch):
    # Create lists to store the grouped sub-items
    images = []
    captions = []
    labels = []
    ocrs = []
    keys = []

    # Iterate through the batch and group the sub-items by key
    for key, value in batch.items():
        images.append(value['image'])
        captions.append(value['caption'])
        labels.append(value['label'])
        ocrs.append(value['ocr'])
        keys.append(key)

    # Return the grouped sub-items as a dictionary of lists
    return {
        'keys': keys,
        'images': images,
        'captions': captions,
        'labels': labels,
        'ocrs': ocrs
    }


def inference(args):
    result = {
        "results": {},
        "phase": args.phase,
    }
    data_multi_image_text = {}
    data_image_text = {}
    with open(args.annotation_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    print("hierarchiecal run image not vs multi-sarcasm")
    not_multi_sarcasm_model = MSD.from_pretrained(args.model_path_not_multi, torch_dtype=torch.bfloat16, device_map = "auto")
    total_batches = (len(data) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(get_n_items_at_a_time(data, args.batch_size), total=total_batches):
        caterories = not_multi_sarcasm_model.predict(batch, args.image_path)
        for key, caterory in caterories.items():
            result["results"][key] = caterory
            data[key]["label"] = caterory
        with open(args.output_dir, "w", encoding="utf") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    
    del not_multi_sarcasm_model

    for key, item in data.items():
        if item["label"] != "not-sarcasm":
            data_multi_image_text[key] = item
    print("hierarchiecal  multi vs image + text sarcasm")
    model_multi_image_text = MSD.from_pretrained(args.model_path_multi_image_text, torch_dtype=torch.bfloat16, device_map = "auto")
    total_batches = (len(data_multi_image_text) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(get_n_items_at_a_time(data_multi_image_text, args.batch_size), total=total_batches):
        caterories = model_multi_image_text.predict(batch, args.image_path)
        for key, caterory in caterories.items():
            result["results"][key] = caterory
            data[key]["label"] = caterory
        with open(args.output_dir, "w", encoding="utf") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    del model_multi_image_text

    for key, item in data_multi_image_text.items():
        if item["label"] != "multi-sarcasm":
            data_image_text[key] = item
    print("hierarchiecal image vs text sarcasm")
    model_image_text = MSD.from_pretrained(args.model_path_image_text, torch_dtype=torch.bfloat16, device_map = "auto")
    total_batches = (len(data_image_text) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(get_n_items_at_a_time(data_image_text, args.batch_size), total=total_batches):
        caterories = model_image_text.predict(batch, args.image_path)
        for key, caterory in caterories.items():
            result["results"][key] = caterory
            data[key]["label"] = caterory
        with open(args.output_dir, "w", encoding="utf") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_not_multi", type=str, default="model/hierarchical/cls_multi_not_sarcasm/merged_model")
    parser.add_argument("--model_path_multi_image_text", type=str, default="model/hierarchical/cls_multi_sarcasm_vs_image_text_sarcasm/merged_model")
    parser.add_argument("--model_path_image_text", type=str, default="model/hierarchical/cls_image_vs_text_sarcasm/merged_model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--annotation_path", type=str, default="data/private_test/processed_data.json")
    parser.add_argument("--image_path", type=str, default="data/private_test/test-images")
    parser.add_argument("--output_dir", type=str, default="submit/results_v2.json")
    parser.add_argument("--phase", type=str, default="test")
    args = parser.parse_args()

    inference(args)