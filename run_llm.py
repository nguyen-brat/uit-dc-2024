from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
import torch
from src.agent.prompt import TRAIN_USER_INS

import json
from os.path import join as osp
import gc
import os
import sys
import itertools
from datetime import datetime
import traceback
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def remove_redundancy(s, n=4):
    result = []
    i = 0
    while i < len(s):
        # Find the longest repeating substring starting at index i
        for j in range(len(s), i, -1):
            substring = s[i:j]
            if len(substring) * n <= len(s[i:]) and s[i:i+len(substring)*n] == substring * n:
                # If the substring repeats n or more times, skip it
                i += len(substring) * n
                break
        else:
            # If no repeating substring is found, add the current character to the result
            result.append(s[i])
            i += 1
    
    return ''.join(result)

def get_n_items_at_a_time(dictionary, n=4):
    # Create an iterator for dictionary items
    dict_items = iter(dictionary.items())
    
    # Use itertools.islice to return n items at a time
    while True:
        # Get n items or less if the iterator is exhausted
        items = dict(itertools.islice(dict_items, n))
        
        if not items:
            # If there are no more items, break the loop
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

def create_infer_prompt(processor, items, image_path):
    user_inputs = [
        (
            TRAIN_USER_INS.format(
                ocr=remove_redundancy(ocr, 4),
                caption=caption
            ).split("<image>")[0],
            "<image>".join(TRAIN_USER_INS.format(
                ocr=remove_redundancy(ocr, 4),
                caption=caption
            ).split("<image>")[1:])
        ) for ocr, caption in zip(items["ocrs"], items["captions"])
    ]

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":user_ins[0]},
                    {"type": "image", "image": osp(image_path, image)},
                    {"type": "text", "text":user_ins[1]},
                ]
            }
        ] for user_ins, image in zip(user_inputs, items["images"]) 
    ]

    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        ) for message in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def inference_llm(model, processor, input_path_annote, image_path, output_path, batch_size):
    output_file_name = output_path.split("/")[-1].split(".")[0]
    with open(input_path_annote, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        for key in result.keys():
            if (result[key] != {}) and (key in data.keys()):
                del data[key]
    else:
        result = {key:{} for key in data.keys()}
    error_keys = []
    
    total_batches = (len(data) + batch - 1) // batch_size
    for batch in tqdm(get_n_items_at_a_time(data, batch_size), total=total_batches):
        items = group_subitems_by_key(batch)
        try:
            inputs = create_infer_prompt(processor, items, image_path).to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=1200)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            del inputs
            del generated_ids
            torch.cuda.empty_cache()
            gc.collect()

            for key, image, caption, ocr, output_text in zip(items["keys"], items["images"], items["captions"], items["ocrs"], output_texts):
                result[key]["image"] = image
                result[key]["reasoning"] = output_text
                result[key]["caption"] = caption
                result[key]["ocr"] = ocr

            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_message = f"{timestamp} - Error: {str(e)}\n"
            error_message += traceback.format_exc()
            with open(f"log/error_log_{output_file_name}.txt", "a") as error_file:
                error_file.write(error_message + "\n")
            
            # Save the keys that caused the error
            error_keys.extend(items["keys"])

            with open(f"log/key_error_{output_file_name}.txt", "a") as key_error_file:
                for key in error_keys:
                    key_error_file.write(f"{key}\n")

    return result


if __name__ == "__main__":
    model_path = "LLaMA-Factory/models/qwen2_vl_pixtral_ds_lora_sft_v1"

    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
    )

    _  = inference_llm(
        model,
        processor,
        "data/public_test/ocr_llm_fix.json",
        "data/public_test/dev-images",
        "submit/public_test_reasoning_qwen2_pixal_ds.json",
        4,
    )

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
