from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import torch
import gc
import os
import traceback
from datetime import datetime
from tqdm import tqdm
import re
import itertools

def get_n_items_at_a_time(dictionary, n):
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
        ocrs.append(value.get('ocr', None))
        keys.append(key)

    # Return the grouped sub-items as a dictionary of lists
    return {
        'keys': keys,
        'images': images,
        'captions': captions,
        'labels': labels,
        'ocrs': ocrs
    }


def clean_hashtag(caption):
    return re.sub(r'\n?#\S+', '', caption).strip()

def get_model_input(captions, tokenizer):
    prompt_format = "Câu nói sau có mang tính châm biếm không vì sao: {caption}"
    prompts = [prompt_format.format(caption=caption) for caption in captions]
    messages = [
        [
            {"role": "system", "content": "Bạn là trợ lí hữu ích."},
            {"role": "user", "content": prompt}
        ] for prompt in prompts
    ]
    texts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        ) for message in messages
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
    return model_inputs


def generate(args, model, tokenizer):
    output_file_name = args.output_path.split("/")[-1].split(".")[0]
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    # continue generating
    if os.path.isfile(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        for key in result.keys():
            if (result[key] != {}) and (key in data.keys()):
                del data[key]
    else:
        result = {key:{} for key in data.keys()}
    
    total_batches = (len(data) + args.batch - 1) // args.batch
    for batch in tqdm(get_n_items_at_a_time(data, args.batch), total=total_batches):
        group_item = group_subitems_by_key(batch)
        try:
            model_inputs = get_model_input([
                    clean_hashtag(caption)
                    for caption in group_item["captions"]
                ],
                tokenizer
            ).to(model.device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.25,
                top_p=0.1,
                use_cache=True
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for key, reasoning in zip(group_item["keys"], responses):
                result[key]["text_reasoning"] = reasoning
                result[key].update(batch[key])
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            del model_inputs
            del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_message = f"{timestamp} - Error: {str(e)}\n"
            error_message += traceback.format_exc()
            with open(f"log/error_log_{output_file_name}.txt", "a") as error_file:
                error_file.write(error_message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="arcee-ai/Arcee-VyLinh")
    parser.add_argument("--input_path", type=str, default="data/public_train/vimmsd-train_01.json")
    parser.add_argument("--output_path", type=str, default="data/public_train/text_Arcee_reasoning_01.json")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        "arcee-ai/Arcee-VyLinh",
        device_map="auto",
        torch_dtype= "auto",# torch.bfloat16,
        # attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Arcee-VyLinh")
    tokenizer.padding_side = "left"
    generate(args, model, tokenizer)