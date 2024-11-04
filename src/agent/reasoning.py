from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)
import json
import re
import torch
import os
import argparse
import itertools
from functools import partial
from os.path import join as osp
from tqdm import tqdm
from prompt import (
    create_reas_prompt_pixtral,
    create_reas_prompt_qwen2_vl,
    create_vi_intern_prompt,
    create_vi_intern_prompt_image
)
import traceback
from datetime import datetime
import gc
# from ..preprocess.clean import clean_hashtag

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


class ReasoningPrompt:
    def __init__(
            self, model="pixtral",
            generation_config = None,
    ):
        self.loaded_model = model
        if model == "pixtral":
            self.load_pixtral()
        elif model == "qwen2":
            self.load_qwen2()
        elif model == "vi_intern":
            self.load_vi_intern()
        else:
            raise KeyError(f"There currently not support {model} only support qwen2 and pixtral")
        self.generation_config = dict(
            max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=2.0
        ) if generation_config == None else generation_config

    @torch.no_grad()
    def generation(
            self,
            image_path,
            caption,
            label,
            ocr,
    )->str:
        inputs = self.prompt_creator(image_paths=image_path, captions=caption, labels=label, ocrs=ocr)
        if self.loaded_model != "vi_intern":
            inputs = inputs.to(self.model.device)
            generated_ids = self.model.generate(**inputs, **self.generation_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            del inputs
            del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            
            return output_text
        else:
            response, _ = self.model.chat(**inputs, generation_config=self.generation_config, history=None, return_history=True)
            return [response]

    def load_pixtral(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "mistral-community/pixtral-12b",
            device_map="auto",
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
        self.processor.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.prompt_creator = partial(create_reas_prompt_pixtral, self.processor)
        print("loading pixal model sccessfully")

    def load_qwen2(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            # attn_implementation="flash_attention_2",
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        ).eval()
        self.processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.prompt_creator = partial(create_reas_prompt_qwen2_vl, self.processor)
        print("loading qwen2-vl model sccessfully")
    
    def load_vi_intern(self):
        self.model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-3B-beta",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-3B-beta", trust_remote_code=True, use_fast=False)
        self.prompt_creator = partial(create_vi_intern_prompt_image, tokenizer)
        print("loading Vintern-3B-beta")



def Reasoning(prompt_creator:ReasoningPrompt, input_path, output_path, image_path, batch=4):
    output_file_name = output_path.split("/")[-1].split(".")[0]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # continue generating
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        for key in result.keys():
            if (result[key] != {}) and (key in data.keys()):
                del data[key]
    else:
        result = {key:{} for key in data.keys()}
    error_keys = []

    total_batches = (len(data) + batch - 1) // batch
    for batch in tqdm(get_n_items_at_a_time(data, batch), total=total_batches):
        group_item = group_subitems_by_key(batch)
        try:
            reasonings = prompt_creator.generation(
                image_path = [
                    osp(image_path, image)
                    for image in group_item["images"]
                ],
                caption=[
                    clean_hashtag(caption)
                    for caption in group_item["captions"]
                ],
                label=group_item["labels"],
                ocr=group_item["ocrs"],
            )

            for key, reasoning in zip(group_item["keys"], reasonings):
                result[key]["image_reasoning"] = reasoning
                result[key].update(batch[key])
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            del reasonings
            del group_item
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_message = f"{timestamp} - Error: {str(e)}\n"
            error_message += traceback.format_exc()
            with open(f"log/error_log_{output_file_name}.txt", "a") as error_file:
                error_file.write(error_message + "\n")
            
            # Save the keys that caused the error
            error_keys.extend(group_item["keys"])

            with open(f"log/key_error_{output_file_name}.txt", "a") as key_error_file:
                for key in error_keys:
                    key_error_file.write(f"{key}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vi_intern")
    parser.add_argument("--image_path", type=str, default="data/public_train/train-images")
    parser.add_argument("--input_path", type=str, default="data/public_train/vimmsd-train_01.json")
    parser.add_argument("--output_path", type=str, default="data/public_train/reasoning_vi_intern_image_reasoning_01.json")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    prompt_creator = ReasoningPrompt(args.model)
    _ = Reasoning(prompt_creator, args.input_path, args.output_path, args.image_path, args.batch_size)