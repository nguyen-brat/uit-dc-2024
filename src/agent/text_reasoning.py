# from transformers import AutoModelForCausalLM, AutoTokenizer
# import argparse
# import json
# import torch
# import gc
# import os
# import traceback
# from datetime import datetime
# from tqdm import tqdm
# import re
# import itertools

# def get_n_items_at_a_time(dictionary, n):
#     # Create an iterator for dictionary items
#     dict_items = iter(dictionary.items())
    
#     # Use itertools.islice to return n items at a time
#     while True:
#         # Get n items or less if the iterator is exhausted
#         items = dict(itertools.islice(dict_items, n))
        
#         if not items:
#             # If there are no more items, break the loop
#             break
        
#         yield items


# def group_subitems_by_key(batch):
#     # Create lists to store the grouped sub-items
#     images = []
#     captions = []
#     labels = []
#     ocrs = []
#     keys = []

#     # Iterate through the batch and group the sub-items by key
#     for key, value in batch.items():
#         images.append(value['image'])
#         captions.append(value['caption'])
#         labels.append(value['label'])
#         ocrs.append(value.get('ocr', None))
#         keys.append(key)

#     # Return the grouped sub-items as a dictionary of lists
#     return {
#         'keys': keys,
#         'images': images,
#         'captions': captions,
#         'labels': labels,
#         'ocrs': ocrs
#     }


# def clean_hashtag(caption):
#     return re.sub(r'\n?#\S+', '', caption).strip()

# def get_model_input(captions, tokenizer):
#     prompt_format = "Câu nói sau có mang tính châm biếm không vì sao: {caption}"
#     prompts = [prompt_format.format(caption=caption) for caption in captions]
#     messages = [
#         [
#             {"role": "system", "content": "Bạn là trợ lí hữu ích."},
#             {"role": "user", "content": prompt}
#         ] for prompt in prompts
#     ]
#     texts = [
#         tokenizer.apply_chat_template(
#             message,
#             tokenize=False,
#             add_generation_prompt=True
#         ) for message in messages
#     ]
#     model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
#     return model_inputs


# def generate(args, model, tokenizer):
#     output_file_name = args.output_path.split("/")[-1].split(".")[0]
#     with open(args.input_path, "r", encoding='utf-8') as f:
#         data = json.load(f)
#     # continue generating
#     if os.path.isfile(args.output_path):
#         with open(args.output_path, "r", encoding="utf-8") as f:
#             result = json.load(f)
#         for key in result.keys():
#             if (result[key] != {}) and (key in data.keys()):
#                 del data[key]
#     else:
#         result = {key:{} for key in data.keys()}
    
#     total_batches = (len(data) + args.batch - 1) // args.batch
#     for batch in tqdm(get_n_items_at_a_time(data, args.batch), total=total_batches):
#         group_item = group_subitems_by_key(batch)
#         try:
#             model_inputs = get_model_input([
#                     clean_hashtag(caption)
#                     for caption in group_item["captions"]
#                 ],
#                 tokenizer
#             ).to(model.device)
#             generated_ids = model.generate(
#                 model_inputs.input_ids,
#                 max_new_tokens=512,
#                 eos_token_id=tokenizer.eos_token_id,
#                 temperature=0.25,
#                 top_p=0.1,
#                 use_cache=True
#             )
#             generated_ids = [
#                 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#             ]
#             responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#             for key, reasoning in zip(group_item["keys"], responses):
#                 result[key]["text_reasoning"] = reasoning
#                 result[key].update(batch[key])
#             with open(args.output_path, "w", encoding="utf-8") as f:
#                 json.dump(result, f, ensure_ascii=False, indent=4)

#             del model_inputs
#             del generated_ids
#             torch.cuda.empty_cache()
#             gc.collect()
#         except Exception as e:
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             error_message = f"{timestamp} - Error: {str(e)}\n"
#             error_message += traceback.format_exc()
#             with open(f"log/error_log_{output_file_name}.txt", "a") as error_file:
#                 error_file.write(error_message + "\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="arcee-ai/Arcee-VyLinh")
#     parser.add_argument("--input_path", type=str, default="data/warn_up/vimmsd-warmup.json")
#     parser.add_argument("--output_path", type=str, default="data/warn_up/dump_output.json")
#     parser.add_argument("--batch", type=int, default=1)
#     args = parser.parse_args()

#     model = AutoModelForCausalLM.from_pretrained(
#         "arcee-ai/Arcee-VyLinh",
#         device_map="auto",
#         torch_dtype= "auto",# torch.bfloat16,
#         attn_implementation="flash_attention_2"
#     )
#     tokenizer = AutoTokenizer.from_pretrained("arcee-ai/Arcee-VyLinh")
#     tokenizer.padding_side = "left"
#     generate(args, model, tokenizer)

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
import torch.multiprocessing as mp
from functools import partial
from os.path import join as osp

def split_dict(data, chunks):
    """Split dictionary into n chunks, handling uneven sizes
    
    Args:
        data (dict): Dictionary to split
        chunks (int): Number of chunks to split into
        
    Returns:
        list: List of dictionaries, where the last chunk may have more items if uneven
    """
    items = list(data.items())
    base_chunk_size = len(items) // chunks
    remainder = len(items) % chunks
    
    result = []
    start = 0
    
    for i in range(chunks):
        # Add one extra item to some chunks if there's a remainder
        current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end = start + current_chunk_size
        
        chunk_dict = {k: v for k, v in items[start:end]}
        result.append(chunk_dict)
        start = end
    
    return result

def get_n_items_at_a_time(dictionary, n):
    dict_items = iter(dictionary.items())
    while True:
        items = dict(itertools.islice(dict_items, n))
        if not items:
            break
        yield items

def group_subitems_by_key(batch):
    images = []
    captions = []
    labels = []
    ocrs = []
    keys = []

    for key, value in batch.items():
        images.append(value['image'])
        captions.append(value['caption'])
        labels.append(value['label'])
        ocrs.append(value.get('ocr', None))
        keys.append(key)

    return {
        'keys': keys,
        'images': images,
        'captions': captions,
        'labels': labels,
        'ocrs': ocrs
    }

def clean_hashtag(caption):
    return re.sub(r'\n?#\S+', '', caption).strip()

class TextGenerator:
    def __init__(
            self, 
            model_name,
            generation_config=None,
            gpu_id=0
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.load_model()
        self.generation_config = {
            'max_new_tokens': 512,
            'temperature': 0.25,
            'top_p': 0.1,
            'use_cache': True
        } if generation_config is None else generation_config

    def load_model(self):
        print(f"Loading model on GPU {self.gpu_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=f"cuda:{self.gpu_id}",
            torch_dtype="auto",
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"

    @torch.no_grad()
    def generate(self, captions):
        prompt_format = "Câu nói sau có mang tính châm biếm không vì sao: {caption}"
        prompts = [prompt_format.format(caption=caption) for caption in captions]
        messages = [
            [
                {"role": "system", "content": "Bạn là trợ lí hữu ích."},
                {"role": "user", "content": prompt}
            ] for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            ) for message in messages
        ]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        
        generated_ids = self.model.generate(
            **inputs,
            **self.generation_config,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        del inputs
        del generated_ids
        torch.cuda.empty_cache()
        gc.collect()
        
        return outputs

def process_chunk(gpu_id, data_chunk, model_name, output_path, batch_size):
    output_file_name = output_path.split("/")[-1].split(".")[0]
    result = {key: {} for key in data_chunk.keys()}
    error_keys = []
    
    generator = TextGenerator(model_name=model_name, gpu_id=gpu_id)
    
    total_batches = (len(data_chunk) + batch_size - 1) // batch_size
    for batch in tqdm(get_n_items_at_a_time(data_chunk, batch_size), 
                     total=total_batches, 
                     desc=f'GPU {gpu_id}'):
        group_item = group_subitems_by_key(batch)
        try:
            responses = generator.generate(
                [clean_hashtag(caption) for caption in group_item["captions"]]
            )

            for key, response in zip(group_item["keys"], responses):
                result[key]["text_reasoning"] = response
                result[key].update(batch[key])

            # Save intermediate results
            intermediate_path = f"{output_path}_gpu{gpu_id}_intermediate.json"
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

        except Exception as e:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_message = f"{timestamp} - GPU {gpu_id} - Error: {str(e)}\n"
            error_message += traceback.format_exc()
            
            os.makedirs("log", exist_ok=True)
            with open(f"log/error_log_{output_file_name}_gpu{gpu_id}.txt", "a") as error_file:
                error_file.write(error_message + "\n")
            
            error_keys.extend(group_item["keys"])
            with open(f"log/key_error_{output_file_name}_gpu{gpu_id}.txt", "a") as key_error_file:
                for key in error_keys:
                    key_error_file.write(f"{key}\n")

    # Save final results for this GPU
    final_path = f"{output_path}_gpu{gpu_id}_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="arcee-ai/Arcee-VyLinh")
    parser.add_argument("--input_path", type=str, default="data/warn_up/vimmsd-warmup.json")
    parser.add_argument("--output_path", type=str, default="data/warn_up/dump_output.json")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    # Load input data
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle continuation of previous run
    if os.path.isfile(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        for key in result.keys():
            if (result[key] != {}) and (key in data.keys()):
                del data[key]
    else:
        result = {key:{} for key in data.keys()}

    # Split data for parallel processing
    data_chunks = split_dict(data, 2)
    
    # Create processes for both GPUs
    processes = []
    for gpu_id in range(2):
        p = mp.Process(
            target=process_chunk,
            args=(
                gpu_id,
                data_chunks[gpu_id],
                args.model,
                args.output_path,
                args.batch
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Merge results from both GPUs
    final_results = {}
    for gpu_id in range(2):
        with open(f"{args.output_path}_gpu{gpu_id}_final.json", "r", encoding="utf-8") as f:
            gpu_results = json.load(f)
            final_results.update(gpu_results)
    
    # Combine with existing results
    final_results = {**result, **final_results}
    
    # Save final merged results
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Cleanup intermediate files
    for gpu_id in range(2):
        for file_type in ['intermediate', 'final']:
            file_path = f"{args.output_path}_gpu{gpu_id}_{file_type}.json"
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()