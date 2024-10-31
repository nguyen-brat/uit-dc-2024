from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import json
from tqdm import tqdm
import re

def contains_chinese(text):
  return bool(re.search(r"[\u4e00-\u9fff]+", text))

few_shot_prompt = "Viết lại 10 caption từ một caption được cung cấp cho một bài đăng trên facebook\
. Hãy đảm bảo nội dung không thay đổi ý nghĩa châm biếm ban đầu của câu nhưng từ ngữ dùng phải khác hoàn toàn câu caption cũ\
. Mỗi câu caption viết lại được cách nhau bằng \"\\n###\\n\".\
. Câu caption cần viết lại là: Không có anh, giới giải trí sẽ mất đi một người suốt ngày hay khóc"

few_shot_answer = """1. Không có anh, showbiz sẽ thiếu đi một người chuyên khóc lóc
###
2. Thiếu anh, làng giải trí sẽ bớt đi một người rơi lệ không ngừng
###
3. Vắng anh, thế giới nghệ thuật sẽ chẳng còn ai hay khóc
###
4. Nếu không có anh, làng giải trí sẽ mất đi một người nước mắt ngắn dài
###
5. Giới showbiz không có anh sẽ chẳng còn ai khóc nhiều đến thế
###
6. Anh mà không có, giới nghệ sĩ sẽ vắng bóng một người thích khóc nhè
###
7. Thiếu vắng anh, giới giải trí sẽ chẳng ai còn khóc thường xuyên
###
8. Không anh, ngành giải trí sẽ thiếu đi một người lúc nào cũng sụt sùi
###
9. Giới giải trí sẽ buồn chán nếu thiếu một người khóc lóc như anh
###
10. Nếu anh không có, làng giải trí sẽ không còn ai dễ xúc động đến vậy"""


prompt = "Viết lại {num} caption từ một caption được cung cấp cho một bài đăng trên facebook\
. Hãy đảm bảo nội dung không thay đổi ý nghĩa châm biếm ban đầu của câu nhưng từ ngữ dùng phải khác hoàn toàn câu caption cũ\
. Mỗi câu caption viết lại được cách nhau bằng \"\\n###\\n\".\
. Câu caption cần viết lại là: {caption}"

def generate_sample(caption, model, tokenizer, num_upsample):
    messages = [
        # {"role": "system", "content": "You are a helpful AI assistant. Answer in proper Vietnamese."},
        {"role": "user", "content": few_shot_prompt},
        {"role": "assistant", "content": few_shot_answer},
        {"role": "user", "content": prompt.format(caption=caption, num=num_upsample)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=20000,
        temperature=0.5,
        top_k=5,
        top_p=0.5,
        # repetition_penalty=1.5,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    responses = responses.split("\n###\n")
    responses = [" ".join(response.split(" ")[1:]) for response in responses if not contains_chinese(response)]
    return responses

def sort_by_len_func(sample):
    return len(sample["caption"])

def up_sampling(input_path, output_path, model, tokenizer, num_sample_used, label_type, num_upsample):
    with open(input_path, "r", encoding='utf-8') as f:
        data = list(json.load(f).values())
    data = [sample for sample in data if sample["label"]==label_type]

    new_data = []
    data.sort(reverse=True, key=sort_by_len_func)
    data = data[:num_sample_used]

    for sample in tqdm(data):
        generated_captions = generate_sample(sample["caption"], model, tokenizer, num_upsample)
        for caption in generated_captions:
            new_data.append({
                "image": sample["image"],
                "caption": caption,
                "label": sample["label"],
                "ocr": sample["ocr"]
            })

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_type", type=str, default="text-sarcasm")
    parser.add_argument("--num_sample_used", type=int, default="60")
    parser.add_argument("--num_upsample", type=int, default="10")
    parser.add_argument("--input_path", type=str, default="data/public_train/ocr_llm_fix_train.json")
    parser.add_argument("--output_path", type=str, default="data/public_train/ocr_llm_fix_train_text_upsample_x10.json")
    args = parser.parse_args()


    model_name = "arcee-ai/Arcee-VyLinh"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _ = up_sampling(args.input_path, args.output_path, model, tokenizer, args.num_sample_used, args.label_type, args.num_upsample)