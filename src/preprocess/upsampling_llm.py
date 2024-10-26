from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import json
from tqdm import tqdm
import re

def contains_chinese(text):
  return bool(re.search(r"[\u4e00-\u9fff]+", text))

few_shot_prompt = "Viết lại 10 caption từ một caption được cung cấp cho một bài đăng trên facebook\
. Hãy đảm bảo nội dung không thay đổi nhưng từ ngữ dùng phải khác hoàn toàn câu caption cũ\
. Mỗi câu caption viết lại được cách nhau bằng \"\\n###\\n\". Không liệt kê thứ tự các câu caption được viết lại\
. Câu caption cần viết lại là: XIUMIN ĐƯỢC FAN TÌNH CỜ BẮT GẶP VỚI VISUAL KHÔNG TUỔI\nAnh cùng ekip đang ở Hải Phòng nghỉ ngơi để chuẩn bị cho Đại nhạc hội hoành tráng, khai trương Phố đi bộ - Công viên Vũ Yên tại Thành phố Đảo Hoàng Gia Vinhomes Royal Island (Vũ Yên, Hải Phòng) vào tối mai (1/6)\nĐược biết, XIUMIN sẽ có buổi tổng duyệt vào lúc 10h (1/6) và trình diễn vào lúc 20h cùng ngày."

few_shot_answer = """Xiumin bất ngờ lộ diện tại Hải Phòng, visual đỉnh cao khiến fan không thể rời mắt! Anh chàng đang cùng ekip chuẩn bị cho Đại nhạc hội khai trương Phố đi bộ - Công viên Vũ Yên vào tối mai (1/6) tại Vinhomes Royal Island. Buổi tổng duyệt sẽ diễn ra lúc 10h và màn trình diễn chính thức sẽ bắt đầu lúc 20h cùng ngày.
###
Tình cờ bắt gặp Xiumin tại Hải Phòng, visual trẻ trung bất chấp thời gian! Nam thần tượng đang ở thành phố hoa phượng đỏ để chuẩn bị cho sự kiện khai trương Phố đi bộ - Công viên Vũ Yên. Fan hãy sẵn sàng cho buổi tổng duyệt lúc 10h và đêm diễn hoành tráng lúc 20h tối mai nhé.
###
Xiumin xuất hiện bất ngờ tại Hải Phòng, visual đỉnh cao khiến fan phát sốt! Anh chàng đang cùng ekip gấp rút chuẩn bị cho Đại nhạc hội khai trương Phố đi bộ - Công viên Vũ Yên vào tối mai (1/6) tại Vinhomes Royal Island.
###
Xiumin khiến fan bất ngờ khi xuất hiện tại Hải Phòng với visual trẻ trung! Nam thần tượng đang ở thành phố hoa phượng đỏ để chuẩn bị cho sự kiện đặc biệt vào tối mai (1/6). Buổi tổng duyệt sẽ diễn ra lúc 10h và màn trình diễn chính thức sẽ bắt đầu lúc 20h cùng ngày.
###
May mắn bắt gặp Xiumin tại Hải Phòng, visual không tuổi khiến fan mê mệt! Anh chàng đang cùng ekip chuẩn bị cho Đại nhạc hội khai trương Phố đi bộ - Công viên Vũ Yên. Fan hãy sẵn sàng cho buổi tổng duyệt lúc 10h và đêm diễn hoành tráng lúc 20h tối mai nhé.
###
Xiumin bất ngờ lộ diện tại Hải Phòng, visual đỉnh cao khiến fan "đứng ngồi không yên"! Nam thần tượng đang ở thành phố hoa phượng đỏ để chuẩn bị cho sự kiện khai trương Phố đi bộ - Công viên Vũ Yên.
###
Tình cờ gặp Xiumin tại Hải Phòng, visual trẻ trung bất chấp thời gian! Anh chàng đang cùng ekip gấp rút chuẩn bị cho Đại nhạc hội khai trương Phố đi bộ - Công viên Vũ Yên vào tối mai (1/6) tại Vinhomes Royal Island.
###
Xiumin xuất hiện bất ngờ tại Hải Phòng, visual đỉnh cao khiến fan "tan chảy"! Nam thần tượng đang ở thành phố hoa phượng đỏ để chuẩn bị cho sự kiện đặc biệt vào tối mai (1/6). Buổi tổng duyệt sẽ diễn ra lúc 10h và màn trình diễn chính thức sẽ bắt đầu lúc 20h cùng ngày.
###
May mắn bắt gặp Xiumin tại Hải Phòng, visual không tuổi khiến fan "phát cuồng"! Anh chàng đang cùng ekip chuẩn bị cho Đại nhạc hội khai trương Phố đi bộ - Công viên Vũ Yên. Fan hãy sẵn sàng cho buổi tổng duyệt lúc 10h và đêm diễn hoành tráng lúc 20h tối mai nhé.
###
Xiumin bất ngờ lộ diện tại Hải Phòng, visual đỉnh cao khiến fan "điên đảo"! Nam thần tượng đang ở thành phố hoa phượng đỏ để chuẩn bị cho sự kiện khai trương Phố đi bộ - Công viên Vũ Yên."""


prompt = "Viết lại {num} caption từ một caption được cung cấp cho một bài đăng trên facebook\
. Hãy đảm bảo nội dung không thay đổi nhưng từ ngữ dùng phải khác hoàn toàn câu caption cũ\
. Mỗi câu caption viết lại được cách nhau bằng \"\\n###\\n\". Không liệt kê thứ tự các câu caption được viết lại\
. Câu caption cần viết lại là: {caption}"

def generate_sample(caption, model, tokenizer, num_upsample):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer in proper Vietnamese."},
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
        max_new_tokens=4096,
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
    responses = [response for response in responses if not contains_chinese(response)]
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
    parser.add_argument("--output_path", type=str, default="data/public_train/ocr_llm_fix_train_text_upsample.json")
    args = parser.parse_args()


    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _ = up_sampling(args.input_path, args.output_path, model, tokenizer, args.num_sample_used, args.label_type, args.num_upsample)