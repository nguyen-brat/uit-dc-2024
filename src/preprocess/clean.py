import re
import json
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

def clean_hashtag(caption):
    return re.sub(r'\n?#\S+', '', caption).strip()

# remove the term To determnine if of the reasoning term
def clean_reasoning(reasoning):
    terms = reasoning.split(",")
    # case for qwen2-vl
    if ("To determine" in terms[0]) or ("to determine" in terms[0]):
        reasoning = ",".join(terms[1:]).strip().capitalize()

    if reasoning.find('.') < reasoning.find(':'):
        terms = reasoning.split(".")
        spliter = '.'
    else:
        terms = reasoning.split(":")
        spliter = ':'
    for label_type in ["image-sarcasm", "text-sarcasm", "not-sarcasm", "multi-sarcasm", "sarcasm"]:
        if label_type in terms[0].lower():
            if label_type == "not-sarcasm":
                terms[0] = terms[0] + " and sarcasm signals"
            else:
                terms[0] += " and not-sarcasm signals"
            return f"{spliter}".join(terms).strip().capitalize()
        
    return reasoning

def remove_chinese_tokens(text):
  # Regular expression to match Chinese characters (simplified and traditional)
  chinese_pattern = r"[\u4e00-\u9fa5\u3400-\u4dbf]"

  # Remove Chinese characters using the regular expression
  text = re.sub(chinese_pattern, "", text)

  return text

def clean_ocr_result(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for _, item in tqdm(data.items()):
        item["ocr"] = remove_redundancy(item["ocr"])
        if "Trích xuất văn bản có trong ảnh." in item["ocr"]:
            item["ocr"] = re.sub("Trích xuất văn bản có trong ảnh.", "", item["ocr"])
        if "Trích xuất văn bản có trong ảnh" in item["ocr"]:
            item["ocr"] = re.sub("Trích xuất văn bản có trong ảnh", "", item["ocr"])

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)