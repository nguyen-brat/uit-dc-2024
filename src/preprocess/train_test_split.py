from sklearn.model_selection import train_test_split
import json

with open("../data/public_train/image_text_reasoning/image_text_ocr_llm_reasoning_v2.json", "r", encoding='utf-8') as f:
    data = json.load(f)

train_data, test_data = train_test_split(list(data.items()), test_size=0.1, random_state=42)

# Convert back to dictionary form
train_data_dict = dict(train_data)
test_data_dict = dict(test_data)

# Save the training and test data to files
train_file_path = "../data/public_train/ocr_llm_reasoning_v2_train.json"
test_file_path = "../data/public_train/ocr_llm_reasoning_v2_test.json"

with open(train_file_path, 'w', encoding='utf-8') as train_file:
    json.dump(train_data_dict, train_file, ensure_ascii=False, indent=4)

with open(test_file_path, 'w', encoding='utf-8') as test_file:
    json.dump(test_data_dict, test_file, ensure_ascii=False, indent=4)

print(train_file_path)
print(test_file_path)