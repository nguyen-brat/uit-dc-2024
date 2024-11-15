from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from os.path import join as osp
from .collator import MSDDataCollator
from .prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    USER_PROMPT,
    USER_PROMPT_V2,
    USER_PROMPT_V3,
    ASSISTANT_ANSWER,
)
from transformers import Qwen2VLProcessor
import random

LABELS_MAP = {
    "multi-sarcasm": 0,
    "not-sarcasm": 1,
    "image-sarcasm": 2,
    "text-sarcasm": 3,
}


class MSDDataloader(Dataset):
    def __init__(self, annotate_paths, image_path, labels_map = None, ignore_labels = []):
        self._annotate = []
        for annotate_path in annotate_paths:
            with open(annotate_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._annotate += data
                elif isinstance(data, dict):
                    self._annotate += list(data.values())
                else:
                    raise TypeError(f"not support read data in: {type(data)} format")
        
        self.annotate = []
        for sample in self._annotate:
            if sample["label"] in ignore_labels:
                pass
            else:
                self.annotate.append(sample)

        random.shuffle(self.annotate)
        self.image_path = image_path
        self.labels_map = labels_map if labels_map else LABELS_MAP
        self.cached_data_dict = {}


    def __len__(self):
        return len(self.annotate)


    def __getitem__(self, idx):
        '''
        Return message and a labels
        '''
        if idx in self.cached_data_dict:
            return self.cached_data_dict[idx]
        user_instruct = USER_PROMPT_V3.format(
            caption=self.annotate[idx]["caption"],
            ocr=self.annotate[idx]["ocr"]
        )
        message = [
            {"role": "system", "content": SYSTEM_PROMPT_V2},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruct.split("<image>")[0]},
                    {"type": "image", "image": osp(self.image_path, self.annotate[idx]["image"])},
                    {"type": "text", "text": "<image>".join(user_instruct.split("<image>")[1:])}
                ]
            },
            {
                "role": "assistant", "content": ASSISTANT_ANSWER.format(
                    text_reasoning=self.annotate[idx]["text_reasoning"],
                    image_reasoning=self.annotate[idx]["image_reasoning"],
                    reasoning=self.annotate[idx]["reasoning"]
                )# self.annotate[idx]["reasoning"]
            }
        ]

        label = self.labels_map[self.annotate[idx]["label"]]

        self.cached_data_dict[idx] = (message, label)

        return message, label
    
    def calculate_class_ratio(self, inver_labels_map):
        ratio = {key: 0 for key in inver_labels_map.values()}
        for sample in self.annotate:
            id_map = self.labels_map[sample["label"]]
            iver_label = inver_labels_map[id_map]
            ratio[iver_label] += 1
        return ratio
    
# class MSDDataloader(Dataset):
#     def __init__(self, annotate_paths, image_path, labels_map = None, ignore_labels = None):
#         self._annotate = []
#         for annotate_path in annotate_paths:
#             with open(annotate_path, "r") as f:
#                 data = json.load(f)
#                 if isinstance(data, list):
#                     self._annotate += data
#                 elif isinstance(data, dict):
#                     self._annotate += list(data.values())
#                 else:
#                     raise TypeError(f"not support read data in: {type(data)} format")
        
#         self.annotate = []
#         for sample in self._annotate:
#             if sample["label"] in ignore_labels:
#                 pass
#             else:
#                 self.annotate.append(sample)

#         random.shuffle(self.annotate)
#         self.image_path = image_path
#         self.labels_map = labels_map if labels_map else LABELS_MAP
#         self.cached_data_dict = {}


#     def __len__(self):
#         return len(self.annotate)


#     def __getitem__(self, idx):
#         '''
#         Return message and a labels
#         '''
#         if idx in self.cached_data_dict:
#             return self.cached_data_dict[idx]

#         message = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": osp(self.image_path, self.annotate[idx]["image"])},
#                     {"type": "text", "text": USER_PROMPT.format(caption=self.annotate[idx]["caption"], ocr=self.annotate[idx]["ocr"])}
#                 ]
#             }
#         ]

#         label = self.labels_map[self.annotate[idx]["label"]]

#         self.cached_data_dict[idx] = (message, label)

#         return message, label
    
#     def calculate_class_ratio(self, inver_labels_map):
#         ratio = {key: 0 for key in inver_labels_map.values()}
#         for sample in self.annotate:
#             id_map = self.labels_map[sample["label"]]
#             iver_label = inver_labels_map[id_map]
#             ratio[iver_label] += 1
#         return ratio
    

if __name__ == "__main__":
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    # collator = MSDDataCollator(processor)
    dataset = MSDDataloader(["data/public_train/image_text_reasoning/image_text_ocr_llm_reasoning_v2.json"], "data/public_train/train-images")
    print(dataset[0])
    # dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    # for sample in dataloader:
    #     print(sample)