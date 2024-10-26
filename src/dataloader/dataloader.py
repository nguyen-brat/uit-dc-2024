from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from os.path import join as osp
from .collator import MSDDataCollator
from .prompt import SYSTEM_PROMPT, USER_PROMPT
from transformers import Qwen2VLProcessor
import random

LABELS_MAP = {
    "multi-sarcasm": 0,
    "not-sarcasm": 1,
    "image-sarcasm": 2,
    "text-sarcasm": 3,
}


class MSDDataloader(Dataset):
    def __init__(self, annotate_paths, image_path, labels_map = None):
        self.annotate = []
        for annotate_path in annotate_paths:
            with open(annotate_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.annotate += data
                elif isinstance(data, dict):
                    self.annotate += list(data.values())
                else:
                    raise TypeError(f"not support read data in: {type(data)} format")
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

        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": osp(self.image_path, self.annotate[idx]["image"])},
                    {"type": "text", "text": USER_PROMPT.format(caption=self.annotate[idx]["caption"], ocr=self.annotate[idx]["ocr"])}
                ]
            }
        ]

        label = self.labels_map[self.annotate[idx]["label"]]

        self.cached_data_dict[idx] = (message, label)

        return message, label
    
    def calculate_class_ratio(self):
        ratio = {key: 0 for key in LABELS_MAP.keys}
        for sample in self.annotate:
            ratio[sample["label"]] += 1
        return ratio
    

if __name__ == "__main__":
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    collator = MSDDataCollator(processor)
    dataset = MSDDataloader("data/ocr.json", "data/warmup-images")
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    for sample in dataloader:
        print(sample)