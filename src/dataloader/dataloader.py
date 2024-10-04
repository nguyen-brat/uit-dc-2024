from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from os.path import join as osp
from .collator import MSDDataCollator
from transformers import Qwen2VLProcessor


LABELS_MAP = {
    "multi-sarcasm": 0,
    "not-sarcasm": 1,
    "image-sarcasm": 2,
    "text-sarcasm": 3,
}

SYSTEM_PROMPT = "You are a helpful assistant. Imagine you are a content moderator on facebook."

USER_PROMPT = """You need to classify which post is multi-sarcasm, non-sarcasm, image-sarcasm, text-sarcasm.\
Sarcasm is any sample that satisfy any condition below:
1. Employs irony by saying the opposite of what is meant, especially to
mock or deride.
2. Contains a mismatch between the text and the image that suggests
sarcasm through contradiction or exaggeration.
3. Uses hyperbole to overstate or understate reality in a way that is
clearly not meant to be taken literally.
4. Incorporates sarcastic hashtags, emojis, or punctuation, which are
commonly used to convey sarcasm online.

The post you need to classify contain the image above. The caption of the post is {caption}. \
Text the image is: {ocr}."""

class MSDDataloader(Dataset):
    def __init__(self, annotate_path, image_path):
        with open(annotate_path, "r") as f:
            self.annotate = list(json.load(f).values())
        self.image_path = image_path
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

        label = LABELS_MAP[self.annotate[idx]["label"]]

        self.cached_data_dict[idx] = (message, label)

        return message, label
    

if __name__ == "__main__":
    processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    collator = MSDDataCollator(processor)
    dataset = MSDDataloader("data/ocr.json", "data/warmup-images")
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    for sample in dataloader:
        print(sample)