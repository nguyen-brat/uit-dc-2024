from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from qwen_vl_utils import process_vision_info

class MSDDataloader(Dataset):
    def __init__(self, path):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        return
    
    
def collate_fn(batch, processor, device):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    return inputs