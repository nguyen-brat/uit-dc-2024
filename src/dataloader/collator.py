from typing import List, Dict, Optional, Union, Tuple
import torch
from transformers import PreTrainedTokenizerBase, Qwen2VLProcessor
from dataclasses import dataclass
from qwen_vl_utils import process_vision_info


@dataclass
class MSDDataCollator:
    processor: Qwen2VLProcessor
    padding: bool = True
    max_length: int = None

    def __call__(self, features: List[Tuple[List[Dict[str, Union[str, List[Dict]]]], int]]):
        '''
        Receive list of dict return by the Dataloader
        '''
        labels = []
        texts = []
        for sample in features:
            labels.append(sample[1])
            texts.append(sample[0])
        
        image_inputs, video_inputs = process_vision_info(texts)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=self.padding,
            return_tensors='pt',
        )

        return inputs, torch.tensor(labels)