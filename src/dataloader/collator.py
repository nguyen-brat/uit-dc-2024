from typing import List, Dict, Optional, Union, Tuple
import torch
from transformers import PreTrainedTokenizerBase, Qwen2VLProcessor
from dataclasses import dataclass
from qwen_vl_utils import process_vision_info

chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

@dataclass
class MSDDataCollator:
    processor: Qwen2VLProcessor
    padding: bool = True
    max_length: int = 3000

    def __call__(self, features: List[Tuple[List[Dict[str, Union[str, List[Dict]]]], int]]):
        '''
        Receive list of dict return by the Dataloader
        '''
        labels = []
        messages = []
        for sample in features:
            labels.append(sample[1])
            messages.append(sample[0])
        
        texts = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template=chat_template if self.processor.chat_template is None else self.processor.chat_template,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            # truncation_side='right',
            return_tensors='pt',
        )

        inputs["labels"] = torch.tensor(labels)

        return inputs
    

if __name__ == "__main__":
    pass