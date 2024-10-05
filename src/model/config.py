from transformers import PretrainedConfig, AutoConfig
from transformers.utils import ModelOutput
import torch

from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict


class MSDConfig(PretrainedConfig):
    model_type = "msd"

    def __init__(
        self,
        base_model="Qwen/Qwen2-VL-7B-Instruct",
        extra_layers=1,
        num_class=4,
        model_kwargs={
            "attn_implementation": "sdpa",
        },
        **kwargs,
    ):
        self.base_model = base_model
        self.extra_layers = extra_layers
        self.num_class = num_class
        self.model_kwargs = model_kwargs
        self.update(AutoConfig.from_pretrained(base_model).to_dict())
        super().__init__(**kwargs)