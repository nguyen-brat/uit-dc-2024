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
        extra_layers=2,
        num_class=None,
        model_kwargs=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.extra_layers = extra_layers
        self.num_labels = num_class
        self.model_kwargs = model_kwargs
        self.based_config = AutoConfig.from_pretrained(base_model)
        self.update(self.based_config.to_dict())
        super().__init__(**kwargs)