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
        min_pixels=None,
        max_pixels=None,
        class_ratio=None,
        smoothing=None,
        model_kwargs={
            "attn_implementation": "sdpa",
        },
        **kwargs,
    ):
        self.base_model = base_model
        self.extra_layers = extra_layers
        self.num_class = num_class
        self.model_kwargs = model_kwargs
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.class_ratio = class_ratio
        self.smoothing = smoothing
        self.update(AutoConfig.from_pretrained(base_model).to_dict())
        super().__init__(**kwargs)


@dataclass
class MSDOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    # msd_labels: Optional[torch.Tensor] = None