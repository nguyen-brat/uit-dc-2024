from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLPreTrainedModel
from transformers import AutoProcessor
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from typing import Optional, List
import functools
from os.path import join as osp
from dataclasses import dataclass
import math
from torch.utils.checkpoint import checkpoint

from .config import MSDConfig, MSDOutput
from .base import Qwen2VLHL, MSDCrossEncoderLayer
from ..dataloader.prompt import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_V2,
    USER_PROMPT,
    USER_PROMPT_V2,
    USER_PROMPT_V3,
    ASSISTANT_ANSWER,
)
from qwen_vl_utils import process_vision_info

@dataclass
class SmoothedCrossEntropyLoss:
    """
    Adds label-smoothing with class weighting on a pre-computed output from a Transformers model.

    Args:
        epsilon (float, optional, defaults to 0.1):
            The label smoothing factor.
        ignore_index (int, optional, defaults to -100):
            The index in the labels to ignore when computing the loss.
        class_weights (torch.Tensor, optional):
            A tensor of weights for each class to handle class imbalance.
    """

    epsilon: float = 0.1
    ignore_index: int = -100
    class_weights: torch.Tensor = None  # Added for class weighting

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        labels = torch.clamp(labels, min=0)

        # Compute negative log likelihood loss
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # Apply class weights if provided
        if self.class_weights is not None:
            weight_tensor = self.class_weights.gather(dim=-1, index=labels)
            nll_loss *= weight_tensor

        # Smoothing loss (total log probabilities for all classes)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        nll_loss = nll_loss.mean()
        smoothed_loss = smoothed_loss.mean()
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Based on your class distribution analysis
        if alpha is None:
            # Calculate alpha based on inverse square root of class frequencies
            frequencies = torch.tensor([1, 1, 1, 1])
            
            # Inverse square root weighting
            alpha = 1.0 / torch.sqrt(frequencies)
            
            # Normalize to sum to 1
            self.alpha = alpha / alpha.sum()
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=-1)
        # Get probability for the target class
        targets = targets.view(-1, 1)
        p_t = p.gather(1, targets).view(-1)
        # Calculate weights for each sample
        alpha_t = self.alpha.gather(0, targets.view(-1)).to(inputs.dtype).to(inputs.device)
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        # Combine focal weight with class weight (alpha)
        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MSD(Qwen2VLPreTrainedModel):
    config_class = MSDConfig

    def __init__(
            self,
            config:MSDConfig,
    ):
        super().__init__(config)
        if config.min_pixels and config.max_pixels:
            self.processor = AutoProcessor.from_pretrained(
                config.base_model,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels
            )
        else:
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.model = Qwen2VLHL.from_pretrained(config.base_model, **config.model_kwargs)
        self.encoder_layers = nn.ModuleList(
            MSDCrossEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + config.extra_layers)
        )
        self.classification_layer = nn.Linear(self.config.hidden_size, config.num_class)
        self.gradient_checkpointing = False
        self.class_ratio = config.class_ratio

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        logits = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            rope_deltas,
        ).logits

        for encoder_layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                logits = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )
            else:
                logits = encoder_layer(
                    logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
        #         sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        #         sequence_lengths = sequence_lengths % input_ids.shape[-1]
        #         sequence_lengths = sequence_lengths.to(logits.device)
        #     else:
        #         sequence_lengths = -1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), -1]

        # pooled_logits = self.masked_mean(logits, attention_mask, 1)
        if self.gradient_checkpointing and self.training:
            logits = self._gradient_checkpointing_func(
                self.classification_layer.__call__,
                pooled_logits,
            )
        else:
            logits = self.classification_layer(pooled_logits)
        
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return MSDOutput(loss=loss, logits=logits)


    def masked_mean(self, tensor:torch.Tensor, mask, dim):
        """
        Calculate the mean along a specified dimension, considering only masked elements.
        
        Args:
        - tensor (torch.Tensor): Input tensor of shape (batch, seq_length, dim)
        - mask (torch.Tensor): Boolean mask of shape (batch, seq_length)
        - dim (int): Dimension along which to compute the mean (typically 1 for seq_length)
        
        Returns:
        - torch.Tensor: Mean tensor of shape (batch, dim)
        """
        mask_expanded = mask.unsqueeze(-1).expand_as(tensor).to(tensor.dtype).to(tensor.device)
        masked_tensor = tensor * mask_expanded
        sum_tensor = torch.sum(masked_tensor, dim=dim)
        # count = torch.sum(mask, dim=dim).unsqueeze(-1).expand_as(sum_tensor).to(tensor.dtype).to(tensor.device)
        
        # Compute mean (avoiding division by zero)
        # mean = torch.div(sum_tensor, (count + torch.finfo(tensor.dtype).min))
        
        # return mean
        return sum_tensor
    

    def compute_loss(self, logits, labels):
        if self.class_ratio:
            weights = 1/torch.sqrt(torch.tensor(self.class_ratio, device=logits.device, dtype=logits.dtype))
            weights = weights/weights.sum()
        else:
            weights = torch.tensor([1/math.sqrt(3813), 1/math.sqrt(5446), 
                                1/math.sqrt(2619), 1/math.sqrt(682)], 
                                device=logits.device, dtype=logits.dtype)
            weights = weights/weights.sum()

        # Use custom loss with label smoothing and weights
        # criterion = SmoothedCrossEntropyLoss(epsilon=self.config.smoothing, class_weights=weights)
        criterion = CrossEntropyLoss(
            weight=weights
        )
        # criterion = FocalLoss(alpha = 1/torch.tensor([1, 1, 1], device=logits.device, dtype=logits.dtype))
        loss = criterion(logits, labels)
        return loss

    def freeze_base(self):
        for p in self.model.parameters():
            p.requires_grad_(False)


    def freeze_vision(self):
        if hasattr(self.model,'visual'):
            self.model.visual.requires_grad_(False)
            if hasattr(self.model.visual,'merger'):
                self.model.visual.merger.requires_grad_(True)


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        self._gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.model.gradient_checkpointing_disable()


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()
    

    @torch.no_grad()
    def predict(self, samples, image_path):
        '''
        predict output for evaluation
        '''
        chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        LABELS_MAP = {
            0: "multi-sarcasm",
            1: "not-sarcasm",
            2: "image-sarcasm",
            3: "text-sarcasm",
        }
        if self.config.id2label and (self.config.id2label != {0: 'LABEL_0', 1: 'LABEL_1'}):
            LABELS_MAP = self.config.id2label
            
        # messages = [
        #     [
        #         {"role": "system", "content": SYSTEM_PROMPT},
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": osp(image_path, sample["image"])},
        #                 {"type": "text", "text": USER_PROMPT.format(caption=sample["caption"], ocr=sample["ocr"])}
        #             ]
        #         }
        #     ] for sample in samples.values()
        # ]
        messages = []
        for sample in samples.values():
            user_instruct = USER_PROMPT_V3.format(
                caption=sample["caption"],
                ocr=sample["ocr"]
            )
            message = [
                    {"role": "system", "content": SYSTEM_PROMPT_V2},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_instruct.split("<image>")[0]},
                            {"type": "image", "image": osp(image_path, sample["image"])},
                            {"type": "text", "text": "<image>".join(user_instruct.split("<image>")[1:])}
                        ]
                    },
                    {
                        "role": "assistant", "content": ASSISTANT_ANSWER.format(
                            text_reasoning=sample["text_reasoning"],
                            image_reasoning=sample["image_reasoning"],
                            reasoning=sample["reasoning"]
                        )
                    }
                ]
            messages.append(message)

        texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
                chat_template=chat_template if self.processor.chat_template is None else self.processor.chat_template,
            ) for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.__call__(**inputs).logits.cpu()
        outputs = [LABELS_MAP[id_max.item()] for id_max in torch.argmax(logits, dim=-1)]
        return {key:output for key,output in zip(list(samples.keys()), outputs)}
    

    @torch.no_grad()
    def predict_origin(self, samples, image_path):
        '''
        predict output for evaluation
        '''
        chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        LABELS_MAP = {
            0: "multi-sarcasm",
            1: "not-sarcasm",
            2: "image-sarcasm",
            3: "text-sarcasm",
        }
        if self.config.id2label and (self.config.id2label != {0: 'LABEL_0', 1: 'LABEL_1'}):
            LABELS_MAP = self.config.id2label
            
        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": osp(image_path, sample["image"])},
                        {"type": "text", "text": USER_PROMPT.format(caption=sample["caption"], ocr=sample["ocr"])}
                    ]
                }
            ] for sample in samples.values()
        ]

        texts = [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
                chat_template=chat_template if self.processor.chat_template is None else self.processor.chat_template,
            ) for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.__call__(**inputs).logits.cpu()
        outputs = [LABELS_MAP[id_max.item()] for id_max in torch.argmax(logits, dim=-1)]
        return {key:output for key,output in zip(list(samples.keys()), outputs)}


if __name__ == "__main__":
    pass