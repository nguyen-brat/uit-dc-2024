from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLPreTrainedModel
from transformers import AutoProcessor
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from typing import Optional, List
import functools
from os.path import join as osp
from torch.utils.checkpoint import checkpoint

from .config import MSDConfig, MSDOutput
from .base import Qwen2VLHL, MSDCrossEncoderLayer
from ..dataloader.prompt import SYSTEM_PROMPT, USER_PROMPT
from qwen_vl_utils import process_vision_info
from transformers import Trainer, LlamaForCausalLM

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # Based on your class distribution analysis
        if alpha is None:
            # Calculate alpha based on inverse square root of class frequencies
            frequencies = torch.tensor([
                0.167,  # multi-sarcasm
                0.078,  # not-sarcasm
                0.0005, # image-sarcasm
                0.00018 # text-sarcasm
            ])
            
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
        alpha_t = self.alpha.gather(0, targets.view(-1)).to(inputs.device)
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

        # mean_logits = self.masked_mean(logits, attention_mask, 1)
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
        count = torch.sum(mask, dim=dim).unsqueeze(-1).expand_as(sum_tensor).to(tensor.dtype).to(tensor.device)
        
        # Compute mean (avoiding division by zero)
        mean = torch.div(sum_tensor, (count + torch.finfo(tensor.dtype).min))
        
        return mean
    

    def compute_loss(self, logits, labels):
        '''
        weights = {
            'multi-sarcasm': 1/√0.167 ≈ 2.45,
            'not-sarcasm': 1/√0.078 ≈ 3.58,
            'image-sarcasm': 1/√0.005 ≈ 14.14,
            'text-sarcasm': 1/√0.00018 ≈ 74.54
        }
        '''
        custom_alpha = torch.tensor([
            0.15,  # multi-sarcasm
            0.20,  # not-sarcasm
            0.30,  # image-sarcasm
            0.35   # text-sarcasm
        ])
        
        # Initialize focal loss with recommended parameters
        criterion = FocalLoss(
            alpha=custom_alpha,  # Can be set to None to use inverse sqrt weighting
            gamma=2.5,          # Recommended value for this case
            reduction='mean'
        )
        # normal loss
        # weights = torch.tensor([2.45, 3.58, 14.14, 74.54], device=logits.device)
        # loss_fc = CrossEntropyLoss(weight=weights)
        # loss = loss_fc(logits, labels)
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
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        return self.model.get_input_embeddings()
    

    @torch.no_grad()
    def predict(self, sample, image_path):
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

        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": osp(image_path, sample["image"])},
                    {"type": "text", "text": USER_PROMPT.format(caption=sample["caption"], ocr=sample["ocr"])}
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True,
            chat_template=chat_template if self.processor.chat_template is None else self.processor.chat_template,
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.__call__(**inputs).logits.cpu()

        return LABELS_MAP[torch.argmax(logits, dim=-1).item()]



if __name__ == "__main__":
    pass