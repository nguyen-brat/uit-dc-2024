from transformers import (
    Qwen2VLForConditionalGeneration,
    PreTrainedModel,
    AutoConfig
)
from qwen_vl_utils import process_vision_info
from torch import nn
import torch
from typing import Optional, List


class MSD(PreTrainedModel):
    def __init__(
            self,
            base_model,
            num_class,
    ):
        self.base = Qwen2VLForConditionalGeneration.from_pretrained(base_model)
        self.config = AutoConfig.from_pretrained(base_model)
        self.classification_layer = nn.Linear(self.config.hidden_size, num_class)

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
        seq_logits = self.base(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            rope_deltas,
        ).logtis

        mean_logits = self.masked_mean(seq_logits, attention_mask)
        logits = self.classification_layer(mean_logits)
        loss = self.compute_loss(logits, labels)

        return dict(
            logits=logits,
            loss=loss 
        )


    def masked_mean(tensor, mask, dim):
        """
        Calculate the mean along a specified dimension, considering only masked elements.
        
        Args:
        - tensor (torch.Tensor): Input tensor of shape (batch, seq_length, dim)
        - mask (torch.Tensor): Boolean mask of shape (batch, seq_length)
        - dim (int): Dimension along which to compute the mean (typically 1 for seq_length)
        
        Returns:
        - torch.Tensor: Mean tensor of shape (batch, dim)
        """
        mask_expanded = mask.unsqueeze(-1).expand_as(tensor)
        masked_tensor = tensor * mask_expanded.float()
        sum_tensor = torch.sum(masked_tensor, dim=dim)
        count = torch.sum(mask, dim=dim).unsqueeze(-1).expand_as(sum_tensor)
        
        # Compute mean (avoiding division by zero)
        mean = sum_tensor / (count + 1e-9)
        
        return mean
    

    def compute_loss(self, logits, labels):
        loss_fc = nn.CrossEntropyLoss()
        loss = loss_fc(logits, labels)

        return loss
    

    def freeze_base(self):
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def freeze_vision(self):
        if hasattr(self.base,'visual'):
            self.base.visual.requires_grad_(False)
            if hasattr(self.base.visual,'merger'):
                self.base.visual.merger.requires_grad_(True)