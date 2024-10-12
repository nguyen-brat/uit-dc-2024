from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLPreTrainedModel
from transformers import AutoProcessor
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from typing import Optional, List
import functools
from os.path import join as osp
from torch.utils.checkpoint import checkpoint

from .config import MSDConfig
from .base import Qwen2VLHL, MSDCrossEncoderLayer
from ..dataloader.prompt import SYSTEM_PROMPT, USER_PROMPT
from qwen_vl_utils import process_vision_info



class MSD(Qwen2VLPreTrainedModel):
    config_class = MSDConfig

    def __init__(
            self,
            config:MSDConfig,
    ):
        super().__init__(config)
        self.processor = AutoProcessor.from_pretrained(
            config.base_model,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels
        )
        self.model = Qwen2VLHL.from_pretrained(config.base_model, **config.model_kwargs)
        self.encoder_layers = nn.ModuleList(
            MSDCrossEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers, config.num_hidden_layers + config.extra_layers)
        )
        self.classification_layer = nn.Linear(self.config.hidden_size, config.num_class)
        self.gradient_checkpointing = False
        self.test_config = config

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
        seq_logits = self.model(
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
                seq_logits = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    seq_logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )
            else:
                seq_logits = encoder_layer(
                    seq_logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )

        mean_logits = self.masked_mean(seq_logits, attention_mask, 1)
        # print("##########")
        # print(mean_logits.dtype)
        if self.gradient_checkpointing and self.training:
            logits = self._gradient_checkpointing_func(
                self.classification_layer.__call__,
                mean_logits,
            )
        else:
            logits = self.classification_layer(mean_logits)

        loss = self.compute_loss(logits, labels)

        return dict(loss=loss, logits=logits)


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
        mask_expanded = mask.unsqueeze(-1).expand_as(tensor).to(tensor.dtype)
        masked_tensor = tensor * mask_expanded
        sum_tensor = torch.sum(masked_tensor, dim=dim)
        count = torch.sum(mask, dim=dim).unsqueeze(-1).expand_as(sum_tensor).to(tensor.dtype)
        
        # Compute mean (avoiding division by zero)
        mean = torch.div(sum_tensor, (count + torch.finfo(tensor.dtype).min))
        
        return mean
    

    def compute_loss(self, logits, labels):
        loss_fc = CrossEntropyLoss()
        loss = loss_fc(logits, labels)

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
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        output = self.__call__(**inputs)

        return LABELS_MAP[output]



if __name__ == "__main__":
    pass