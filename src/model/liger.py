from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

from functools import partial
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)


def apply_liger_kernel_to_msd(
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    layer_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2-VL models.
    NOTE: Qwen2-VL is not available in transformers<=4.44.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        layer_norm (bool): Whether to apply Liger's LayerNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from transformers.models.qwen2_vl import modeling_qwen2_vl
    from ..model import msd

    from liger_kernel.transformers.model.qwen2_vl import (
        lce_forward as qwen2_vl_lce_forward,
    )

    # TODO: Support Qwen2-VL's multimodal RoPE implementation

    LigerRMSNormForQwen2VL = partial(LigerRMSNorm, init_fn="ones", casting_mode="gemma")
    if rms_norm:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L439
        modeling_qwen2_vl.Qwen2RMSNorm = LigerRMSNormForQwen2VL
    if layer_norm:
        modeling_qwen2_vl.LayerNorm = LigerLayerNorm
    if cross_entropy:
        msd.CrossEntropyLoss = LigerCrossEntropyLoss
    if fused_linear_cross_entropy:
        modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2_vl_lce_forward
    if swiglu:
        modeling_qwen2_vl.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules
        config: PretrainedConfig = model.config

        torch_dtype = config.torch_dtype

        if hasattr(model, "model"):
            # The case for Qwen2VLForConditionalGeneration.
            base_model = model.model.model
        else:
            # Direct Qwen2VLModel
            base_model = model

        if hasattr(model.model, "visual"):
            # Patch Qwen2VisionTransformerPretrainedModel
            for vision_block in model.model.visual.blocks:
                if layer_norm:
                    vision_block.norm1 = LigerLayerNorm(config.vision_config["embed_dim"], eps=1e-6).to(
                        torch_dtype
                    )
                    vision_block.norm2 = LigerLayerNorm(config.vision_config["embed_dim"], eps=1e-6).to(
                        torch_dtype
                    )

        if rms_norm:
            base_model.norm = LigerRMSNormForQwen2VL(
                config.hidden_size, eps=config.rms_norm_eps
            ).to(torch_dtype)
        for decoder_layer in base_model.layers:
            if swiglu:
                decoder_layer.mlp = LigerSwiGLUMLP(config).to(torch_dtype)
            if rms_norm:
                decoder_layer.input_layernorm = LigerRMSNormForQwen2VL(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)
                decoder_layer.post_attention_layernorm = LigerRMSNormForQwen2VL(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(torch_dtype)