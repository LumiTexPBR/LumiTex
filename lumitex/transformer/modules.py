# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FusedFluxAttnProcessor2_0,
)
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import FluxPosEmbed
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle

from diffusers.models.modeling_utils import ModelMixin

import os
import json
from contextlib import contextmanager
from einops import rearrange

from utils.logger import *
from .attention_processor import ShadedBranchAttnProcessor

logger = get_logger("Transformer2DModel")  # pylint: disable=invalid-name


class DINOv2Featurizer(nn.Module):
    
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def __init__(self, arch, patch_size, feat_type):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type

        self.n_feats = 128
        self.model = torch.hub.load('facebookresearch/dinov2', arch)
        if "vits" in arch:
            self.dim = 384
        elif "vitb" in arch:
            self.dim = 768
        elif "vitl" in arch:
            self.dim = 1024
        elif "vitg" in arch:
            self.dim = 1536
        else:
            raise NotImplementedError(f"Unknown architecture {arch}")
        
        for param in self.parameters():
            param.requires_grad = False

        self.model.eval()

    def get_cls_token(self, img):
        return self.model.forward(img)

    def forward(self, img, n=1, include_cls=False):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size
        return self.model.forward_features(img)["x_norm_patchtokens"].reshape(-1, h, w, self.dim).permute(0, 3, 1, 2)

    def preprocess_img(self, img, resolution=512):
        assert resolution % 16 == 0
        size = resolution // 16 * 14
        
        transform = T.Compose([
            T.Resize(size, T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            self.norm])
        
        if isinstance(img, Image.Image):
            return transform(img)
        elif isinstance(img, List):
            processed_imgs = [transform(im) for im in img]
            return torch.stack(processed_imgs)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")


# ============ Multi-View Attention Constants ============
NUM_VIEWS = 6               # Number of multi-view images per batch
NUM_TEXT_TOKENS = 729       # Text token sequence length (T5 512 + CLIP 217)
NUM_COND_TOKENS = 1024      # Condition token sequence length (normal + position)
ATTN_HEAD_DIM = 128         # Per-head dimension in FluxTransformer2DModel



@maybe_allow_in_graph
class BasicMultiViewSingleTransformerBlock(nn.Module):
    """Wraps ``FluxSingleTransformerBlock`` with optional multi-view self-attention.

    When ``use_ma=True``, an additional attention layer (``attn_mv``) performs
    self-attention across all views on the *image tokens only* (excluding text
    and condition tokens). The result is blended back via a zero-initialized
    linear projection so the block starts as an identity w.r.t. the base model.
    """

    def __init__(self, transformer: FluxSingleTransformerBlock, layer_name: str,
                 use_ma: bool = False, use_ra: bool = False):
        super().__init__()
        self.transformer = transformer
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra

        self.num_views = NUM_VIEWS
        self.mva_scale = 1.0
        self.ref_scale = 1.0

        if use_ra:
            raise NotImplementedError("Reference attention is not implemented for single blocks")

        if use_ma:
            processor = FluxAttnProcessor2_0()
            self.attn_mv = Attention(
                query_dim=self.transformer.attn.query_dim,
                cross_attention_dim=None,
                dim_head=ATTN_HEAD_DIM,
                heads=self.transformer.attn.heads,
                out_dim=self.transformer.attn.out_dim,
                bias=True,
                processor=processor,
                qk_norm="rms_norm",
                eps=1e-6,
                pre_only=True,
            )
            self.mv_out = nn.Linear(self.transformer.attn.out_dim, self.transformer.attn.out_dim)
            self._initialize_mv_weights()

    def _initialize_mv_weights(self):
        """Copy base attention weights; small random init for output projection."""
        self.attn_mv.load_state_dict(self.attn.state_dict())
        nn.init.normal_(self.mv_out.weight, std=0.02)
        nn.init.zeros_(self.mv_out.bias)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mv_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_ref_token: Optional[int] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        joint_attention_kwargs = joint_attention_kwargs or {}

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # 1. Standard self-attention
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # 2. Multi-view self-attention (image tokens only, across all views)
        if self.use_ma:
            # Extract image tokens: skip text prefix and condition suffix
            img_tokens = norm_hidden_states[:, NUM_TEXT_TOKENS:-NUM_COND_TOKENS, :]
            # Merge views: (B*N, L, C) -> (B, N*L, C)
            img_tokens = rearrange(img_tokens, '(b n) l c -> b (n l) c', n=self.num_views)
            # Flatten RoPE embeddings to match merged sequence
            mv_emb = tuple(rearrange(emb, 'n s d -> (n s) d') for emb in mv_rotary_emb)

            attn_output_mv = self.attn_mv(
                hidden_states=img_tokens,
                image_rotary_emb=mv_emb,
                **joint_attention_kwargs,
            )
            # Restore per-view layout: (B, N*L, C) -> (B*N, L, C)
            attn_output_mv = rearrange(attn_output_mv, 'b (n l) c -> (b n) l c', n=self.num_views)
            attn_output_mv = self.mv_out(attn_output_mv)

            # Blend into image token region
            attn_output[:, NUM_TEXT_TOKENS:-NUM_COND_TOKENS, :] += self.mva_scale * attn_output_mv

        # 3. Combine attention + MLP and apply gating
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states



@maybe_allow_in_graph
class BasicMultiViewTransformerBlock(nn.Module):
    """Wraps ``FluxTransformerBlock`` with optional reference cross-attention.

    When ``use_ra=True``, an additional cross-attention layer (``attn_ref``)
    attends from ``hidden_states`` to ``cond_hidden_states`` (condition features).
    The result is blended back via a zero-initialized linear projection so the
    block starts as an identity w.r.t. the base model. Condition hidden states
    also receive their own residual update stream (reusing ``norm1_context`` and
    ``ff_context`` to save memory).
    """

    def __init__(self, transformer: FluxTransformerBlock, layer_name: str,
                 use_ma: bool = False, use_ra: bool = False):
        super().__init__()

        self.transformer = transformer
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra

        self.num_views = NUM_VIEWS
        self.ref_scale = 1.0

        if use_ma:
            raise NotImplementedError("Multi-view self-attention is not implemented for double blocks")

        if use_ra:
            processor = FluxAttnProcessor2_0()
            self.attn_ref = Attention(
                query_dim=self.transformer.attn.query_dim,
                cross_attention_dim=None,
                added_kv_proj_dim=self.transformer.attn.added_kv_proj_dim,
                dim_head=ATTN_HEAD_DIM,
                heads=self.transformer.attn.heads,
                out_dim=self.transformer.attn.out_dim,
                context_pre_only=False,
                bias=True,
                processor=processor,
                qk_norm="rms_norm",
                eps=1e-6,
            )
            # self.norm1_ref = AdaLayerNormZero(self.transformer.attn.query_dim)
            # self.norm2_ref = nn.LayerNorm(self.transformer.attn.query_dim, elementwise_affine=False, eps=1e-6)
            # self.ff_ref = FeedForward(self.transformer.attn.out_dim, self.transformer.attn.out_dim, activation_fn="gelu-approximate")
            self.ref_out = nn.Linear(self.transformer.attn.out_dim, self.transformer.attn.out_dim)
            self._initialize_ref_weights()

    def _initialize_ref_weights(self):
        """Copy base attention weights; small random init for output projection."""
        self.attn_ref.load_state_dict(self.attn.state_dict())
        nn.init.normal_(self.ref_out.weight, std=0.02)
        nn.init.zeros_(self.ref_out.bias)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_hidden_states: Optional[torch.Tensor] = None,
        cond_image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}

        # --- AdaLN for hidden_states and encoder_hidden_states ---
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 1. Joint cross-attention (image <-> text)
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # 2. Reference cross-attention (hidden_states <-> cond_hidden_states)
        if self.use_ra:
            # Reuse context norm for condition stream (saves memory vs dedicated AdaLN)
            norm_cond_hidden_states, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = self.norm1_context(
                cond_hidden_states, emb=temb
            )
            # Build RoPE: [cond_tokens, image_tokens] (skip text tokens from image_rotary_emb)
            expanded_image_rotary_emb = tuple(
                torch.cat([cond_image_rotary_emb[i], image_rotary_emb[i][NUM_TEXT_TOKENS:]], dim=0)
                for i in range(2)
            )

            attn_output_ref, cond_attn_output = self.attn_ref(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_cond_hidden_states,
                image_rotary_emb=expanded_image_rotary_emb,
                **joint_attention_kwargs,
            )
            attn_output_ref = self.ref_out(attn_output_ref)
            attn_output = attn_output + self.ref_scale * attn_output_ref

        # --- Process hidden_states residual stream ---
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # --- Process encoder_hidden_states residual stream ---
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # --- Process cond_hidden_states residual stream (reuses context norm/ff) ---
        if self.use_ra:
            cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
            cond_hidden_states = cond_hidden_states + cond_attn_output

            norm_cond_hidden_states = self.norm2_context(cond_hidden_states)
            norm_cond_hidden_states = norm_cond_hidden_states * (1 + cond_scale_mlp[:, None]) + cond_shift_mlp[:, None]
            cond_ff_output = self.ff_context(norm_cond_hidden_states)
            cond_hidden_states = cond_hidden_states + cond_gate_mlp.unsqueeze(1) * cond_ff_output

            if cond_hidden_states.dtype == torch.float16:
                cond_hidden_states = cond_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class CustomFluxTransformer2DModel(
    ModelMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin
): # ModelMixin for LoRA
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `19`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `38`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `4096`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _lora_loadable_modules = ["transformer"]
    _no_split_modules = ["BasicMultiViewTransformerBlock", "BasicMultiViewSingleTransformerBlock", "FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["ccm_embed", "pos_embed", "norm"]

    def __init__(self, transformer: FluxTransformer2DModel, pbr_mode=False):
        """
            double_layers: [0, 18]
            single_layers: [0, 37]
        """
        super().__init__()
        
        self.transformer = transformer
        self.ccm_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56)) # (42, 42, 44) -> (1536, 128)
        self.pbr_mode = pbr_mode
        # logger.info(f"FluxMVPipeline PBR Mode: {pbr_mode}")
        self.init_attention(self.transformer, use_ma=False, use_ra=False)
        
        if pbr_mode:
            self.register_parameter('learned_al_token', nn.Parameter(torch.zeros(1, 128, 4096, dtype=self.transformer.dtype)))
            self.register_parameter('learned_mr_token', nn.Parameter(torch.zeros(1, 128, 4096, dtype=self.transformer.dtype)))
            if pbr_mode == "MR":
                self.dino_embedder = nn.Linear(768, 3072, bias=True, dtype=self.transformer.dtype, device=self.transformer.device)
                logger.info("[MR] DINOv2 embedder initialized")

    def fuse_lora(
        self,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        components: List[str] = ["transformer"],
        **kwargs,
    ):
        # https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/loaders/lora_pipeline.py
        # override to fix the huggingface/diffusers bug, reorder the parameters
        # c.f. https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/loaders/lora_base.py#L603
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """

        transformer = getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
        if (
            hasattr(transformer, "_transformer_norm_layers")
            and isinstance(transformer._transformer_norm_layers, dict)
            and len(transformer._transformer_norm_layers.keys()) > 0
        ):
            logger.info(
                "The provided state dict contains normalization layers in addition to LoRA layers. The normalization layers will be directly updated the state_dict of the transformer "
                "as opposed to the LoRA layers that will co-exist separately until the 'fuse_lora()' method is called. That is to say, the normalization layers will always be directly "
                "fused into the transformer and can only be unfused if `discard_original_layers=True` is passed."
            )
        
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )
        
    @property
    def attn_processors(self):
        """
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        processors = {}
        
        # traverse all transformer_blocks
        for i, block in enumerate(self.transformer.transformer_blocks):
            self._collect_block_processors(block, f"transformer_blocks.{i}", processors)
        
        # traverse all single_transformer_blocks
        for i, block in enumerate(self.transformer.single_transformer_blocks):
            self._collect_block_processors(block, f"single_transformer_blocks.{i}", processors)
        
        return processors
    
    def _collect_block_processors(self, block, prefix, processors):
        """recursive collect all attention processors in a block"""
        def collect_recursive(module, current_prefix):
            if hasattr(module, "get_processor") and callable(getattr(module, "get_processor")):
                processors[f"{current_prefix}.processor"] = module.get_processor()
            
            for name, child in module.named_children():
                collect_recursive(child, f"{current_prefix}.{name}")
        
        collect_recursive(block, prefix)
    
    def set_attn_processor(self, processor):
        """
        Sets the attention processor to use to compute attention.
        """
        count = len(self.attn_processors.keys())
        
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )
        
        # set processor for all transformer_blocks
        for i, block in enumerate(self.transformer.transformer_blocks):
            self._set_block_processors(block, f"transformer_blocks.{i}", processor)
        
        # set processor for all single_transformer_blocks
        for i, block in enumerate(self.transformer.single_transformer_blocks):
            self._set_block_processors(block, f"single_transformer_blocks.{i}", processor)
    
    def _set_block_processors(self, block, prefix, processor):
        """recursive set all attention processors in a block"""
        def set_recursive(module, current_prefix):
            if hasattr(module, "set_processor") and callable(getattr(module, "set_processor")):
                if isinstance(processor, dict):
                    module.set_processor(processor[f"{current_prefix}.processor"])
                else:
                    module.set_processor(processor)
            
            for name, child in module.named_children():
                set_recursive(child, f"{current_prefix}.{name}")
        
        set_recursive(block, prefix)
    
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        strict = kwargs.pop('strict', False)
        torch_dtype = kwargs.pop('torch_dtype', torch.bfloat16)
        adapter_name = kwargs.pop('adapter_name', None)
        transformer_lora_config = kwargs.pop('transformer_lora_config', None)
        pbr_mode = kwargs.pop('pbr_mode', False)
        is_ckpt = "checkpoint" in pretrained_model_name_or_path
        
        # load sclice index
        index_path = os.path.join(pretrained_model_name_or_path, 'diffusion_pytorch_model.safetensors.index.json')
        single_file_path = os.path.join(pretrained_model_name_or_path, 'diffusion_pytorch_model.safetensors')
        
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        transformer = FluxTransformer2DModel(**config)
        transformer = CustomFluxTransformer2DModel(transformer, pbr_mode=pbr_mode)
        
        if adapter_name is not None:
            transformer.add_adapter(transformer_lora_config, adapter_name=adapter_name)
        
        from safetensors.torch import load_file
        
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            shard_files = set(index['weight_map'].values())
            
            transformer_ckpt = {}
            for shard_file in shard_files:
                shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
                shard = load_file(shard_path, device='cpu')
                transformer_ckpt.update(shard)
        else:
            transformer_ckpt = load_file(single_file_path, device='cpu')
        
        if strict:
            transformer.load_state_dict(transformer_ckpt, strict=True)
        else:
            missing_keys, unexpected_keys = transformer.load_state_dict(transformer_ckpt, strict=False)
            # Filter out any keys containing "attn_mv" from missing_keys
            # missing_keys = [k for k in missing_keys if "lora" not in k]
            # if missing_keys:
            #     logger.warning(f"Missing keys: {missing_keys}")
            # if unexpected_keys:
            #     logger.warning(f"Unexpected keys: {unexpected_keys}")
            
        transformer = transformer.to(dtype=torch_dtype)
        return transformer

    @property
    def dtype(self):
        """Return the dtype of the transformer."""
        return self.transformer.dtype

    def init_attention(self, transformer, use_ma=False, use_ra=False):
        """Replace vanilla Flux blocks with multi-view aware wrappers."""
        for i, block in enumerate(transformer.transformer_blocks):
            if isinstance(block, FluxTransformerBlock):
                transformer.transformer_blocks[i] = BasicMultiViewTransformerBlock(
                    block, f'transformer_{i}', use_ma=False, use_ra=False,
                )

        for i, block in enumerate(transformer.single_transformer_blocks):
            if isinstance(block, FluxSingleTransformerBlock):
                transformer.single_transformer_blocks[i] = BasicMultiViewSingleTransformerBlock(
                    block, f'single_{i}', use_ma=False, use_ra=False,
                )

    # ============ Shaded-Branch Feature (KV sharing across inference modes) ============
    #
    # Usage sketch (in pipeline.denoise, inside the denoising loop):
    #
    #   for i, t in enumerate(timesteps):
    #       # Step 1: shaded pass — cache K/V from every attention layer
    #       with self.transformer.shaded_branch_ctx("store"):
    #           shaded_pred = self.transformer(hidden_states=..., infer_mode="RGB", ...)
    #
    #       # Step 2: albedo pass — inject cached K/V (replaces own K/V)
    #       with self.transformer.shaded_branch_ctx("inject", branch_scale=0.8):
    #           albedo_pred = self.transformer(hidden_states=..., infer_mode="AL", ...)
    #
    #       # Use albedo_pred for the scheduler step, shaded_pred is discarded.
    #       latents = self.scheduler.step(albedo_pred, t, latents, ...).prev_sample

    def _shaded_branch_install_processors(self):
        """Replace all attention processors with ShadedBranchAttnProcessor instances."""
        if getattr(self, '_shaded_branch_processors', None) is not None:
            return  # already installed
        self._shaded_branch_processors: List[ShadedBranchAttnProcessor] = []
        self._original_processors: Dict[str, Any] = {}

        for name, module in self.named_modules():
            if hasattr(module, 'get_processor') and hasattr(module, 'set_processor'):
                proc = ShadedBranchAttnProcessor()
                self._original_processors[name] = module.get_processor()
                module.set_processor(proc)
                self._shaded_branch_processors.append(proc)

    def _shaded_branch_uninstall_processors(self):
        """Restore original attention processors and discard cache."""
        if getattr(self, '_shaded_branch_processors', None) is None:
            return
        for name, module in self.named_modules():
            if name in self._original_processors:
                module.set_processor(self._original_processors[name])
        for proc in self._shaded_branch_processors:
            proc.clear_cache()
        self._shaded_branch_processors = None
        self._original_processors = None

    def _shaded_branch_set_mode(self, mode: str, branch_scale: float = 1.0):
        """Set mode on all installed ShadedBranchAttnProcessor instances."""
        assert mode in ("off", "store", "inject")
        if self._shaded_branch_processors is None:
            raise RuntimeError("Shaded-branch processors not installed. Call _shaded_branch_install_processors() first.")
        for proc in self._shaded_branch_processors:
            proc.mode = mode
            proc.branch_scale = branch_scale
            proc.reset_counter()

    @contextmanager
    def shaded_branch_ctx(self, mode: str, branch_scale: float = 1.0):
        """Context manager for shaded-branch KV sharing.

        Args:
            mode: ``"store"`` to cache K/V, ``"inject"`` to replace K/V from cache.
            branch_scale: Blend factor for inject mode (0 = own K/V, 1 = cached K/V).
        """
        self._shaded_branch_install_processors()
        self._shaded_branch_set_mode(mode, branch_scale)
        try:
            yield
        finally:
            self._shaded_branch_set_mode("off")

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _precompute_casual_mask(
        self,
        scaled_seq_len: int,
        num_cond_token: int,
        num_ref_token: int,
        num_dino_token: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_block_mask: bool = False,
    ) -> torch.Tensor:
        if not use_block_mask:
            return None

        scaled_block_size = scaled_seq_len - num_cond_token - num_ref_token - num_dino_token
        mask = torch.ones((scaled_seq_len, scaled_seq_len), device=device, dtype=dtype)
        mask[:, :scaled_block_size] = 0
        start, end = scaled_block_size, scaled_block_size + num_cond_token
        mask[start:, start:end] = 0
        start, end = end, end + num_ref_token
        mask[start:, start:end] = 0
        start, end = end, end + num_dino_token
        mask[start:, start:end] = 0
        mask = mask * -1e20
        return mask
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    # https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_flux.py
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        dino_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        mv_ids: torch.Tensor = None,
        num_cond_token: int = 1024,
        num_ref_token: int = 1024,
        guidance: torch.Tensor = None,
        infer_mode: str = "RGB",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        dump_zs: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        
        # update encoder_hidden_states
        if infer_mode == "AL":
            encoder_hidden_states = encoder_hidden_states + self.learned_al_token
        elif infer_mode == "MR":
            encoder_hidden_states = encoder_hidden_states + self.learned_mr_token
            dino_hidden_states = self.dino_embedder(dino_hidden_states)
            hidden_states = torch.cat([hidden_states, dino_hidden_states], dim=1)
            num_ref_token += dino_hidden_states.shape[1]
        
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        ids_2 = torch.cat((txt_ids, mv_ids), dim=0)
        mv_rotary_emb = self.pos_embed(ids_2)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
        
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        zs = (encoder_hidden_states, hidden_states)

        # Reshape for single blocks: merge views and prepend shared text / append shared ref tokens
        # Layout: [text_tokens] + [view_0_latents, view_0_cond, ..., view_N_latents, view_N_cond] + [ref_tokens]
        # The multi-view layer can be applied to any layer in the transformer.
        bsz = hidden_states.shape[0]
        encoder_hidden_states = encoder_hidden_states.mean(dim=0, keepdim=True)
        temb = temb.mean(dim=0, keepdim=True)
        ref_hidden_states = hidden_states[:, -num_ref_token: , :]
        ref_hidden_states = ref_hidden_states.mean(dim=0, keepdim=True)
        hidden_states = hidden_states[:, :-num_ref_token, :]
        hidden_states = hidden_states.reshape(1, -1, hidden_states.shape[-1])
        hidden_states = torch.cat([encoder_hidden_states, hidden_states, ref_hidden_states], dim=1)
        
        # Causual mask for multi-view self-attention
        # casual_mask = self._precompute_casual_mask(
        #     scaled_seq_len=hidden_states.shape[1],
        #     num_cond_token=num_cond_token,
        #     num_ref_token=num_ref_token,
        #     num_dino_token=dino_hidden_states.shape[1] if dino_hidden_states is not None else 0,
        # )
        # joint_attention_kwargs.update({"casual_mask": casual_mask})
        
        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    mv_rotary_emb,
                    mv_rotary_emb,
                    num_ref_token,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=mv_rotary_emb,
                    mv_rotary_emb=mv_rotary_emb,
                    num_ref_token=num_ref_token,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # Extract output: strip text tokens and ref tokens, reshape to batch
        num_text_token = encoder_hidden_states.shape[1]
        hidden_states = hidden_states[:, num_text_token : -num_ref_token, :]
        hidden_states = hidden_states.reshape(bsz, -1, hidden_states.shape[-1])
        hidden_states = hidden_states[:, :-num_cond_token, :]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        if dump_zs:
            if not return_dict:
                return (output, zs)
            return Transformer2DModelOutput(sample=output), zs

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
