from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
# from flash_attn_interface import flash_attn_func

class NativeFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        index_ref_num: Optional[int] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class MultiViewFluxAttnProcessor2_0:
    """Highly optimized attention processor with automatic state management."""

    def __init__(self, domain_token_num=128, num_views=6):
        self.domain_token_num = domain_token_num
        self.num_views = num_views
        
        # Pre-compute static values
        self.num_views_plus_one = num_views + 1
        self.domain_indices = None  # Will be set on first call
        
        # Use a hash to track if input configuration has changed
        self._config_hash = None
        self._cached_indices = {}

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _compute_config_hash(self, hidden_states, index_ref_num):
        """Compute hash of current configuration."""
        return hash((
            hidden_states.shape[1],  # sequence length
            hidden_states.device,
            index_ref_num,
            self.domain_token_num,
            self.num_views
        ))

    def _ensure_indices_cached(self, attn, hidden_states, index_ref_num):
        """Ensure indices are cached for current configuration."""
        config_hash = self._compute_config_hash(hidden_states, index_ref_num)
        
        if config_hash != self._config_hash:
            self._config_hash = config_hash
            self._initialize_constants(attn, hidden_states, index_ref_num)

    def _initialize_constants(self, attn, hidden_states, index_ref_num):
        """Initialize constants that depend on input dimensions."""
        device = hidden_states.device
        seq_len = hidden_states.shape[1]
        
        # Cache attention parameters
        self.heads = attn.heads
        self.inner_dim = hidden_states.shape[-1]  # Assuming QKV have same dim
        self.head_dim = self.inner_dim // self.heads
        
        # Pre-compute latent token calculations
        latent_token_num = seq_len - self.domain_token_num - index_ref_num
        assert latent_token_num % 2 == 0
        self.latent_token_num = latent_token_num // 2
        assert self.latent_token_num % self.num_views == 0
        self.latent_token_num_per_view = self.latent_token_num // self.num_views
        
        # Pre-compute all indices
        self.domain_indices = torch.arange(self.domain_token_num, device=device)
        self.index_kv_ref = torch.arange(index_ref_num, device=device) + seq_len - index_ref_num
        self.index_kv_mv = torch.arange(self.domain_token_num + self.latent_token_num, device=device)
        
        # Pre-compute view-specific indices
        self.index_q_list = []
        self.index_kv_list = []
        
        for i in range(self.num_views):
            view_start = self.domain_token_num + i * self.latent_token_num_per_view
            view_idx = torch.arange(self.latent_token_num_per_view, device=device) + view_start
            
            geo_start = self.domain_token_num + self.latent_token_num + i * self.latent_token_num_per_view
            geo_idx = torch.arange(self.latent_token_num_per_view, device=device) + geo_start
            
            # Pre-compute combined indices
            index_q = torch.cat([view_idx, geo_idx])
            index_kv = torch.cat([self.index_kv_mv, geo_idx, self.index_kv_ref])
            self.index_q_list.append(index_q)
            self.index_kv_list.append(index_kv)
        
        # Residual component indices
        residual_q = torch.cat([self.domain_indices, self.index_kv_ref])
        residual_kv = torch.cat([self.index_kv_mv, self.index_kv_ref])
        self.index_q_list.append(residual_q)
        self.index_kv_list.append(residual_kv)
        
        # # Pre-compute reshape parameters
        # self.qkv_shape = (-1, self.heads, self.head_dim)
        # self.output_shape = (-1, self.inner_dim)
        
        # self._initialized = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        index_ref_num: Optional[int] = None,
    ) -> torch.FloatTensor:
        
        self._ensure_indices_cached(attn, hidden_states, index_ref_num)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Pre-compute QKV projections once
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        # Reshape QKV once for all views (more efficient than per-view)
        query = query.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Apply normalization once
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Pre-apply rotary embeddings if present (more efficient than per-view)
        if image_rotary_emb is not None:
            # Apply to full tensors once, then slice
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # Process all views in a more efficient way
        # Use advanced indexing to avoid creating intermediate tensors
        output_hidden_states = hidden_states.clone()  # Start with original
        
        for i in range(self.num_views_plus_one):
            index_q = self.index_q_list[i]
            index_kv = self.index_kv_list[i]
            
            # Use advanced indexing (more efficient than slicing)
            view_query = query[:, :, index_q, :]
            view_key = key[:, :, index_kv, :]
            view_value = value[:, :, index_kv, :]
            
            # Compute attention
            view_hidden_states = F.scaled_dot_product_attention(
                view_query, view_key, view_value, 
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            
            # Reshape and assign back (in-place operation)
            view_hidden_states = view_hidden_states.transpose(1, 2).reshape(batch_size, len(index_q), self.inner_dim)
            output_hidden_states[:, index_q, :] = view_hidden_states
        
        return output_hidden_states


class ShadedBranchAttnProcessor:
    """Drop-in replacement for ``FluxAttnProcessor`` that can store or inject K/V.

    Modes (controlled by ``self.mode``):
        ``"off"``   — passthrough, identical to the default processor.
        ``"store"`` — run attention normally AND save post-RoPE K/V into ``self.cache``.
        ``"inject"``— compute Q from current hidden_states but **replace** K/V with
                      the cached values from a prior "store" pass.  An optional
                      ``branch_scale`` (0–1) blends between own K/V and cached K/V.

    The cache is a ``dict[int, (Tensor, Tensor)]`` keyed by a counter that
    increments each time ``__call__`` is invoked within one forward pass.
    Call ``reset_counter()`` before each forward pass to re-align the counter.
    """

    _attention_backend = None  # use default SDPA

    def __init__(self, branch_scale: float = 1.0):
        self.mode: str = "off"
        self.branch_scale = branch_scale
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._counter: int = 0

    def reset_counter(self):
        self._counter = 0

    def clear_cache(self):
        self.cache.clear()
        self._counter = 0

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        idx = self._counter
        self._counter += 1

        # --- project Q / K / V ---
        query, key, value, enc_query, enc_key, enc_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.added_kv_proj_dim is not None:
            enc_query = attn.norm_added_q(enc_query.unflatten(-1, (attn.heads, -1)))
            enc_key = attn.norm_added_k(enc_key.unflatten(-1, (attn.heads, -1)))
            enc_value = enc_value.unflatten(-1, (attn.heads, -1))
            query = torch.cat([enc_query, query], dim=1)
            key = torch.cat([enc_key, key], dim=1)
            value = torch.cat([enc_value, value], dim=1)

        # --- apply RoPE ---
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # --- store / inject ---
        if self.mode == "store":
            self.cache[idx] = (key.detach(), value.detach())

        elif self.mode == "inject" and idx in self.cache:
            cached_key, cached_value = self.cache[idx]
            s = self.branch_scale
            if s >= 1.0:
                key, value = cached_key, cached_value
            else:
                key = (1 - s) * key + s * cached_key
                value = (1 - s) * value + s * cached_value

        # --- scaled dot-product attention ---
        out = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        )
        out = out.flatten(2, 3).to(query.dtype)

        # --- output projection ---
        if encoder_hidden_states is not None:
            encoder_out, out = out.split_with_sizes(
                [encoder_hidden_states.shape[1], out.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)
            encoder_out = attn.to_add_out(encoder_out)
            return out, encoder_out
        else:
            return out