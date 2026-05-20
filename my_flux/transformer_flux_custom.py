# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from builtins import super
import inspect
from json import load
import os
from pydoc import text
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# from av import time_base
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import (
    FluxTransformer2DLoadersMixin,
    FromOriginalModelMixin,
    PeftAdapterMixin,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from pathlib import Path
from typing import Optional



import torch
import math
import os
import matplotlib.pyplot as plt
import numpy as np



from config import *
import csv
import numpy as np
import cv2
import torch

def _select_components(mask_np, score_map_np=None,
                       mode="topk_area", k=3, mass_ratio=0.9, min_area=8):
    """
    mask_np: (64,64) uint8 0/1
    score_map_np: (64,64) float32，可用 softmax map，用于按mass排序
    mode:
      - "topk_area": 取面积最大的k个连通块
      - "topk_mass": 取 score_map_np 求和最大的k个连通块
      - "mass_ratio": 按mass从大到小累加，直到覆盖 mass_ratio
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if num <= 1:
        return mask_np, []

    comps = []
    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_mask = (labels == cid)
        if score_map_np is None:
            mass = float(area)
        else:
            mass = float(score_map_np[comp_mask].sum())
        comps.append({"cid": cid, "area": area, "mass": mass})

    if len(comps) == 0:
        return np.zeros_like(mask_np), []

    if mode == "topk_area":
        comps.sort(key=lambda x: x["area"], reverse=True)
        keep = comps[:k]
    elif mode == "topk_mass":
        comps.sort(key=lambda x: x["mass"], reverse=True)
        keep = comps[:k]
    elif mode == "mass_ratio":
        comps.sort(key=lambda x: x["mass"], reverse=True)
        keep = []
        total = sum(c["mass"] for c in comps) + 1e-12
        acc = 0.0
        for c in comps:
            keep.append(c)
            acc += c["mass"]
            if acc / total >= mass_ratio:
                break
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = np.zeros_like(mask_np)
    for c in keep:
        out[labels == c["cid"]] = 1

    return out, keep

def build_continuous_mask_multi(attn_2d: torch.Tensor,
                                pre_thr=0.6,          # ✅ softmax前阈值
                                blur_sigma=0.9,
                                close_ks=3,
                                open_ks=0,
                                fill_holes=True,
                                # ✅ 多块选择策略
                                keep_mode="topk_mass",  # "topk_area" | "topk_mass" | "mass_ratio"
                                k=3,
                                mass_ratio=0.9,
                                min_area=8,
                                soft_edge=True,
                                soft_edge_sigma=1.2,
                                eps=1e-8):
    device = attn_2d.device
    x = attn_2d.detach().float()

    # reshape to (64,64)
    if x.dim() == 1:
        assert x.numel() == 4096
        x2 = x.view(64, 64)
    else:
        assert x.shape[0] * x.shape[1] == 4096
        x2 = x

    # 1) pre-softmax threshold (hard safety gate)
    pre_mask = (x2 > pre_thr)
    if pre_mask.sum().item() == 0:
        z = torch.zeros_like(x2)
        return z, z, z, z, []

    # 2) masked softmax over flattened 2D
    neg_inf = torch.finfo(x2.dtype).min
    x_masked = torch.where(pre_mask, x2, torch.tensor(neg_inf, device=device, dtype=x2.dtype))
    attn_sm = torch.softmax(x_masked.view(-1), dim=0).view(64, 64)  # (0,1), sum=1 over candidates

    # 3) start from pre_mask for morphology (do NOT introduce <pre_thr pixels)
    mask_np = pre_mask.detach().cpu().numpy().astype(np.uint8)

    # open/close to remove speckles & connect near parts
    if open_ks and open_ks > 1:
        k0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, k0, iterations=1)
    if close_ks and close_ks > 1:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, k1, iterations=1)

    # fill holes (optional)
    if fill_holes:
        h, w = mask_np.shape
        flood = mask_np.copy()
        m = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(flood, m, (0, 0), 2)
        holes = (flood == 0).astype(np.uint8)
        mask_np = np.clip(mask_np + holes, 0, 1).astype(np.uint8)

    # 4) ✅ select multiple components (no longer largest only)
    score_np = attn_sm.detach().cpu().numpy().astype(np.float32)
    mask_np, kept_info = _select_components(
        mask_np,
        score_map_np=score_np,
        mode=keep_mode,
        k=k,
        mass_ratio=mass_ratio,
        min_area=min_area
    )

    mask_hard = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)

    # 5) build soft mask: blur softmax map + apply hard mask + optional edge soften
    soft_np = score_np.copy()
    if blur_sigma and blur_sigma > 0:
        soft_np = cv2.GaussianBlur(soft_np, (0, 0), blur_sigma)

    soft_np *= mask_np.astype(np.float32)

    if soft_edge:
        dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, 3)
        dist = dist / (dist.max() + 1e-6)
        edge = 1.0 - np.exp(-(dist**2) / (2*(soft_edge_sigma**2)))
        soft_np *= edge.astype(np.float32)

    if soft_np.max() > 0:
        soft_np /= (soft_np.max() + eps)

    mask_soft = torch.from_numpy(soft_np).to(device=device, dtype=torch.float32)

    return mask_hard, mask_soft, pre_mask.float(), attn_sm, kept_info

empty_device="cuda:0"

def sam_scaled_dot_product_attention_visualized(
    query, key, value, 
    prefix="attn",
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    single_flag=False,
    scale=None,
    idx=None,
    timestep=None,
    layer_id=None,
):
    device_prev = query.device
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    
    B, H, L, D = query.shape

    IMG = 4096
    TXT = 512
    SIDE = 64

    # -------------------------
    # 1. attention score
    # -------------------------
    timestep = timestep.to(empty_device)
    query = query.to(empty_device)
    key = key.to(empty_device)
    value = value.to(empty_device)
    scale_factor = 1 / math.sqrt(D) if scale is None else scale
    attn_scores = torch.einsum("b h q d, b h k d -> b h q k", query, key) * scale_factor
    if attn_mask is not None:
        attn_scores += attn_mask

    attn = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0:
        attn = torch.dropout(attn, dropout_p, True)

    global REDUCE_IDXs
    out = torch.einsum("b h q k, b h k d -> b h q d", attn, value)
    if True and single_flag and layer_id==10 and int(timestep)>850:
        key_tensor = None # need to compute
        if key_tensor:
            attn_perturb = torch.einsum("b h q d, b h d k -> b h q k", out[:,:,TXT:,:], key_tensor) * scale_factor
            attn_perturb = attn_perturb.mean(dim=0).mean(dim=0).mean(-1)
            mask_hard, mask_soft, pre_mask, attn_sm, kept = build_continuous_mask_multi(
                attn_perturb,
                pre_thr=threshold,
                close_ks=3,
                keep_mode="mass_ratio",
                mass_ratio=0.3,
                min_area=5
            )
            REDUCE_IDXs = (mask_hard.reshape(-1) > 0).nonzero(as_tuple=True)[0].tolist()
            REDUCE_IDXs=REDUCE_IDXs[:2000]
  
    for idx in REDUCE_IDXs:
        out[:,:, TXT+idx, :] = out[:,:, TXT+idx, :]*0.001+ torch.randn_like(out[:,:, TXT+idx, :])*(1-0.001)
    out = out.permute(0, 2, 1, 3)
    out = out.to(device_prev)
    return out, attn


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None, subject_embeds=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    subject_query = subject_key = subject_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)
    if subject_embeds is not None and attn.added_kv_proj_dim is not None:
        subject_query = attn.add_q_proj(subject_embeds)
        subject_key = attn.add_k_proj(subject_embeds)
        subject_value = attn.add_v_proj(subject_embeds)

    return query, key, value, encoder_query, encoder_key, encoder_value, subject_query, subject_key, subject_value

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    # print("stop")
    # raise ValueError(f"sequence_dim= but should be 1 or 2.")
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

def _get_fused_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None, subject_embeds=None
):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    subject_query = subject_key = subject_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(
            encoder_hidden_states
        ).chunk(3, dim=-1)
        subject_query, subject_key, subject_value = attn.to_added_qkv(
            subject_embeds
        ).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value, subject_query, subject_key, subject_value


def _get_qkv_projections(
    attn: "FluxAttention", hidden_states, encoder_hidden_states=None, subject_embeds=None
):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states, subject_embeds)
    return _get_projections(attn, hidden_states, encoder_hidden_states, subject_embeds)

class FluxAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        index_block: int,
        single_flag: bool = False,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        PERTURB_FLAG=False,
        save_anchor_flag=False,
    ) -> torch.Tensor:
        # print("shape of encoder_hidden_states:", None if encoder_hidden_states is None else encoder_hidden_states.shape)
        query, key, value, encoder_query, encoder_key, encoder_value, subject_query, subject_key, subject_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states, subject_embeds)
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            if subject_query is not None:
                subject_query = subject_query.unflatten(-1, (attn.heads, -1))
                subject_key = subject_key.unflatten(-1, (attn.heads, -1))
                subject_value = subject_value.unflatten(-1, (attn.heads, -1))
                subject_query = attn.norm_added_q(subject_query)
                subject_key = attn.norm_added_k(subject_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
            if subject_query is not None:
                subject_key = torch.cat([subject_key, key], dim=1)
                subject_value = torch.cat([subject_value, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)


        if PERTURB_FLAG and (timestep>=last_step):
            if not single_flag:
                hidden_states, attn_map = sam_scaled_dot_product_attention_visualized(
                    query, key, value,
                    timestep=timestep,
                    prefix="attn",
                    layer_id=index_block,
                    single_flag=False
                )
            else:
                hidden_states, attn_map = sam_scaled_dot_product_attention_visualized(
                    query, key, value,
                    timestep=timestep,
                    prefix="attn",
                    layer_id=index_block,
                    single_flag=True
                )
        else:
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, backend=self._attention_backend
            )
            subject_embeds = dispatch_attention_fn(
                subject_query, subject_key, subject_value, attn_mask=attention_mask, backend=self._attention_backend
            ) if subject_query is not None else None
        if save_anchor_flag and timestep==104 and index_block==10 and single_flag:
            print(f"Saving hidden_states at timestep {int(timestep)}...")
            if  single_flag:
                print(f"Saving hidden_states at timestep {int(timestep)} for unsafe image...")
                save_dir = f"FLUX_KONTEXT/out_single/unsafe_img/block_{index_block}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    hidden_states[:,512:4608,:,:].cpu(),
                    f"{save_dir}/time_{int(timestep)}_out.pt",
                )
        
        
        hidden_states = hidden_states.flatten(2, 3) # (B, seq_len, heads, head_dim) -> (B, seq_len, hidden_dim)
        hidden_states = hidden_states.to(query.dtype)
        if subject_embeds is not None:
            subject_embeds = subject_embeds.flatten(2, 3).to(subject_query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if subject_embeds is not None:
                subject_embeds = attn.to_add_out(subject_embeds)

            return (hidden_states, encoder_hidden_states), subject_embeds
        else:
            return hidden_states


class FluxIPAdapterAttnProcessor(torch.nn.Module):
    """Flux Attention processor for IP-Adapter."""

    _attention_backend = None

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens=(4,),
        scale=1.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError(
                "`scale` should be a list of integers with the same length as `num_tokens`."
            )
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [
                nn.Linear(
                    cross_attention_dim,
                    hidden_size,
                    bias=True,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(len(num_tokens))
            ]
        )
        self.to_v_ip = nn.ModuleList(
            [
                nn.Linear(
                    cross_attention_dim,
                    hidden_size,
                    bias=True,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(len(num_tokens))
            ]
        )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_attn_output = torch.zeros_like(hidden_states)

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                current_ip_hidden_states = dispatch_attention_fn(
                    ip_query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                    backend=self._attention_backend,
                )
                current_ip_hidden_states = current_ip_hidden_states.reshape(
                    batch_size, -1, attn.heads * attn.head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


class FluxAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = FluxAttnProcessor
    _available_processors = [
        FluxAttnProcessor,
        FluxIPAdapterAttnProcessor,
    ]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        context_pre_only: Optional[bool] = None,
        pre_only: bool = False,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = torch.nn.RMSNorm(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.pre_only:
            self.to_out = torch.nn.ModuleList([])
            self.to_out.append(
                torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
            )
            self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        index_block: int,
        single_flag: bool = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
        perturb_flag: bool = False,
        save_anchor_flag: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k
            for k, _ in kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states = hidden_states,
            index_block = index_block,
            single_flag = single_flag,
            encoder_hidden_states = encoder_hidden_states,
            attention_mask = attention_mask,
            timestep = timestep,
            save_anchor_flag = save_anchor_flag,
            perturb_flag = perturb_flag,
            image_rotary_emb = image_rotary_emb,
            **kwargs,
        )


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        idx: Optional[int] = None,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            from ..attention_processor import FluxAttnProcessor2_0_NPU

            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor()

        self.attn = FluxAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        index_block: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        perturb_flag: bool = False,
        timestep: Optional[int] = None,
        save_anchor_flag: bool = False,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states

        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            index_block=index_block,
            single_flag=True,
            hidden_states=norm_hidden_states,
            save_flag=save_anchor_flag,
            timestep=timestep,
            perturb_flag=perturb_flag,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states
        
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        # del residual, norm_hidden_states, mlp_hidden_states, attn_output, gate
        return encoder_hidden_states, hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = FluxAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor(),
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        index_block: int,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        perturb_flag: bool = False,
        save_anchor_flag: bool = False,
        timestep: Optional[int] = None,
        
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        norm_subject_embeds = None
        if subject_embeds is not None:
            norm_subject_embeds, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
                self.norm1_context(subject_embeds, emb=temb)
            )
            
        attention_outputs = None


        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.

        attention_outputs, sub_output = self.attn(
            index_block=index_block,
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            subject_embeds=norm_subject_embeds,
            image_rotary_emb=image_rotary_emb,
            timestep=timestep,
            perturb_flag=perturb_flag,
            save_anchor_flag=save_anchor_flag,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states.to(attn_output.device)
        encoder_hidden_states = encoder_hidden_states.to(attn_output.device)
        # hidden_states = hidden_states + attn_output

        hidden_states = hidden_states + attn_output
        # Process attention outputs for the `encoder_hidden_states`
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        if sub_output is not None:
            sub_output = s_gate_msa.unsqueeze(1) * sub_output
            subject_embeds = subject_embeds.to(sub_output.device)
            subject_embeds = subject_embeds + sub_output

        encoder_hidden_states = encoder_hidden_states + context_attn_output
                
        norm_hidden_states = self.norm2(hidden_states)
        if norm_hidden_states.device!=scale_mlp.device:
                norm_hidden_states = norm_hidden_states.to(scale_mlp.device)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        if hidden_states.device!=ff_output.device:
            hidden_states=hidden_states.to(ff_output.device)
        hidden_states = hidden_states + ff_output
        if attention_outputs is not None and len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        if c_scale_mlp.device!=norm_encoder_hidden_states.device:
            c_scale_mlp=c_scale_mlp.to(norm_encoder_hidden_states.device)
            c_shift_mlp=c_shift_mlp.to(norm_encoder_hidden_states.device)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if encoder_hidden_states.device!=c_gate_mlp.device:
            encoder_hidden_states=encoder_hidden_states.to(c_gate_mlp.device)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if subject_embeds is not None:
            norm_subject_embeds = self.norm2_context(subject_embeds)        
            if s_scale_mlp.device!=norm_subject_embeds.device:
                s_scale_mlp=s_scale_mlp.to(norm_subject_embeds.device)
                s_shift_mlp=s_shift_mlp.to(norm_subject_embeds.device)
            norm_subject_embeds = (
                norm_subject_embeds * (1 + s_scale_mlp[:, None])
                + s_shift_mlp[:, None]
            )
            subject_ff_output = self.ff_context(norm_subject_embeds)
            if subject_embeds.device!=s_gate_mlp.device:
                subject_embeds=subject_embeds.to(s_gate_mlp.device)
            subject_embeds = subject_embeds + s_gate_mlp.unsqueeze(1) * subject_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states, subject_embeds


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class FluxTransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
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
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        orig_img_ids: torch.Tensor = None,
        orig_txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        perturb_flag: bool = False,
        save_anchor_flag: bool = False,
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
        # print("FluxTransformer2DModel forward called")
        # apply_rotary_emb(torch.zeros(1,1,1,1), torch.zeros(1,1), sequence_dim=1)
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if subject_embeds is not None:
            subject_embeds = self.context_embedder(subject_embeds)

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
        # print("check img_ids:", img_ids[10])
        image_rotary_emb = self.pos_embed(ids)
        
        orig_ids = torch.cat((orig_txt_ids, orig_img_ids), dim=0)
        orig_image_rotary_emb = self.pos_embed(orig_ids)
        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )

            if ip_adapter_image_embeds[0].device != next(
                self.encoder_hid_proj.parameters()
            ).device:
                ip_adapter_image_embeds = [
                    embed.to(next(self.encoder_hid_proj.parameters()).device)
                    for embed in ip_adapter_image_embeds
                ]
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        double_index = []
        single_index = []
        global REDUCE_IDXs
        if int(timestep) == 1000:
            REDUCE_IDXs = []
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        orig_image_rotary_emb,
                        joint_attention_kwargs,
                    )
                )

            else:
                if index_block not in double_index:
                    encoder_hidden_states, hidden_states, subject_embeds = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        subject_embeds=subject_embeds,
                        temb=temb,
                        timestep=timestep,
                        perturb_flag=perturb_flag,
                        save_anchor_flag=save_anchor_flag,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    # print("shape of encoder_hidden_states:", encoder_hidden_states.shape)
                    encoder_hidden_states, hidden_states, subject_embeds = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        subject_embeds=subject_embeds,
                        temb=temb,
                        timestep=timestep,
                        perturb_flag=perturb_flag,
                        save_anchor_flag=save_anchor_flag,
                        image_rotary_emb=orig_image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[
                            index_block % len(controlnet_block_samples)
                        ]
                    )
                else:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block // interval_control]
                    )

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        orig_image_rotary_emb,
                        joint_attention_kwargs,
                    )
                )

            else:
                if index_block not in single_index:
                    encoder_hidden_states, hidden_states = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        timestep=timestep,
                        perturb_flag=perturb_flag,
                        save_anchor_flag=save_anchor_flag,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        index_block=index_block,
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        timestep=timestep,
                        perturb_flag=perturb_flag,
                        save_anchor_flag=save_anchor_flag,
                        image_rotary_emb=orig_image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states = (
                    hidden_states
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
from collections.abc import Mapping

