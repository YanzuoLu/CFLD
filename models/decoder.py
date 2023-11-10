"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock

from .xf import FrozenCLIPImageEmbedder


class CrossAttnFirstTransformerBlock(BasicTransformerBlock):
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        query_pos: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 1. Cross-Attention
        if self.attn2 is not None:
            hidden_states = hidden_states + query_pos
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 2. Self-Attention
        hidden_states = hidden_states + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, n_ctx, ctx_dim, heads, depth, last_norm, img_size,
                 embed_dim, depths, pose_query, pose_channel):
        super().__init__()
        self.last_norm = last_norm
        self.pose_query = pose_query
        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth

        if self.depth > 0:
            n_layers = len(depths)
            embed_dim = embed_dim * 2 ** (n_layers - 1)

            if not self.pose_query:
                self.query_feat = nn.Parameter(torch.zeros(n_ctx, ctx_dim))
                nn.init.normal_(self.query_feat, std=0.02)
            else:
                self.decoder_fc = nn.Linear(pose_channel, ctx_dim, bias=False)

            self.pos_embed = nn.Parameter(torch.zeros(n_ctx, ctx_dim))
            nn.init.normal_(self.pos_embed, std=0.02)

            self.blocks = []
            for _ in range(depth):
                self.blocks.append(CrossAttnFirstTransformerBlock(
                    dim=ctx_dim,
                    num_attention_heads=heads,
                    attention_head_dim=ctx_dim//heads,
                    cross_attention_dim=embed_dim
                ))
            self.blocks = nn.ModuleList(self.blocks)

            if not self.last_norm:
                H, W = img_size[0] // 32, img_size[1] // 32
                self.kv_pos_embed = nn.Parameter(torch.zeros(1, H*W, embed_dim))
                nn.init.normal_(self.kv_pos_embed, std=0.02)

            # enable xformers
            def fn_recursive_set_mem_eff(module: torch.nn.Module):
                if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                    module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

                for child in module.children():
                    fn_recursive_set_mem_eff(child)

            for module in self.children():
                if isinstance(module, torch.nn.Module):
                    fn_recursive_set_mem_eff(module)
        elif self.depth == 0:
            self.clip_model = FrozenCLIPImageEmbedder()

    def forward(self, x, features, pose_features):
        if self.depth > 0:
            if self.last_norm:
                B, C = x.shape
                encoder_hidden_states = x.unsqueeze(1)
            else:
                B, L, C = features[-1].shape
                encoder_hidden_states = features.pop()
                kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)
                encoder_hidden_states = encoder_hidden_states + kv_pos_embed

            if self.pose_query:
                hidden_states = pose_features.pop()
                if self.training:
                    hidden_states = hidden_states.reshape(B*2, self.pose_channel, -1).permute(0, 2, 1)
                    pos_embed = self.pos_embed.expand(B*2, -1, -1)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states])
                else:
                    hidden_states = hidden_states.reshape(B, self.pose_channel, -1).permute(0, 2, 1)
                    pos_embed = self.pos_embed.expand(B, -1, -1)

                hidden_states = self.decoder_fc(hidden_states)
            else:
                hidden_states = self.query_feat.expand(B, -1, -1)
                pos_embed = self.pos_embed.expand(B, -1, -1)

            for blk in self.blocks:
                hidden_states = blk(hidden_states, pos_embed, encoder_hidden_states=encoder_hidden_states)
            return hidden_states
        elif self.depth == 0:
            x = x * 0.5 + 0.5
            x = x - torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dtype=x.dtype, device=x.device)
            x = x / torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dtype=x.dtype, device=x.device)
            return self.clip_model(x)
        else:
            encoder_hidden_states = features.pop()
            encoder_hidden_states = encoder_hidden_states * 0.
            encoder_hidden_states = encoder_hidden_states.mean(dim=2, keepdim=True).expand(-1, -1, self.ctx_dim)
            return encoder_hidden_states