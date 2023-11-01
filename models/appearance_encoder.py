"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock


class AppearanceEncoder(nn.Module):
    def __init__(self, attn_residual_block_idx, inner_dims, ctx_dims, embed_dims, heads, depth,
                 to_self_attn, to_queries, to_keys, to_values, aspect_ratio, detach_input,
                 convin_kernel_size, convin_stride, convin_padding):
        super().__init__()
        self.attn_residual_block_idx = attn_residual_block_idx
        self.inner_dims = inner_dims
        self.ctx_dims = ctx_dims
        self.embed_dims = embed_dims
        self.to_self_attn = to_self_attn
        self.to_queries = to_queries
        self.to_keys = to_keys
        self.to_values = to_values
        self.aspect_ratio = aspect_ratio
        self.detach_input = detach_input

        self.zero_conv_ins = []
        self.zero_conv_outs = []
        self.blocks = []
        for inner_dim, embed_dim, ctx_dim, num_head in zip(inner_dims, self.embed_dims, self.ctx_dims, heads):
            self.zero_conv_ins.append(nn.Conv2d(inner_dim, embed_dim, kernel_size=convin_kernel_size,
                                                stride=convin_stride, padding=convin_padding))
            self.zero_conv_outs.append(nn.Conv2d(embed_dim, ctx_dim, kernel_size=1, stride=1, padding=0))
            self.blocks.append(nn.Sequential(*[BasicTransformerBlock(
                dim=embed_dim,
                num_attention_heads=num_head,
                attention_head_dim=embed_dim//num_head,
                double_self_attention=True
            ) for _ in range(depth)]))

        self.blocks = nn.ModuleList(self.blocks)
        self.zero_conv_ins = nn.ModuleList(self.zero_conv_ins)
        self.zero_conv_outs = nn.ModuleList(self.zero_conv_outs)

        for n in self.zero_conv_ins.parameters():
            nn.init.zeros_(n)
        for n in self.zero_conv_outs.parameters():
            nn.init.zeros_(n)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, features):
        additional_residuals = {}

        for i, block in enumerate(self.blocks):
            hidden_states = features[0]
            if self.detach_input:
                hidden_states = hidden_states.detach()

            in_H = in_W = int(features[0].shape[1] ** 0.5)
            hidden_states = features[0].permute(0, 2, 1).reshape(-1, self.inner_dims[i], in_H, in_W)
            hidden_states = self.zero_conv_ins[i](hidden_states)
            H = W = hidden_states.shape[2]
            hidden_states = hidden_states.reshape(-1, self.embed_dims[i], H * W).permute(0, 2, 1)

            hidden_states = block(hidden_states)

            hidden_states = hidden_states.permute(0, 2, 1).reshape(-1, self.embed_dims[i], H, W)
            hidden_states = self.zero_conv_outs[i](hidden_states)
            hidden_states = hidden_states.reshape(-1, self.ctx_dims[i], H * W).permute(0, 2, 1)

            if self.to_self_attn:
                if self.to_queries:
                    additional_residuals[f"block_{self.attn_residual_block_idx[i]}_self_attn_q"] = hidden_states
                elif self.to_keys:
                    additional_residuals[f"block_{self.attn_residual_block_idx[i]}_self_attn_k"] = hidden_states
                elif self.to_values:
                    additional_residuals[f"block_{self.attn_residual_block_idx[i]}_self_attn_v"] = hidden_states
            else:
                if self.to_queries:
                    additional_residuals[f"block_{self.attn_residual_block_idx[i]}_cross_attn_q"] = hidden_states
                elif self.to_keys:
                    additional_residuals[f"block_{self.attn_residual_block_idx[i]}_cross_attn_k"] = hidden_states
                elif self.to_values:
                    additional_residuals[f"block_{self.attn_residual_block_idx[i]}_cross_attn_v"] = hidden_states

            if i != len(self.blocks) - 1 and self.inner_dims[i] != self.inner_dims[i + 1]:
                features.pop(0)

        return additional_residuals