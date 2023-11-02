"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import diffusers
import torch
import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        self.model = diffusers.AutoencoderKL.from_pretrained(pretrained_path, use_safetensors=True)
        self.model.requires_grad_(False)
        self.model.enable_slicing()

    @torch.no_grad()
    def encode(self, x):
        z = self.model.encode(x).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z
        return z

    @torch.no_grad()
    def decode(self, z):
        z = 1. / self.model.scaling_factor * z
        x = self.model.decode(z).sample
        return x