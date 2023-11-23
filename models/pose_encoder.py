"""
@author: anonymous
@email:  anonymous
"""

import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D, Downsample2D


class PoseEncoder(nn.Module):
    def __init__(self, downscale_factor, pose_channels, in_channels, channels):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(int(pose_channels * (downscale_factor ** 2)), in_channels, kernel_size=1)

        resnets = []
        downsamplers = []
        for i in range(len(channels)):
            in_channels = in_channels if i == 0 else channels[i - 1]
            out_channels = channels[i]

            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=None, # no time embed
            ))
            downsamplers.append(Downsample2D(
                out_channels,
                use_conv=False,
                out_channels=out_channels,
                padding=1,
                name="op"
            ) if i != len(channels) - 1 else nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(self, hidden_states):
        features = []
        hidden_states = self.unshuffle(hidden_states)
        hidden_states = self.conv_in(hidden_states)
        for resnet, downsampler in zip(self.resnets, self.downsamplers):
            hidden_states = resnet(hidden_states, temb=None)
            features.append(hidden_states)
            hidden_states = downsampler(hidden_states)
        return features