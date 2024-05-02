import math
import utility
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

def double_conv(in_channels, mid_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class down_conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, mid_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class up_conv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = double_conv(in_channels, mid_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = double_conv(in_channels, mid_channels, out_channels)

    def forward(self, latent, skip):
        latent = self.up(latent)
        x = torch.cat([skip, latent], dim=1)
        return self.conv(x)