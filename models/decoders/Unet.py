import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class UnetStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_channels: int,
                 mid_channels: int = None,
                 up: bool = True,
                 batch_norm: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        self.batch_norm = batch_norm
        self.up = up
        if mid_channels is None:
            self.mid_channels = self.in_channels

        self.conv1 = nn.Conv2d(self.in_channels + skip_channels,
                               self.mid_channels, kernel_size=3, padding=1, bias=False)
        if up:
            self.conv2 = nn.Conv2d(self.mid_channels, self.out_channels * 2, kernel_size=3, padding=1, bias=False)
            self.up = nn.ConvTranspose2d(self.out_channels * 2, self.out_channels, kernel_size=2, stride=2)
        else:
            self.conv2 = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, padding=1, bias=False)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(self.mid_channels)
            if self.up:
                self.bn2 = nn.BatchNorm2d(self.out_channels * 2)
            else:
                self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if self.up:
            x = self.relu2(x)
        if self.up:
            x = self.up(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 skip_channels: List[int],
                 n_blocks: int = 5,
                 batch_norm: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.batch_norm = batch_norm
        self.main_stages = []
        self.stage1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        t_channels = in_channels // 2
        for i in range(n_blocks - 1):

            self.main_stages.append(UnetStage(in_channels=t_channels,
                                              out_channels=t_channels//2,
                                              skip_channels=skip_channels[i], ## reversed
                                              batch_norm=batch_norm))
            t_channels = t_channels // 2


        self.main_stages = nn.ModuleList(self.main_stages)

        self.last_stage = UnetStage(in_channels=t_channels,
                                    out_channels=out_channels,
                                    skip_channels=0, ## reversed
                                    batch_norm=batch_norm,
                                    up=False)

    def forward(self, dec_input):
        dec_input = dec_input[::-1]
        output = self.stage1(dec_input[0])
        for i_stage, stage in enumerate(self.main_stages):
            output = stage(torch.cat([output, dec_input[i_stage + 1]], dim=1))
        output = self.last_stage(output)
        return output
