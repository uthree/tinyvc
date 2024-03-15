import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, groups=groups, dilation=dilation)
        self.pad_size = (kernel_size - 1) * dilation

    def forward(self, x):
        x = F.pad(x, [self.pad_size, 0])
        x = self.conv(x)
        return x
