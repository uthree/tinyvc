import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# Oscillate harmonic signal
# 基音と倍音を生成
#
# Inputs ---
# f0: [BatchSize, 1, Frames]
#
# frame_size: int
# sample_rate: float or int
# min_frequency: float
# num_harmonics: int
#
# Output: [BatchSize, NumHarmonics+1, Length]
#
# length = Frames * frame_size
def oscillate_harmonics(
        f0,
        frame_size=480,
        sample_rate=24000,
        num_harmonics=0,
        min_frequency=20.0,
    ):
    C = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(C, device=device) + 1).unsqueeze(0).unsqueeze(2)

    # change length to wave's
    fs = F.interpolate(f0, Lw, mode='linear') * mul

    # unvoiced / voiced mask
    uv = (f0 > min_frequency).to(torch.float)
    uv = F.interpolate(uv, Lw, mode='linear')

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    theta = 2 * math.pi * (I % 1) # convert to radians
    harmonics = torch.sin(theta) * uv

    return harmonics.to(device)


class FiLM(nn.Module):
    def __init__(self, in_channels, condition_channels):
        super().__init__()
        self.to_shift = nn.Conv1d(condition_channels, in_channels, 1)
        self.to_scale = nn.Conv1d(condition_channels, in_channels, 1)

    # x: [BatchSize, in_channels, Length]
    # c: [BatchSize, condition_channels, Length]
    def forward(self, x, c):
        shift = self.to_shift(c)
        scale = self.to_scale(c)
        return x * scale + shift


class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            factor
            ):
        super().__init__()
        self.factor = factor
        self.res = nn.Conv1d(in_channels, out_channels, 1)
        self.c1 = nn.Conv1d(in_channels, out_channels, 3, 1, get_padding(3, 1), dilation=1, padding_mode='replicate')
        self.c2 = nn.Conv1d(out_channels, out_channels, 3, 1, get_padding(3, 2), dilation=2, padding_mode='replicate')
        self.c3 = nn.Conv1d(out_channels, out_channels, 3, 1, get_padding(3, 4), dilation=4, padding_mode='replicate')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1.0 / self.factor)
        
        res = self.res(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        x = x + res
        return x
    

class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            condition_channels,
            factor
    ):
        super().__init__()
        self.factor = factor
        self.c1 = nn.Conv1d(in_channels, in_channels, 3, 1, get_padding(3, 1), dilation=1, padding_mode='replicate')
        self.c2 = nn.Conv1d(in_channels, in_channels, 3, 1, get_padding(3, 3), dilation=3, padding_mode='replicate')
        self.film1 = FiLM(in_channels, condition_channels)
        self.c3 = nn.Conv1d(in_channels, in_channels, 3, 1, get_padding(3, 9), dilation=9, padding_mode='replicate')
        self.c4 = nn.Conv1d(in_channels, in_channels, 3, 1, get_padding(3, 27), dilation=27, padding_mode='replicate')
        self.film2 = FiLM(in_channels, condition_channels)
        self.out = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x, c):
        x = F.interpolate(x, scale_factor=self.factor, mode='linear')

        res = x
        x = F.leaky_relu(x, 0.1)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c2(x)
        x = self.film1(x, c)
        x = x + res

        res = x
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c4(x)
        x = self.film2(x, c)
        x = x + res

        x = F.leaky_relu(x, 0.1)
        x = self.out(x)
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            content_channels=768,
            num_harmonics=0,
            sample_rate=24000,
            frame_size=480,
            channels=[384, 192, 96, 48, 24],
            factors=[2, 3, 4, 4, 5],
    ):
        super().__init__()
        # parameters
        self.num_harmonics = num_harmonics
        self.content_channels = content_channels
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        # input layers
        self.content_input = nn.Conv1d(content_channels, channels[0], 1)
        self.f0_input = nn.Conv1d(1, channels[0], 1)

        # downsamples
        self.downs = nn.ModuleList()
        self.downs.append(nn.Conv1d(num_harmonics+3, channels[-1], 3, 1, 1))
        cs = list(reversed(channels[1:]))
        ns = cs[1:] + [channels[0]]
        fs = list(reversed(factors[1:]))
        for c, n, f, in zip(cs, ns, fs):
            self.downs.append(DownBlock(c, n, f))
        
        # upsamples
        self.ups = nn.ModuleList()
        cs = channels
        ns = channels[1:] + [channels[-1]]
        fs = factors
        for c, n, f in zip(cs, ns, fs):
            self.ups.append(UpBlock(c, n, c, f))
        
        # output layer
        self.output_layer = nn.Conv1d(channels[-1], 1, 5, 1, 2, padding_mode='replicate')
    
    # x: [Batch, ssl_channels, Length]
    # f0: [Batch, 1, Length]
    # energy: [Batch, 1, Length*frame_size]
    # Output: [Batch, Length * frame_size]
    def forward(self, x, f0, energy):
        x = self.content_input(x)
        x = x + self.f0_input(torch.log(f0.abs() + 1e-6)) # add f0 conditioning

        harmonics = oscillate_harmonics(f0, self.frame_size, self.sample_rate, self.num_harmonics)
        noise = torch.randn(f0.shape[0], 1, f0.shape[2] * self.frame_size, device=f0.device)
        source = torch.cat([harmonics, noise, energy], dim=1)

        skips = []
        for down in self.downs:
            source = down(source)
            skips.append(source)

        for up, s in zip(self.ups, reversed(skips)):
            x = up(x, s)
        x = self.output_layer(x)
        return x
    
    def infer(self, x, g, f0):
        return self.forward(x, g, f0).squeeze(1)