import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class SineOscillator(nn.Module):
    def __init__(
            self,
            sample_rate=24000,
            frame_size=480,
            min_frequency=20.0,
            noise_scale=0.05
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.min_frequency = min_frequency
        self.noise_scale = noise_scale

    def forward(self, f0):
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode='linear')
        uv = (f0 >= self.min_frequency).to(torch.float)
        integrated = torch.cumsum(f0 / self.sample_rate, dim=2)
        theta = 2 * math.pi * (integrated % 1)
        sinusoid = torch.sin(theta) * uv
        sinusoid = sinusoid + torch.randn_like(sinusoid) * self.noise_scale
        return sinusoid


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList([])
        self.convs2 = nn.ModuleList([])
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, 1), 1)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x
    

class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.convs = nn.ModuleList([])
        for d in dilations:
            self.convs.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            content_channels=768,
            sample_rate=24000,
            frame_size=480,
            upsample_initial_channels=256,
            resblock_type="2",
            resblock_kernel_sizes=[3, 5, 7],
            resblock_dilations=[[1, 2], [2, 6], [3, 12]],
            upsample_kernel_sizes=[24, 20, 8],
            upsample_rates=[12, 10, 4]
        ):
        super().__init__()
        self.content_channels = content_channels
        self.frame_size = frame_size
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.oscillator = SineOscillator(sample_rate, frame_size)
        self.conv_pre = weight_norm(nn.Conv1d(content_channels, upsample_initial_channels, 7, 1, 3))
        self.source_pre = weight_norm(nn.Conv1d(2, upsample_initial_channels//(2**(self.num_upsamples)), 7, 1, 3))
        self.ups = nn.ModuleList([])
        downs = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels//(2**i)
            c2 = upsample_initial_channels//(2**(i+1))
            p = (k-u)//2
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
            downs.append(weight_norm(nn.Conv1d(c2, c1, k, u, p)))
        self.downs = nn.ModuleList(reversed(downs))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0, energy):
        sinusoid = self.oscillator(f0)
        s = torch.cat([sinusoid, energy], dim=1)
        s = self.source_pre(s)
        skips = [s]
        for i in range(self.num_upsamples):
            s = self.downs[i](s)
            skips.append(s)
        skips = list(reversed(skips))
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = x + skips[i]
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = x + skips[-1]
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
    
    def infer(self, x, f0, energy):
        return self.forward(x, f0, energy).squeeze(1)