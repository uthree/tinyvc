import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import spectrogram, CausalConv1d


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=(1, 2), keepdim=True)
        sigma = x.std(dim=(1, 2), keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.c1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = LayerNorm(channels)
        self.c2 = nn.Conv1d(channels, channels * 2, 1)
        self.c3 = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        return x + res


class Encoder(nn.Module):
    def __init__(self, n_fft=1920,
                 hop_size=480,
                 channels=256,
                 dilations=[1, 3, 5, 1, 3, 5],
                 kernel_size=3,
                 num_phones=32,
                 num_f0_classes=256,
                 f0_min=20):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.num_f0_classes = num_f0_classes

        self.input_layer = nn.Conv1d(n_fft//2+1, channels, 1)
        self.blocks = nn.Sequential(*[
            ResBlock(channels, kernel_size, d) for d in dilations])
        self.to_phone = nn.Conv1d(channels, num_phones, 1)
        self.to_f0 = nn.Conv1d(channels, num_f0_classes, 1)

    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.blocks(x)
        phone = self.to_phone(x)
        f0_logits = self.to_f0(x)
        return phone, f0_logits

    def infer(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        phone, f0_logits = self.forward(spec)
        ids = torch.argmax(f0_logits, dim=1, keepdim=True)
        f0 = self.id2freq(ids)
        return phone, f0

    def freq2id(self, f):
        return torch.round(torch.clamp(50 * torch.log2(f / self.f0_min), 0, self.num_f0_classes-1)).to(torch.long)

    def id2freq(self, ids):
        x = ids.to(torch.float)
        x = self.f0_min * (2 ** (x / 50))
        x[x <= self.f0_min] = 0
        return x

