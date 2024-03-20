import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import spectrogram


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
        padding = (kernel_size - 1) * dilation // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, padding, dilation=dilation, padding_mode='replicate')
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
                 dilations=[1, 3, 5, 7, 9, 1],
                 kernel_size=3,
                 hubert_channels=768,
                 num_phones=32,
                 num_f0_classes=512,
                 f0_min=20,
                 f0_estimate_topk=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.num_f0_classes = num_f0_classes
        self.f0_estimate_topk = f0_estimate_topk

        self.input_layer = nn.Conv1d(n_fft//2+1, channels, 1)
        self.blocks = nn.Sequential(*[
            ResBlock(channels, kernel_size, d) for d in dilations])
        self.to_content = nn.Conv1d(channels, hubert_channels, 1)
        self.to_f0 = nn.Conv1d(channels, num_f0_classes, 1)

    def forward(self, spec):
        x = self.input_layer(spec)
        x = self.blocks(x)
        content = self.to_content(x)
        f0_logits = self.to_f0(x)
        return content, f0_logits

    def infer(self, wave):
        spec = spectrogram(wave, self.n_fft, self.hop_size)
        content, f0_logits = self.forward(spec)
        probs, indices = torch.topk(f0_logits, self.f0_estimate_topk, dim=1)
        probs = F.softmax(probs, dim=1)
        freqs = self.id2freq(indices)
        f0 = (probs * freqs).sum(dim=1, keepdim=True)
        f0[f0 <= self.f0_min] = 0
        return content, f0

    def freq2id(self, f):
        return torch.ceil(torch.clamp(50 * torch.log2(f / 10.0), 0, self.num_f0_classes-1)).to(torch.long)

    def id2freq(self, ids):
        x = ids.to(torch.float)
        x = 10.0 * (2 ** (x / 50))
        x[x <= self.f0_min] = 0
        return x

