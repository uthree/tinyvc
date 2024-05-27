import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import ConvNeXtLayer


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
    N = f0.shape[0]
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


# Oscillate noise via gaussian noise and equalizer
#
# fft_bin = n_fft // 2 + 1
# kernel: [BatchSize, fft_bin, Length]
#
# Output: [BatchSize, 1, Length * frame_size]
def oscillate_noise(
        kernel,
        frame_size=480,
        n_fft=1920
    ):
    N = kernel.shape[0]
    Lf = kernel.shape[2] # frame length
    fft_bin = n_fft // 2 + 1
    dtype = kernel.dtype

    kernel = kernel.to(torch.float) # to fp32

    # calculate convolution in fourier-domain
    # Since the input is an aperiodic signal such as Gaussian noise,
    # there is no need to consider the phase on the kernel side.
    angle = torch.rand(N, fft_bin, Lf, device=kernel.device) * 2 * math.pi - math.pi
    noise_stft = torch.exp(1j * angle)
    y_stft = noise_stft * kernel # In fourier domain, Multiplication means convolution.
    y_stft = F.pad(y_stft, [1, 0]) # pad
    y = torch.istft(y_stft, n_fft, frame_size)
    y = y.unsqueeze(1)
    y = y.to(dtype)
    return y


class FiLM(nn.Module):
    def __init__(self, input_channels, condition_channels):
        super().__init__()
        self.to_shift = nn.Conv1d(condition_channels, input_channels, 1)
        self.to_scale = nn.Conv1d(condition_channels, input_channels, 1)

    def forward(self, x, c):
        shift = self.to_shift(c)
        scale = self.to_scale(c)
        return x * scale + shift


# Source Network.
# This network estimate each harmonic's amplitude and linear filter to noise transformation.
class SourceNet(nn.Module):
    def __init__(self,
                 content_channels=768,
                 channels=128,
                 kernel_size=7,
                 num_layers=3,
                 n_fft=1920,
                 frame_size=480,
                 num_harmonics=14,
                 sample_rate=24000):
        super().__init__()
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.content_channels = content_channels
        fft_bin = n_fft // 2 + 1
        self.content_in = nn.Conv1d(content_channels, channels, 1)
        self.energy_in = nn.Conv1d(1, channels, 1)
        self.f0_in = nn.Conv1d(1, channels, 1)
        self.mid_layers = nn.Sequential(*[ConvNeXtLayer(channels, kernel_size) for _ in range(num_layers)])
        self.to_amps = nn.Conv1d(channels, num_harmonics + 1, 1)
        self.to_kernel = nn.Conv1d(channels, fft_bin, 1)

    def forward(self, content, f0, energy):
        energy = F.max_pool1d(energy, self.frame_size, self.frame_size)
        x = self.content_in(content) + self.energy_in(energy) + self.f0_in(torch.log(F.relu(f0) + 1e-6))
        x = self.mid_layers(x)
        # 実はこの活性化めっちゃ優秀説ある
        # 正の方向にはLinear, 負の方向にはexp(x)として働くので勾配消えないし値デカくなりすぎない。最高。
        amps = F.elu(self.to_amps(x)) + 1.0
        kernel = F.elu(self.to_kernel(x)) + 1.0
        return amps, kernel


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = nn.Conv1d(input_channels, input_channels, 3, 1, 1, dilation=1, padding_mode='replicate')
        self.c2 = nn.Conv1d(input_channels, input_channels, 3, 1, 2, dilation=2, padding_mode='replicate')
        self.c3 = nn.Conv1d(input_channels, output_channels, 3, 1, 4, dilation=4, padding_mode='replicate')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1.0/self.factor, mode='linear')
        res = self.down_res(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        x = x + res
        return x


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, cond_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.c1 = nn.Conv1d(input_channels, input_channels, 3, 1, 1, dilation=1, padding_mode='replicate')
        self.c2 = nn.Conv1d(input_channels, input_channels, 3, 1, 3, dilation=3, padding_mode='replicate')
        self.film1 = FiLM(input_channels, cond_channels)
        self.c3 = nn.Conv1d(input_channels, input_channels, 3, 1, 9, dilation=9, padding_mode='replicate')
        self.c4 = nn.Conv1d(input_channels, input_channels, 3, 1, 27, dilation=27, padding_mode='replicate')
        self.film2 = FiLM(input_channels, cond_channels)
        self.c5 = nn.Conv1d(input_channels, output_channels, 1)

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
        x = self.c5(x)
        return x


class FilterNet(nn.Module):
    def __init__(self,
                 channels=[384, 192, 96, 48, 24],
                 factors=[2, 3, 4, 4, 5],
                 content_channels=768,
                 num_harmonics=14):
        super().__init__()
        # input layer 
        self.content_in = nn.Conv1d(content_channels, channels[0], 1)
        self.f0_in = nn.Conv1d(1, channels[0], 1)

        # donwsamples
        self.downs = nn.ModuleList([])
        self.downs.append(nn.Conv1d(num_harmonics + 3, channels[-1], 3, 1, 1, padding_mode='replicate'))
        cs = list(reversed(channels[1:]))
        ns = cs[1:] + [channels[0]]
        fs = list(reversed(factors[1:]))
        for c, n, f in zip(cs, ns, fs):
            self.downs.append(Downsample(c, n, f))

        # upsamples
        self.ups = nn.ModuleList([])
        cs = channels
        ns = channels[1:] + [channels[-1]]
        fs = factors
        for c, n, f in zip(cs, ns, fs):
            self.ups.append(Upsample(c, n, c, f))
        self.output_layer = nn.Conv1d(channels[-1], 1, 7, 1, 3, padding_mode='replicate')

    def forward(self, content, f0, energy, source):
        x = self.content_in(content) + self.f0_in(torch.log(F.relu(f0) + 1e-6))
        src = torch.cat([source, energy], dim=1)

        skips = []
        for down in self.downs:
            src = down(src)
            skips.append(src)

        for up, s in zip(self.ups, reversed(skips)):
            x = up(x, s)
        return self.output_layer(x)


class Decoder(nn.Module):
    def __init__(
            self,
            sample_rate=24000,
            n_fft = 1920,
            frame_size=480,
            num_harmonics=14,
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.n_fft = n_fft

        self.source_net = SourceNet(frame_size=frame_size, sample_rate=sample_rate, n_fft=n_fft)
        self.filter_net = FilterNet()

    def infer(self, content, f0, energy):
        amps, kernel = self.source_net(content, f0, energy)
        src = self.dsp(f0, amps, kernel)
        out = self.filter_net(content, f0, energy, src)
        return out.squeeze(1)
    
    @torch.cuda.amp.autocast(enabled=False)
    def dsp(self, f0, amps, kernel):
        harmonics = oscillate_harmonics(f0, self.frame_size, self.sample_rate, self.num_harmonics)
        amps = F.interpolate(amps, scale_factor=self.frame_size, mode='linear')
        harmonics = harmonics * amps
        noise = oscillate_noise(kernel, self.frame_size, self.n_fft)
        source = torch.cat([harmonics, noise], dim=1)
        return source
