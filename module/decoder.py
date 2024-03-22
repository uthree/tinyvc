import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Oscillate harmonic signal
#
# Inputs ---
# f0: [BatchSize, 1, Frames]
# frame_size: int
# num_harmonics: int
# min_frequency: float
# noise_scale: float
#
# Output: [BatchSize, NumHarmonics, Length]
#
# length = Frames * frame_size
def oscillate_harmonics(f0,
                        frame_size=480,
                        sample_rate=24000,
                        num_harmonics=0,
                        min_frequency=20.0,
                        device=torch.device('cpu')):
    N = f0.shape[0]
    Nh = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(Nh, device=device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lw)

    # change length to wave's
    fs = F.interpolate(f0, Lw, mode='linear') * mul

    # unvoiced / voiced mask
    uv = F.interpolate((fs >= min_frequency).to(torch.float), Lw, mode='linear')

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    theta = 2 * math.pi * (I % 1) # convert to radians

    harmonics = torch.sin(theta) * uv

    return harmonics.to(device)


# Convert style based kNN.
# Warning: this method is not optimized.
# Do not give long sequence. computing complexy is quadratic.
# 
# source: [BatchSize, Channels, Length]
# reference: [BatchSize, Channels, Length]
# k: int
# alpha: float (0.0 ~ 1.0)
# metrics: one of ['IP', 'L2', 'cos'], 'IP' means innner product, 'L2' means euclid distance, 'cos' means cosine similarity
# Output: [BatchSize, Channels, Length]
def match_features(source, reference, k=4, alpha=0.0, metrics='cos'):
    input_data = source

    source = source.transpose(1, 2)
    reference = reference.transpose(1, 2)
    if metrics == 'IP':
        sims = torch.bmm(source, reference.transpose(1, 2))
    elif metrics == 'L2':
        sims = -torch.cdist(source, reference)
    elif metrics == 'cos':
        reference_norm = torch.norm(reference, dim=2, keepdim=True, p=2) + 1e-6
        source_norm = torch.norm(source, dim=2, keepdim=True, p=2) + 1e-6
        sims = torch.bmm(source / source_norm, (reference / reference_norm).transpose(1, 2))
    best = torch.topk(sims, k, dim=2)

    result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
    result = result.transpose(1, 2)

    return result * (1-alpha) + input_data * alpha


class FiLM(nn.Module):
    def __init__(self, input_channels, condition_channels):
        super().__init__()
        self.to_shift = nn.Conv1d(condition_channels, input_channels, 1)
        self.to_scale = nn.Conv1d(condition_channels, input_channels, 1)

    def forward(self, x, c):
        shift = self.to_shift(c)
        scale = self.to_scale(c)
        return x * scale + shift


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
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


class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps
    
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class SourceNetLayer(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, padding, dilation=dilation, padding_mode='replicate', groups=channels)
        self.norm = LayerNorm(channels)
        self.c2 = nn.Conv1d(channels, channels * 2, 1)
        self.grn = GRN(channels * 2)
        self.c3 = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.c3(x)
        return x + res


# Source Network.
# This network estimate each harmonic's amplitude and linear filter to noise transformation.
class SourceNet(nn.Module):
    def __init__(self,
                 content_channels=768,
                 channels=256,
                 kernel_size=7,
                 num_layers=6,
                 n_fft=1920,
                 frame_size=480,
                 num_harmonics=14,
                 sample_rate=24000):
        super().__init__()
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        fft_bin = n_fft // 2 + 1
        self.content_in = nn.Conv1d(content_channels, channels, 1)
        self.energy_in = nn.Conv1d(1, channels, 1)
        self.f0_in = nn.Conv1d(1, channels, 1)
        self.mid_layers = nn.Sequential(*[SourceNetLayer(channels, kernel_size) for _ in range(num_layers)])
        self.to_amps = nn.Conv1d(channels, num_harmonics + 1, 1)
        self.to_kernel = nn.Conv1d(channels, fft_bin, 1)

    def forward(self, content, energy, f0):
        x = self.content_in(content) + self.energy_in(energy) + self.f0_in(torch.log(F.relu(f0) + 1e-6))
        x = self.mid_layers(x)
        amps = torch.exp(self.to_amps(x)).clamp_max(6.0)
        kernel = torch.exp(self.to_kernel(x)).clamp_max(6.0)
        return amps, kernel

    def synthesize(self, content, energy, f0):
        waveform_length = content.shape[2] * self.frame_size
        N = content.shape[0]
        device = content.device
        
        # estimate DSP parameters
        amps, kernel = self.forward(content, energy, f0)

        # ---  sinusoid additive synthesizer
        # interpolate amplitude signals
        amps = F.interpolate(amps, scale_factor=self.frame_size, mode='linear')

        # oscillate harmonics
        harmonics = oscillate_harmonics(f0, self.frame_size, self.sample_rate, self.num_harmonics, device=device)

        # amplitude modulation
        harmonics = harmonics * amps

        # --- noise synthesizer
        # oscillate gaussian noise
        gaussian_noise = torch.randn(N, waveform_length, device=device)

        # calculate convolution in fourier-domain
        # Since the input is an aperiodic signal such as Gaussian noise,
        # there is no need to consider the phase on the kernel side.
        w = torch.hann_window(self.n_fft).to(device) # get window
        noise_stft = torch.stft(gaussian_noise, self.n_fft, hop_length=self.frame_size, window=w, return_complex=True)[:, :, 1:]
        n = noise_stft * kernel # In fourier domain, Multiplication means convolution.
        n = F.pad(n, [1, 0]) # pad
        noise = torch.istft(n, self.n_fft, self.frame_size, window=w)
        noise = noise.unsqueeze(1)

        # concatenate and return output
        output = torch.cat([harmonics, noise], dim=1)
        return output


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = nn.Conv1d(input_channels, input_channels, 3, 1, 1, dilation=1, padding_mode='replicate')
        self.c2 = nn.Conv1d(input_channels, input_channels, 3, 1, 2, dilation=2, padding_mode='replicate')
        self.c3 = nn.Conv1d(input_channels, output_channels, 3, 1, 4, dilation=4, padding_mode='replicate')
        self.pool = nn.AvgPool1d(factor * 2, factor)

    def forward(self, x):
        x = F.pad(x, [self.factor, 0], mode='replicate')
        x = self.pool(x)
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
        c = F.interpolate(c, scale_factor=self.factor, mode='linear')
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
        self.content_channels = content_channels

        # input layers
        self.energy_in = nn.Conv1d(1, channels[0], 1)
        self.content_in = nn.Conv1d(content_channels, channels[0], 1)

        # downsample layers
        self.down_input = nn.Conv1d(num_harmonics + 2, channels[-1], 5, 1, 2, padding_mode='replicate')
        self.downs = nn.ModuleList([])
        cond = list(reversed(channels))
        cond_next = cond[1:] + [cond[-1]]
        for c, c_n, f in zip(cond, cond_next, reversed(factors)):
            self.downs.append(
                    Downsample(c, c_n, f))

        # upsample layers
        self.ups = nn.ModuleList([])
        up = channels
        up_next = channels[1:] + [channels[-1]]
        for u, u_n, c_n, f in zip(up, up_next, reversed(cond_next), factors):
            self.ups.append(Upsample(u, u_n, c_n, f))

        # output layer
        self.output_layer = nn.Conv1d(channels[-1], 1, 5, 1, 2, padding_mode='replicate')

    def forward(self, content, energy, source):
        x = self.content_in(content) + self.energy_in(energy)

        skips = []
        src = self.down_input(source)
        for d in self.downs:
            src = d(src)
            skips.append(src)

        # upsamples
        for u, s in zip(self.ups, reversed(skips)):
            x = u(x, s)

        x = self.output_layer(x)
        return x

    def synthesize(self, content, energy, source):
        return self.forward(content, energy, source)


class Decoder(nn.Module):
    def __init__(self, sample_rate=24000, frame_size=480):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.source_net = SourceNet(frame_size=frame_size, sample_rate=sample_rate)
        self.filter_net = FilterNet()

    def infer(self, content, energy, f0):
        src = self.source_net.synthesize(content, energy, f0)
        out = self.filter_net.synthesize(content, energy, source)
        return out
