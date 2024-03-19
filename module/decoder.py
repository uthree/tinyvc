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
                        noise_scale=0.33):
    N = f0.shape[0]
    Nh = num_harmonics + 1
    Lf = f0.shape[2]
    Lw = Lf * frame_size

    device = f0.device

    # generate frequency of harmonics
    mul = (torch.arange(Nh, device=device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)
    fs = f0 * mul

    # change length to wave's
    fs = F.interpolate(fs, Lw, mode='linear')

    # unvoiced / voiced mask
    uv = F.interpolate((fs >= min_frequency).to(torch.float), Lw, mode='linear')

    # generate harmonics
    I = torch.cumsum(fs / sample_rate, dim=2) # numerical integration
    theta = 2 * math.pi * (I % 1) # convert to radians

    harmonics = torch.sin(theta) * uv

    # add noise
    noise = torch.randn_like(harmonics) * noise_scale
    output = harmonics + noise

    return output


class SpeakerEmbedding(nn.Module):
    def __init__(self, num_speakers=1024, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_speakers, embedding_dim)

    def forward(self, spk_id):
        e = self.embedding(spk_id)
        e = e.unsqueeze(2)
        return e


class FiLM(nn.Module):
    def __init__(self, input_channels, condition_channels):
        super().__init__()
        self.to_shift = nn.Conv1d(condition_channels, input_channels, 1)
        self.to_scale = nn.Conv1d(condition_channels, input_channels, 1)

    def forward(self, x, c):
        shift = self.to_shift(c)
        scale = self.to_scale(c)
        return x * scale + shift


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = nn.Conv1d(input_channels, input_channels, 3, 1, 1, dilation=1, padding_mode='replicate')
        self.c2 = nn.Conv1d(input_channels, input_channels, 3, 1, 2, dilation=2, padding_mode='replicate')
        self.c3 = nn.Conv1d(input_channels, output_channels, 3, 1, 4, dilation=4, padding_mode='replicate')
        self.pool = nn.AvgPool1d(factor)

    def forward(self, x):
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


class Synthesizer(nn.Module):
    def __init__(self,
                 channels=[384, 192, 96, 48, 24],
                 factors=[2, 3, 4, 4, 5],
                 num_harmonics=0,
                 spk_dim=256,
                 content_channels=64,
                 sample_rate=24000,
                 frame_size=480):
        super().__init__()
        self.spk_dim = spk_dim
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.content_channels = content_channels

        # input layers
        self.spk_in = nn.Conv1d(spk_dim, channels[0], 1)
        self.energy_in = nn.Conv1d(1, channels[0], 1)
        self.phone_in = nn.Conv1d(content_channels, channels[0], 1)
        self.film = FiLM(channels[0], channels[0])

        # downsample layers
        self.down_input = nn.Conv1d(num_harmonics + 1, channels[-1], 5, 1, 2, padding_mode='replicate')
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

    def forward(self, phone, energy, spk, source_signals):
        x = self.phone_in(phone)
        c = self.energy_in(energy) + self.spk_in(spk)
        x = self.film(x, c)

        skips = []
        src = self.down_input(source_signals)
        for d in self.downs:
            src = d(src)
            skips.append(src)

        # upsamples
        for u, s in zip(self.ups, reversed(skips)):
            x = u(x, s)

        x = self.output_layer(x)
        x = x.squeeze(1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = Synthesizer()
        self.speaker_embedding = SpeakerEmbedding()

    def infer(self, phone, energy, spk_id, f0):
        src = oscillate_harmonics(
                f0,
                self.synthesizer.frame_size,
                self.synthesizer.sample_rate,
                self.synthesizer.num_harmonics)
        spk = self.speaker_embedding(spk_id)
        output = self.synthesizer(phone, energy, spk, src)
        output = src[:, 0]
        return output
