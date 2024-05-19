import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder, match_features
from .common import estimate_energy, spectrogram
from .f0_estimation import estimate_f0

from tqdm import tqdm


def frqeuency_to_midi(f):
    return torch.log2(F.relu(f / 440) + 1e-6) * 12 + 69

def midi_to_frequency(n):
    return 440 * 2 ** ((n - 69) / 12)

def shift_frequency(f0, shift):
    pitch = frqeuency_to_midi(f0)
    pitch += shift
    f0 = midi_to_frequency(pitch)
    return f0

def get_pitch_stats(f0):
    mask = f0 > 20.0
    n = frqeuency_to_midi(f0)
    mu = (n * mask).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True)
    sigma = torch.sqrt(((mu - n) ** 2).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True) + 1e-6)
    return mu, sigma

def normalize_frequency(f0, input_stats, output_stats):
    n = frqeuency_to_midi(f0)
    mu_x, sigma_x = input_stats
    mu_y, sigma_y = output_stats
    n = (n - mu_x) / sigma_x
    n = n * sigma_y + mu_y
    f0 = midi_to_frequency(n)
    return f0


class Generator(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode_target(self, wf, with_stats=False):
        spec = spectrogram(wf, self.encoder.n_fft, self.encoder.hop_size)
        z, f0 = self.encoder.infer(spec)
        pitch_stats = get_pitch_stats(f0)
        if with_stats:
            return z, pitch_stats
        else:
            return z

    def convert(self, wf, tgt, pitch_shift, device=torch.device('cpu'), f0_estimation='default', input_pitch_stats=None, output_pitch_stats=None):
        # estimate energy, encode content, estimate pitch
        energy = estimate_energy(wf)
        spec = spectrogram(wf, self.encoder.n_fft, self.encoder.hop_size)
        z, f0 = self.encoder.infer(spec)
        if f0_estimation != 'default':
            f0 = estimate_f0(wf, algorithm=f0_estimation)

        if input_pitch_stats is not None and output_pitch_stats is not None:
            f0 = normalize_frequency(f0, input_pitch_stats, output_pitch_stats)
        
        f0 = shift_frequency(f0, pitch_shift)

        # match features
        z = match_features(z, tgt)

        # synthesize new wave
        y = self.decoder.infer(z, energy, f0).squeeze(1)

        return y


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = a * (fade_out ** 2) + b * (fade_in ** 2) + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    return result


# stream inference with SOLA
class StreamInfer:
    def __init__(
            self,
            generator: Generator,
            target = None,
            pitch_shift=0.,
            device=torch.device('cpu'),
            block_size=1920,
            extra_size=0,
            use_phase_vocoder=False):
        self.pitch_shift = pitch_shift
        self.generator = generator
        self.device = device
        self.target = target

        self.block_size = block_size
        self.extra_size = extra_size
        self.sola_search_size = 960
        self.last_dilay_size = 480
        self.crossfade_size = 960
        self.use_phase_vocoder = use_phase_vocoder

        self.input_size = max(
            self.block_size + self.crossfade_size + self.sola_search_size + 2 * self.last_dilay_size,
            self.block_size + self.extra_size
        )

    def init_buffer(self):
        # initialize buffer
        self.fade_in_window = torch.sin(np.pi * torch.arange(0, 1, 1 / self.crossfade_size, device=self.device) / 2) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.input_wav = torch.zeros(self.input_size, device=self.device)
        self.sola_buffer = torch.zeros(self.crossfade_size, device=self.device)

    # input_data: [Length]
    @torch.inference_mode()
    def audio_callback(self, block):
        self.input_wav = torch.roll(self.input_wav, -self.block_size)
        self.input_wav[-self.block_size:] = block

        y = self.generator.convert(self.input_wav[None], self.target, self.pitch_shift, device=self.device).squeeze(0)

        # sola shift
        temp_wav = y[-self.block_size-self.crossfade_size-self.sola_search_size-self.last_dilay_size:-self.last_dilay_size ]
        conv_input = temp_wav[None, None, :self.crossfade_size + self.sola_search_size]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, torch.ones(1, 1, self.crossfade_size, device=self.device)) + 1e-8)
        sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item()
        temp_wav = temp_wav[sola_shift: sola_shift + self.block_size + self.crossfade_size]

        # phase vocoder
        if self.use_phase_vocoder:
            temp_wav[:self.crossfade_size] = phase_vocoder(
                self.sola_buffer,
                temp_wav[:self.crossfade_size],
                self.fade_out_window,
                self.fade_in_window
            )
        else:
            temp_wav[:self.crossfade_size] *= self.fade_in_window
            temp_wav[:self.crossfade_size] += self.sola_buffer * self.fade_out_window
        
        self.sola_buffer = temp_wav[-self.crossfade_size:]
        block = temp_wav[:-self.crossfade_size]
        return block