import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import Generator

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
            use_phase_vocoder=False,
            f0_estimation='default'):
        self.pitch_shift = pitch_shift
        self.generator = generator
        self.device = device
        self.target = target

        self.block_size = block_size
        self.extra_size = extra_size
        self.sola_search_size = 1920
        self.last_dilay_size = 3840
        self.crossfade_size = 1920
        self.use_phase_vocoder = use_phase_vocoder
        self.f0_estimation = f0_estimation

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

        y = self.generator.convert(self.input_wav[None], self.target, self.pitch_shift, device=self.device, f0_estimation=self.f0_estimation).squeeze(0)

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