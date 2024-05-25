import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.tinyvc import Encoder, Decoder, match_features
from module.utils import spectrogram, shift_frequency, estimate_energy, autopad_waveform

from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @torch.inference_mode()
    def encode(self, wf):
        wf = autopad_waveform(wf)
        spec = spectrogram(wf)
        tgt, f0 = self.encoder.infer(spec)
        return tgt, f0
    
    @torch.inference_mode()
    def convert(self, wf, tgt, pitch_shift, f0_estimation='default', device=torch.device('cpu')):
        wf = autopad_waveform(wf)
        spec = spectrogram(wf)
        energy = estimate_energy(wf)
        z, f0 = self.encoder.infer(spec)
        z = match_features(z, tgt)
        f0 = shift_frequency(f0, pitch_shift)
        out = self.decoder.infer(z, f0, energy)
        return out