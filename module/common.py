import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import resample
import numpy as np
import pyworld as pw


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def spectrogram(wave, n_fft, hop_size):
    dtype = wave.dtype
    wave = wave.to(torch.float)
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window).abs()
    spec = spec[:, :, 1:]
    spec = spec.to(dtype)
    return spec


# estimate energy
# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def energy(wave,
           frame_size=480):
    return F.max_pool1d((wave.abs()).unsqueeze(1), frame_size)


# Oscillate harmonic signal
#
# Inputs ---
# f0: [BatchSize, 1, Frames]
#
# Outputs ---
# (signals, phase)
# signals: [BatchSize, NumHarmonics, Length]
# phase: [BatchSize, NumHarmonics Length]
#
# phase's range is 0 to 1, multiply 2 * pi if you need radians
# length = Frames * frame_size
def oscillate_harmonics(f0,
                        frame_size=480,
                        sample_rate=24000,
                        num_harmonics=0,
                        begin_point=0,
                        min_frequency=10.0,
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


def estimate_f0_dio(wf, sample_rate=24000, segment_size=480, f0_min=20, f0_max=20000):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        _f0, t = pw.dio(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = pw.stonemask(signal, _f0, t, sample_rate)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // segment_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [estimate_f0_dio(wave[0], sample_rate, segment_size) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


def estimate_f0_harvest(wf, sample_rate=24000, segment_size=480, f0_min=20, f0_max=20000):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        f0, t = pw.harvest(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // segment_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [estimate_f0_dio(wave[0], sample_rate, segment_size) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


def estimate_f0(wf, sample_rate=24000, segment_size=480, algorithm='harvest'):
    l = wf.shape[1]
    wf = resample(wf, sample_rate, 16000)
    if algorithm == 'harvest':
        pitchs = estimate_f0_harvest(wf, 16000)
    elif algorithm == 'dio':
        pitchs = estimate_f0_dio(wf, 16000)
    return F.interpolate(pitchs, l // segment_size, mode='linear')

