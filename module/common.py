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
def estimate_energy(wave,
           frame_size=480):
    return F.max_pool1d((wave.abs()).unsqueeze(1), frame_size)


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
