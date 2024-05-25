import torch
import torch.nn as nn
import torch.nn.functional as F


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def spectrogram(wave, n_fft=1920, hop_size=480):
    dtype = wave.dtype
    wave = wave.to(torch.float)
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window).abs()
    spec = spec[:, :, 1:]
    spec = spec.to(dtype)
    return spec





