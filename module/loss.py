import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def safe_log(x):
    return torch.log(x.clamp_min(1e-6))


class LogMelSpectrogramLoss(nn.Module):
    def __init__(
            self,
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=128
            ):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate,
                n_fft,
                hop_length=hop_length,
                n_mels=n_mels)
    
    def forward(self, x, y):
        x = x.to(torch.float)
        y = y.to(torch.float)
        
        x = safe_log(self.to_mel(x))
        y = safe_log(self.to_mel(y))

        x[x.isnan()] = 0
        x[x.isinf()] = 0
        y[y.isnan()] = 0
        y[y.isinf()] = 0

        return (x - y).abs().mean()
