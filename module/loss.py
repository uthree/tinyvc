import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)


class MultiScaleSTFTLoss(nn.Module):
    def __init__(
            self,
            scales=[16, 32, 64, 128, 256, 512]
            ):
        super().__init__()
        self.scales = scales

    def forward(self, x, y):
        x = x.to(torch.float)
        y = y.to(torch.float)

        loss = 0
        num_scales = len(self.scales)
        for s in self.scales:
            hop_length = s
            n_fft = s * 4
            window = torch.hann_window(n_fft, device=x.device)
            x_spec = torch.stft(x, n_fft, hop_length, return_complex=True, window=window).abs()
            y_spec = torch.stft(y, n_fft, hop_length, return_complex=True, window=window).abs()

            x_spec[x_spec.isnan()] = 0
            x_spec[x_spec.isinf()] = 0
            y_spec[y_spec.isnan()] = 0
            y_spec[y_spec.isinf()] = 0

            loss += ((x_spec - y_spec) ** 2).mean() + (safe_log(x_spec) - safe_log(y_spec)).abs().mean()
        return loss / num_scales
