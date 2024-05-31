import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorS(nn.Module):
    def __init__(self, scale=1, channels=32, num_layers=4, use_spectral_norm=False):
        super().__init__()
        self.pool = nn.AvgPool1d(scale)
        norm_f = nn.utils.parametrizations.spectral_norm if use_spectral_norm else nn.utils.parametrizations.weight_norm
        self.pre = norm_f(nn.Conv1d(1, channels, 41, 3, 20))
        self.convs = nn.ModuleList([])
        c = channels
        g = 1
        for _ in range(num_layers):
            self.convs.append(norm_f(nn.Conv1d(c, c*2, 21, 3, 10, groups=g)))
            c = c*2
            g = g*2
        self.convs.append(norm_f(nn.Conv1d(c, 1, 21, 3, 10)))

    def forward(self, x):
        fmap = []
        x = x.unsqueeze(1)
        x = self.pool(x)
        x = self.pre(x)
        x = F.leaky_relu(x, 0.1)
        for c in self.convs:
            fmap.append(x)
            x = c(x)
            x = F.leaky_relu(x, 0.1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            scales=[1, 2, 3],
            channels=32,
            num_layers=4
            ):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for s in scales:
            use_spectral_norm = (s == 1)
            self.sub_discs.append(DiscriminatorS(s, channels, num_layers, use_spectral_norm))

    def forward(self, x):
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSD = MultiScaleDiscriminator()

    # x: [BatchSize, Length(waveform)]
    def forward(self, x):
        msd_logits, msd_feats = self.MSD(x)
        return msd_logits, msd_feats
