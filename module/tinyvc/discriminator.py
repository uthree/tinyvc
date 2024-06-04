import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorS(nn.Module):
    def __init__(self, scale=1, channels=32, num_layers=4):
        super().__init__()
        self.pool = nn.AvgPool1d(scale)
        self.convs = nn.ModuleList([weight_norm(nn.Conv1d(1, channels, 41, 3, 20))])
        c = channels
        g = 1
        for _ in range(num_layers):
            self.convs.append(weight_norm(nn.Conv1d(c, c*2, 21, 3, 10, 1, g)))
            c = c*2
            g = g*2
        self.post = weight_norm(nn.Conv1d(c, 1, 21, 3, 10))
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(x)
        fmap = []
        for c in self.convs:
            x = c(x)
            fmap.append(x)
            F.leaky_relu(x, 0.1)
        x = self.post(x)
        fmap.append(x)
        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            scales=[1, 2, 4],
            channels=32,
            num_layers=4
            ):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for s in scales:
            self.sub_discs.append(DiscriminatorS(s, channels, num_layers))

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