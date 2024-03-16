import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, channels=16, num_layers=4, max_channels=256, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm
        
        k = kernel_size
        s = stride
        c = channels

        convs = [nn.Conv2d(1, c, (k, 1), (s, 1), (get_padding(5, 1), 0))]
        for i in range(num_layers):
            c_next = min(c * 2, max_channels)
            convs.append(nn.Conv2d(c, c_next, (k, 1), (s, 1), (get_padding(5, 1), 0)))
            c = c_next
        self.convs = nn.ModuleList([norm_f(c) for c in convs])
        self.post = norm_f(nn.Conv2d(c, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiPeriodicDiscriminator(nn.Module):
    def __init__(self, periods, channels, max_channels, num_layers):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for p in periods:
            self.sub_discs.append(DiscriminatorP(p,
                                                 channels=channels,
                                                 max_channels=max_channels,
                                                 num_layers=num_layers))

    def forward(self, x):
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class DiscriminatorS(nn.Module):
    def __init__(self, scale=1, channels=32, num_layers=6, max_channels=256, max_groups=8, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.weight_norm if use_spectral_norm == False else nn.utils.spectral_norm

        if scale != 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(scale*2, scale)

        c = channels
        g = 1
        convs = [nn.Conv1d(1, c, 15, 1, 7)]
        for _ in range(num_layers):
            g = min(g * 2, max_groups)
            c_next = min(c * 2, max_channels)
            convs.append(nn.Conv1d(c, c_next, 41, 2, 20, groups=2))
            c = c_next

        self.convs = nn.ModuleList([norm_f(c) for c in convs])
        self.post = norm_f(nn.Conv1d(c, 1, 3, 1, 1))

    def forward(self, x):
        fmap = []
        x = self.pool(x)
        fmap.append(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales, channels, max_channels, max_groups, num_layers):
        super().__init__()
        self.sub_discs = nn.ModuleList([])
        for i, s in enumerate(scales):
            use_spectral_norm = (i == 0)
            self.sub_discs.append(DiscriminatorS(s, channels, num_layers, max_channels, max_groups, use_spectral_norm))

    def forward(self, x):
        feats = []
        logits = []
        for d in self.sub_discs:
            logit, fmap = d(x)
            logits.append(logit)
            feats += fmap
        return logits, feats


class Discriminator(nn.Module):
    def __init__(self,
                 scales=[1, 2, 3],
                 periods=[], #[5, 7, 11, 13, 17, 23, 31]
                 mpd_num_layers=4,
                 msd_num_layers=6,
                 mpd_channels=32,
                 msd_channels=32,
                 mpd_max_channels=256,
                 msd_max_groups=8,
                 msd_max_channels=256,
                 ):
        super().__init__()
        self.MPD = MultiPeriodicDiscriminator(periods, mpd_channels, mpd_max_channels, mpd_num_layers)
        self.MSD = MultiScaleDiscriminator(scales, msd_channels, msd_max_channels, msd_max_groups, msd_num_layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        mpd_logits, mpd_feats = self.MPD(x)
        msd_logits, msd_feats = self.MSD(x)
        return mpd_logits + msd_logits, mpd_feats + msd_feats
