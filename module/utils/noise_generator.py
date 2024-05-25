from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
import random


class NoiseGenerator():
    def __init__(self, dir_path, sample_rate=24000):
        self.waveforms = []
        dir_path = Path(dir_path)
        support_formats = ['mp3', 'ogg', 'wav']
        for fmt in support_formats:
            for p in dir_path.glob(f"*.{fmt}"):
                wf, sr = torchaudio.load(p)
                if sr != sample_rate:
                    wf = resample(wf, sr, sample_rate)
                wf = wf.mean(dim=0) # to single channels
                self.waveforms.append(wf)

    def _add_noise(self, x):
        # x: [Length]
        if random.random() < 0.3:
            noise = random.choice(self.waveforms)
            s = random.randint(0, noise.shape[0] - x.shape[0] - 1)
            noise = noise[s:s+x.shape[0]].to(x.device)
            x = x + noise * random.random()
        return x

    def add_noise(self, xs):
        return torch.cat([self._add_noise(x.squeeze(0)).unsqueeze(0) for x in xs.split(1, dim=0)], dim=0)
