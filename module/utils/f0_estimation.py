import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import resample
from torchfcpe import spawn_bundled_infer_model

import numpy as np
import pyworld as pw


def estimate_f0_dio(wf, sample_rate=24000, frame_size=480, f0_min=20, f0_max=20000):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        _f0, t = pw.dio(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = pw.stonemask(signal, _f0, t, sample_rate)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // frame_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [estimate_f0_dio(wave[0], sample_rate, frame_size) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


def estimate_f0_harvest(wf, sample_rate=24000, frame_size=480, f0_min=20, f0_max=20000):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        f0, t = pw.harvest(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // frame_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [estimate_f0_harvest(wave[0], sample_rate, frame_size) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


global torchfcpe_model
torchfcpe_model = {}
def estimate_f0_fcpe(wf, sample_rate=24000, frame_size=480, f0_min=20, f0_max=20000):
    if wf.device not in torchfcpe_model:
        torchfcpe_model[wf.device] = spawn_bundled_infer_model(wf.device)
    f0 = torchfcpe_model[wf.device].infer(wf.unsqueeze(2), sample_rate)
    f0 = f0.transpose(1, 2)
    return f0


def estimate_f0(wf, sample_rate=24000, frame_size=480, algorithm='fcpe'):
    l = wf.shape[1]
    if algorithm == 'harvest':
        f0 = estimate_f0_harvest(wf, sample_rate)
    elif algorithm == 'dio':
        f0 = estimate_f0_dio(wf, sample_rate)
    elif algorithm == 'fcpe':
        f0 = estimate_f0_fcpe(wf, sample_rate)
    return F.interpolate(f0, l // frame_size, mode='linear')
