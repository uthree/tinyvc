import torch
import torch.nn as nn
import torch.nn.functional as F 

def autopad_waveform(wf, frame_size=480):
    if wf.shape[1] % frame_size != 0:
        device = wf.device
        N = wf.shape[0]
        pad_size = frame_size - (wf.shape[1] % frame_size)
        wf = torch.cat([wf, torch.zeros(N, pad_size, device=device)], dim=1)
    return wf