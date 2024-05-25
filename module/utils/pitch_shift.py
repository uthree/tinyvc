import torch
import torch.nn as nn
import torch.nn.functional as F

def frqeuency_to_midi(f):
    return torch.log2(F.relu(f / 440) + 1e-6) * 12 + 69

def midi_to_frequency(n):
    return 440 * 2 ** ((n - 69) / 12)

def shift_frequency(f0, shift):
    pitch = frqeuency_to_midi(f0)
    pitch += shift
    f0 = midi_to_frequency(pitch)
    return f0