import torch
import torch.nn as F
import torch.nn.functional as F


# estimate energy
# wave: [BatchSize, Length]
# Output: [BatchSize, 1, Length]
def estimate_energy(wave,
           frame_size=64):
    wave_length = wave.shape[1]
    energy = F.max_pool1d((wave ** 2).unsqueeze(1), frame_size)
    energy = F.interpolate(energy, wave_length, mode='linear')
    return energy