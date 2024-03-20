import torch
import torch.nn.functional as F


def instance_norm(x, eps=1e-6):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + eps
    x = (x - mean) / std
    return x


def incremental_norm(x, buffer=None, eps=1e-6, alpha=0.95):
    if buffer is None:
        buffer_mean = x.mean(dim=2, keepdim=True)
        buffer_std = x.std(dim=2, keepdim=True) + eps
    else:
        buffer_mean, buffer_std = buffer
        # update buffer
        buffer_mean = buffer_mean * alpha + x.mean(dim=2, keepdim=True) * (1 - alpha)
        buffer_std = buffer_std * alpha + x.std(dim=2, keepdim=True) * (1 - alpha)
        buffer_std = F.relu(buffer_std) + eps

    x = (x - buffer_mean) / buffer_std
    return x, (buffer_mean, buffer_std)
