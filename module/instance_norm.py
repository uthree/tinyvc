import torch


def instance_norm(x, eps=1e-6):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True) + eps
    x = (x - mean) / std
    return x
