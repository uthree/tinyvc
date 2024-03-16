import torch


def instance_norm(x):
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    x = (x - mean) / std
    return x
