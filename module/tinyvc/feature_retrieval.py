import torch
import torch.nn as nn
import torch.nn.functional as F

# Convert style based kNN.
# Warning: this method is not optimized.
# Do not give long sequence. computing complexy is quadratic.
# 
# source: [BatchSize, Channels, Length]
# reference: [BatchSize, Channels, Length]
# k: int
# alpha: float (0.0 ~ 1.0)
# metrics: one of ['IP', 'L2', 'cos'], 'IP' means innner product, 'L2' means euclid distance, 'cos' means cosine similarity
# Output: [BatchSize, Channels, Length]
def match_features(source, reference, k=4, alpha=0.0, metrics='cos'):
    input_data = source

    source = source.transpose(1, 2)
    reference = reference.transpose(1, 2)
    if metrics == 'IP':
        sims = torch.bmm(source, reference.transpose(1, 2))
    elif metrics == 'L2':
        sims = -torch.cdist(source, reference)
    elif metrics == 'cos':
        reference_norm = torch.norm(reference, dim=2, keepdim=True, p=2) + 1e-6
        source_norm = torch.norm(source, dim=2, keepdim=True, p=2) + 1e-6
        sims = torch.bmm(source / source_norm, (reference / reference_norm).transpose(1, 2))
    best = torch.topk(sims, k, dim=2)

    result = torch.stack([reference[n][best.indices[n]] for n in range(source.shape[0])], dim=0).mean(dim=2)
    result = result.transpose(1, 2)

    return result * (1-alpha) + input_data * alpha

