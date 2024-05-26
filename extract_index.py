import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.utils.dataset import Dataset
from module.tinyvc import Encoder
from module.utils.spectrogram import spectrogram

parser = argparse.ArgumentParser(description="extract index")
parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-encp', '--encoder-path', default='models/encoder.pt')
parser.add_argument('-size', default=2048, type=int)
parser.add_argument('-o', '--output', default='models/index.pt')
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('--stride', default=4, type=int)

args = parser.parse_args()

device = torch.device(args.device) # use cpu because content encoder is lightweight.

encoder = Encoder().to(device).eval()
encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

def shuffle(tensor, dim):
    indices = torch.randperm(tensor.size(dim))
    shuffled_tensor = tensor.index_select(dim, indices)
    return shuffled_tensor

features = []
total_length = 0

print("Extracting...")
bar = tqdm(total=args.size)
for i, (wave, f0) in enumerate(dl):
    spec = spectrogram(wave, encoder.n_fft, encoder.hop_size)
    z, f0 = encoder.infer(spec)
    z = z.cpu()[:, :, ::args.stride]
    total_length += z.shape[2]
    features.append(z)
    bar.update(z.shape[2])
    if total_length > args.size:
        break

features = torch.cat(features, dim=2)
tgt = shuffle(features, dim=2)[:, :, :args.size]
print(f"Extracted {tgt.shape[2]} vectors")

print("Saving...")
torch.save(tgt, args.output)

print("Complete")
