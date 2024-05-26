import argparse
import os
from pathlib import Path
import json

import torch
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.utils import estimate_f0


parser = argparse.ArgumentParser(description="preprocess")

parser.add_argument('input')
parser.add_argument('-o', '--output', '--dataset-cache', default='dataset_cache')
parser.add_argument('-len', '--length', default=48000, type=int)
parser.add_argument('--num-speakers', default=8192, type=int)
parser.add_argument('-m', '--max-files', default=-1, type=int)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('--f0-estimation', default='fcpe')

args = parser.parse_args()

device = torch.device(args.device)

input_parent = Path(args.input)
dataset_files = []

support_exts = ['mp3', 'wav', 'ogg']
for e in support_exts:
    dataset_files += list(input_parent.glob(f"**/*.{e}"))
if args.max_files != -1:
    dataset_files = dataset_files[:args.max_files]

# create output directory
output_parent = Path(args.output)
if not output_parent.exists():
    output_parent.mkdir()

counter = 0
for path in tqdm(dataset_files):
    tqdm.write(f"processing {str(path)}")
    parent_path = path.parent
    wf, sr = torchaudio.load(path)
    wf = wf.mean(dim=0, keepdim=True)
    wf = resample(wf, sr, 24000)
    # chunk
    chunks = wf.split(args.length, dim=1)
    for chunk in chunks:
        if chunk.shape[1] < args.length:
            # padding
            pad_len = args.length - chunk.shape[1]
            pad = torch.zeros(1, pad_len)
            chunk = torch.cat([chunk, pad], dim=1)

        # f0
        f0 = estimate_f0(chunk, algorithm=args.f0_estimation)

        # save
        output_pt_path = output_parent / f"{counter}.pt"
        obj = {
            "f0": f0
        }
        torch.save(obj, output_pt_path)
        output_wave_path = output_parent / f"{counter}.wav"
        torchaudio.save(output_wave_path, src=chunk, sample_rate=24000)
        counter += 1

print("complete!")
