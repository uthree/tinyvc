import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.dataset import Dataset
from module.encoder import Encoder
from module.common import spectrogram
from module.noise_generator import NoiseGenerator
from transformers import  HubertModel

parser = argparse.ArgumentParser(description="distillation of HuBERT-Base 4th layer / Pitch Estimation")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('--noises', default='NONE')
parser.add_argument('--hubert', default='rinna/japanese-hubert-base')
parser.add_argument('-path', '--path', default='models/encoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-fp16', default=False, type=bool)
args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    model = Encoder().to(device)
    if os.path.exists(args.path):
        model.load_state_dict(torch.load(args.path, map_location=device))
    return model

def save_models(model):
    print("Saving models...")
    torch.save(model.state_dict(), args.path)
    print("Complete!")

device = torch.device(args.device)
ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

model = load_or_init_models(device)
Opt = optim.AdamW(model.parameters(), lr=args.learning_rate)

hubert = HubertModel.from_pretrained(args.hubert).to(device).eval()

weight = torch.ones(model.num_f0_classes)
weight[0] = 1e-3
cross_entropy_f0 = nn.CrossEntropyLoss(weight).to(device)

# for noise reduction
if args.noises != None:
    add_noise = True
    noise_generator = NoiseGenerator(args.noises)
else:
    add_noise = False

step_count = 0
for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        f0 = f0.to(device)

        Opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave_16k = resample(wave, 24000, 16000)
            # get pseudo-label
            hubert_features = hubert(wave_16k, output_hidden_states=True).hidden_states[4]
            hubert_features = hubert_features.transpose(1, 2)
            f0_label = model.freq2id(f0).squeeze(1)

            # data argumentation
            wave = wave * torch.rand(N, 1, device=device) * 2

            # add noise(optional)
            if add_noise:
                wave = noise_generator.add_noise(wave)

            # estimate
            z, f0_out = model(spectrogram(wave, model.n_fft, model.hop_size))

            # loss
            loss_distill = (z - F.interpolate(hubert_features, z.shape[2])).abs().mean()
            loss_f0 = cross_entropy_f0(f0_out, f0_label)
            loss = loss_f0 + loss_distill * 45

        scaler.scale(loss).backward()
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Epoch: {epoch}, Step {step_count}, F0 Est.: {loss_f0.item():.4f}, Hubert Distill.: {loss_distill.item():.4f}")

        bar.update(N)

        if batch % 500 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)
