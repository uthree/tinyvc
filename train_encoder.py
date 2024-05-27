import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.utils.dataset import Dataset
from module.tinyvc.encoder import Encoder
from module.utils.spectrogram import spectrogram
from module.utils.noise_generator import NoiseGenerator
from transformers import WavLMModel

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="distillation of WavLM-Base 4th layer / Pitch Estimation")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('--noises', default='NONE')
parser.add_argument('--wavlm', default='microsoft/wavlm-base-plus')
parser.add_argument('-path', '--path', default='models/encoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--log-interval', default=50, type=int)
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

hubert = WavLMModel.from_pretrained(args.wavlm).to(device).eval()

writer = SummaryWriter(log_dir="./logs")

# for noise reduction
if args.noises != 'NONE':
    add_noise = True
    noise_generator = NoiseGenerator(args.noises)
else:
    add_noise = False

cross_entropy_weight = torch.ones(model.pitch_estimator.num_classes, device=device)
cross_entropy_weight[0] = 0.1

step_count = 0
for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0) in enumerate(dl):
        N = wave.shape[0]
        wave = wave.to(device)
        f0 = f0.to(device)

        Opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave_16k = resample(wave, 24000, 16000)
            # get pseudo-label
            hubert_features = hubert(wave_16k, output_hidden_states=True).hidden_states[4]
            hubert_features = hubert_features.transpose(1, 2)
            f0_label = model.pitch_estimator.freq2id(f0).squeeze(1)

            # data argumentation
            wave = wave * torch.rand(N, 1, device=device) * 2

            # add noise(optional)
            if add_noise:
                wave = noise_generator.add_noise(wave)

            # estimate
            z, f0_out = model(spectrogram(wave, model.n_fft, model.hop_size))

            # loss
            loss_distill = (z - F.interpolate(hubert_features, z.shape[2])).abs().mean()
            loss_f0 = F.cross_entropy(f0_out, f0_label, weight=cross_entropy_weight)
            loss = loss_f0 + loss_distill * 45

            # logging
            if step_count % args.log_interval == 0:
                writer.add_scalar("loss/Pitch Estimation", loss_f0.item(), step_count)
                writer.add_scalar("loss/Distillation", loss_distill.item(), step_count)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Epoch: {epoch}, Step {step_count}, F0 Est.: {loss_f0.item():.4f}, Distill.: {loss_distill.item():.4f}")

        bar.update(N)

        if batch % 500 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)
writer.close()
