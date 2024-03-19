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
from transformers import  HubertForCTC

parser = argparse.ArgumentParser(description="distillation of HuBERT CTC / Pitch Estimation")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('--hubert-ctc', default='facebook/hubert-large-ls960-ft')
parser.add_argument('-path', '--path', default='models/encoder.pt')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=60, type=int)
parser.add_argument('-b', '--batch-size', default=2, type=int)
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

hubert = HubertForCTC.from_pretrained(args.hubert_ctc).to(device).eval()

cross_entropy_phone = nn.CrossEntropyLoss().to(device)
weight = torch.ones(model.num_f0_classes)
weight[0] = 1e-3
cross_entropy_f0 = nn.CrossEntropyLoss(weight).to(device)

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
            logits = hubert(wave_16k).logits
            phone_label = torch.argmax(logits, dim=-1)
            f0_label = model.freq2id(f0).squeeze(1)

            # estimate
            z, f0_out = model(spectrogram(wave, model.n_fft, model.hop_size))
            phone_out = model.to_phone(z)

            # loss
            length = phone_label.shape[1]
            loss_phone = cross_entropy_phone(F.interpolate(phone_out, length, mode='linear'), phone_label) 
            loss_f0 = cross_entropy_f0(f0_out, f0_label)
            loss = loss_f0 + loss_phone * 10

        scaler.scale(loss).backward()
        scaler.step(Opt)

        scaler.update()

        step_count += 1

        tqdm.write(f"Epoch: {epoch}, Step {step_count}, F0 Est.: {loss_f0.item():.4f}, CTC Distill.: {loss_phone.item():.4f}")

        bar.update(N)

        if batch % 500 == 0:
            save_models(model)

print("Training Complete!")
save_models(model)
