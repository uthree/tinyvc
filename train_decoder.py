import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.dataset import Dataset
from module.common import spectrogram, estimate_energy
from module.loss import MultiScaleSTFTLoss
from module.encoder import Encoder
from module.decoder import Decoder, oscillate_harmonics
from module.discriminator import Discriminator
from module.instance_norm import instance_norm


parser = argparse.ArgumentParser(description="train voice conversion model")

parser.add_argument('--dataset-cache', default='dataset_cache')
parser.add_argument('-encp', '--encoder-path', default='models/encoder.pt')
parser.add_argument('-decp', '--decoder-path', default='models/decoder.pt')
parser.add_argument('-dip', '--discriminator-path', default='models/discriminator.pt')
parser.add_argument('-d-join', '--discriminator-join', default=100000, type=int)
parser.add_argument('-step', '--max-steps', default=300000, type=int)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=100000, type=int)
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--save-interval', default=100, type=int)
parser.add_argument('-fp16', default=False, type=bool)

parser.add_argument('--weight-adv', default=1.0, type=float)
parser.add_argument('--weight-aux', default=1.0, type=float)
parser.add_argument('--weight-feat', default=2.0, type=float)

args = parser.parse_args()

def load_or_init_models(device=torch.device('cpu')):
    dec = Decoder().to(device)
    dis = Discriminator().to(device)
    if os.path.exists(args.decoder_path):
        dec.load_state_dict(torch.load(args.decoder_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return dec, dis


def save_models(dec, dis):
    print("Saving models...")
    torch.save(dec.state_dict(), args.decoder_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("Complete!")


def center(wave, length=8000):
    c = wave.shape[1] // 2
    half_len = length // 2
    return wave[:, c-half_len:c+half_len]


device = torch.device(args.device)
decoder, discriminator = load_or_init_models(device)
encoder = Encoder().to(device).eval()
encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))

ds = Dataset(args.dataset_cache)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

AuxLoss = MultiScaleSTFTLoss().to(device)

OptG = optim.AdamW(decoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptD = optim.AdamW(discriminator.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0, spk_id) in enumerate(dl):
        d_join = (args.discriminator_join <= step_count)

        N = wave.shape[0]
        wave = wave.to(device) * torch.rand(N, 1, device=device) * 2.0
        f0 = f0.to(device)
        spk_id = spk_id.to(device)

        # train Generator
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            phone, f0 = encoder.infer(wave)
            phone = instance_norm(phone)
            energy = estimate_energy(wave, decoder.synthesizer.frame_size)
            src = oscillate_harmonics(
                    f0,
                    decoder.synthesizer.frame_size,
                    decoder.synthesizer.sample_rate,
                    decoder.synthesizer.num_harmonics)
            spk = decoder.speaker_embedding(spk_id)
            fake = decoder.synthesizer(phone, energy, spk, src)
            loss_aux = AuxLoss(fake, wave)

            if d_join:
                loss_adv = 0
                fake = fake.clamp(-1.0, 1.0)
                logits, feats_fake = discriminator(center(fake))
                _, feats_real = discriminator(center(wave))
                for logit in logits:
                    loss_adv += (logit ** 2).mean() / len(logits)
                loss_feat = 0
                for r, f in zip(feats_real, feats_fake):
                    loss_feat += (r - f).abs().mean() / len(feats_fake)
                loss_g = loss_adv * args.weight_adv + loss_aux * args.weight_aux + loss_feat * args.weight_feat
            else:
                loss_g = loss_aux * args.weight_aux

        scaler.scale(loss_g).backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        scaler.step(OptG)

        if d_join:
            fake = fake.detach()
            OptD.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                loss_d = 0
                logits, _ = discriminator(center(wave))
                for logit in logits:
                    loss_d += (logit ** 2).mean() / len(logits)
                logits, _ = discriminator(center(fake))
                for logit in logits:
                    loss_d += ((logit - 1) ** 2).mean() / len(logits)

            scaler.scale(loss_d).backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler.step(OptD)
        
        if d_join:
            tqdm.write(f"Epoch: {epoch}, Step: {step_count}, Dis.: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Aux.: {loss_aux.item():.4f}, Feat. {loss_feat.item():.4f}")
        else:
            tqdm.write(f"Epoch: {epoch}, Step: {step_count}, Aux.: {loss_aux.item():.4f}")

        scaler.update()
        step_count += 1
        bar.update(N)

        if batch % args.save_interval == 0:
            save_models(decoder, discriminator)
    if step_count >= args.max_steps:
        break

print("Training Complete!")
save_models(decoder, discriminator)
