import argparse
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from module.utils.dataset import Dataset
from module.utils import spectrogram, estimate_energy
from module.utils.loss import MultiScaleSTFTLoss, LogMelSpectrogramLoss
from module.tinyvc import Encoder, Decoder, Discriminator, match_features

from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('--save-interval', default=500, type=int)
parser.add_argument('-spec-type', choices=['ms-stft', 'mel'], default='ms-stft')
parser.add_argument('-fp16', default=False, type=bool)

parser.add_argument('--weight-adv', default=2.0, type=float)
parser.add_argument('--weight-dsp', default=1.0, type=float)
parser.add_argument('--weight-spec', default=1.0, type=float)
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

if args.spec_type == 'ms-stft':
    SpecLoss = MultiScaleSTFTLoss().to(device)
else:
    SpecLoss = LogMelSpectrogramLoss().to(device)

OptG = optim.AdamW(decoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptD = optim.AdamW(discriminator.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

writer = SummaryWriter(log_dir="./logs")

# Training
step_count = 0

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, (wave, f0) in enumerate(dl):
        d_join = (args.discriminator_join <= step_count)

        N = wave.shape[0]
        wave = wave.to(device) * torch.rand(N, 1, device=device) * 2.0
        f0 = f0.to(device)
        spec = spectrogram(wave, encoder.n_fft, encoder.hop_size)

        # train Generator
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            z, f0 = encoder.infer(spec)
            z_fake = match_features(z, z).detach()
            energy = estimate_energy(wave, decoder.frame_size)

            amps, kernel = decoder.source_net(z_fake, f0, energy)
            dsp_out = decoder.dsp(f0, amps, kernel)
            fake = decoder.filter_net(z_fake, f0, energy, dsp_out)

            loss_dsp = SpecLoss(dsp_out.sum(1), wave)
            loss_spec = SpecLoss(fake.squeeze(1), wave)

            fake = fake.squeeze(1)
            if d_join:
                loss_adv = 0
                fake = fake.clamp(-1.0, 1.0)
                _, feats_real = discriminator(center(wave))
                logits, feats_fake = discriminator(center(fake))
                for logit in logits:
                    loss_adv += (logit ** 2).mean() / len(logits)
                loss_feat = 0
                for r, f in zip(feats_real, feats_fake):
                    loss_feat += (r - f).abs().mean() / len(feats_real)
                loss_g = loss_adv * args.weight_adv + loss_spec * args.weight_spec + loss_feat * args.weight_feat + loss_dsp * args.weight_dsp

                if step_count % 50 == 0:
                    writer.add_scalar("loss/Feature Matching", loss_feat.item(), step_count)
                    writer.add_scalar("loss/Generator Adversarial", loss_adv.item(), step_count)
            else:
                loss_g = loss_spec * args.weight_spec + loss_dsp * args.weight_dsp
            
            if step_count % 50 == 0:
                writer.add_scalar("loss/Spectrogram", loss_spec.item(), step_count)
                writer.add_scalar("loss/DSP", loss_dsp.item(), step_count)

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
            if step_count % 50 == 0:
                writer.add_scalar("loss/Discriminator Adversarial", loss_d.item(), step_count)
        
        if d_join:
            tqdm.write(f"Epoch: {epoch}, Step: {step_count}, Dis.: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Spec.: {loss_spec.item():.4f}, Feat. {loss_feat.item():.4f}, DSP: {loss_dsp.item():.4f}")
        else:
            tqdm.write(f"Epoch: {epoch}, Step: {step_count}, Spec.: {loss_spec.item():.4f}, DSP: {loss_dsp.item():.4f}")

        scaler.update()
        step_count += 1
        bar.update(N)

        if batch % args.save_interval == 0:
            save_models(decoder, discriminator)
    if step_count >= args.max_steps:
        break

print("Training Complete!")
save_models(decoder, discriminator)
writer.close()