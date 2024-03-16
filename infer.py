import argparse
import os
import glob

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm

from module.encoder import Encoder
from module.decoder import Decoder
from module.instance_norm import instance_norm
from module.common import estimate_energy

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-encp', '--encoder-path', default='./models/encoder.pt')
parser.add_argument('-decp', '--decoder-path', default='./models/decoder.pt')
parser.add_argument('-t', '--target', default=0, type=int)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-p', '--pitch-shift', default=0.0, type=float)

args = parser.parse_args()

device = torch.device(args.device)
encoder = Encoder().to(device).eval()
encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

spk_id = torch.LongTensor([args.target]).to(device)

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

paths = []
support_formats = ['wav', 'ogg', 'mp3']
for fmt in support_formats:
    paths += glob.glob(os.path.join(args.inputs, "*." + fmt))

for i, path in enumerate(paths):
    print(f"Converting {path} ...")
    wf, sr = torchaudio.load(path)
    wf = resample(wf, sr, 24000)
    wf = wf.mean(dim=0, keepdim=True).to(device)

    phone, f0 = encoder.infer(wf)
    scale = 12 * torch.log2(f0 / 440)
    scale += args.pitch_shift
    f0 = 440 * (2 ** (scale / 12))

    energy = estimate_energy(wf, decoder.synthesizer.frame_size)
    phone = instance_norm(phone)
    wf = decoder.infer(phone, energy, spk_id, f0).cpu()

    file_name = f"{os.path.splitext(os.path.basename(path))[0]}"
    torchaudio.save(os.path.join(args.outputs, f"{file_name}.wav"), src=wf, sample_rate=24000)
