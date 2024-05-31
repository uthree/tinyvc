import argparse
import torch
from pathlib import Path
import numpy as np
from module.tinyvc import Encoder
from module.tinyvc import Decoder

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output-dir', default='onnx')
parser.add_argument('-encp', '--encoder-path', default='./models/encoder.pt')
parser.add_argument('-decp', '--decoder-path', default='./models/decoder.pt')
parser.add_argument('-opset', default=17, type=int)
args = parser.parse_args()

device = torch.device('cpu')

print("Loading models")
encoder = Encoder().to(device).eval()
encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

output_dir = Path(args.output_dir)
if not output_dir.exists():
    output_dir.mkdir()

print("Exporting encoder")
fft_bin = encoder.n_fft // 2 + 1
x = torch.randn(1, fft_bin, 100)
torch.onnx.export(
        encoder,
        x,
        output_dir / 'encoder.onnx',
        opset_version=args.opset,
        input_names=['spectrogram'],
        output_names=['content', 'f0'],
        dynamic_axes={
            'spectrogram': {0: 'batch_size', 2: 'length'}})


print("Exporting decoder")
content_channels = decoder.content_channels
z = torch.randn(1, content_channels, 100)
energy = torch.randn(1, 1, 100 * decoder.frame_size)
f0 = torch.randn(1, 1, 100)

torch.onnx.export(
        decoder,
        (z, f0, energy),
        output_dir / 'decoder.onnx',
        opset_version=args.opset,
        input_names=['content', 'f0', 'energy'],
        output_names=['waveform'],
        dynamic_axes={
            'content': {0: 'batch_size', 2: 'length'},
            'f0': {0: 'batch_size', 2: 'length'},
            'energy': {0: 'batch_size', 2: 'length'},})