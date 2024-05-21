import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample, gain

import numpy as np
import pyaudio

from module.encoder import Encoder
from module.decoder import Decoder
from module.generator import Generator, StreamInfer


parser = argparse.ArgumentParser(description="realtime inference")
parser.add_argument('-encp', '--encoder-path', default='./models/encoder.pt')
parser.add_argument('-decp', '--decoder-path', default='./models/decoder.pt')
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-idx', '--index', default='NONE')
parser.add_argument('-p', '--pitch-shift', default=0, type=float)
parser.add_argument('-t', '--target', default='target.wav')
parser.add_argument('-c', '--chunk', default=1920, type=int)
parser.add_argument('-e', '--extra', default=1920, type=int)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-sr', '--sample-rate', default=24000, type=int)
parser.add_argument('-ig', '--input-gain', default=0, type=float)
parser.add_argument('-og', '--output-gain', default=0, type=float)
parser.add_argument('-f0-est', '--f0-estimation', default='default', choices=['default', 'fcpe', 'dio', 'harvest'])

args = parser.parse_args()

device = torch.device(args.device)

encoder = Encoder().to(device).eval()
encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
generator = Generator(encoder, decoder).to(device)
stream_infer = StreamInfer(generator, pitch_shift=args.pitch_shift, block_size=args.chunk, device=device, extra_size=args.extra, f0_estimation=args.f0_estimation)

audio = pyaudio.PyAudio()

stream_input = audio.open(
        format=pyaudio.paInt16,
        rate=args.sample_rate,
        channels=1,
        input_device_index=args.input,
        input=True)
stream_output = audio.open(
        format=pyaudio.paInt16,
        rate=args.sample_rate, 
        channels=1,
        output_device_index=args.output,
        output=True)
stream_loopback = audio.open(
        format=pyaudio.paInt16,
        rate=args.sample_rate, 
        channels=1,
        output_device_index=args.loopback,
        output=True) if args.loopback != -1 else None

# load target
if args.index == 'NONE':
    wf, sr = torchaudio.load(args.target)
    wf = resample(wf, sr, 24000).to(device)
    tgt = generator.encode_target(wf)
else:
    tgt = torch.load(args.index).to(device)
stream_infer.target = tgt

pitch_shift = args.pitch_shift

# initialize buffer
stream_infer.init_buffer()

# inference loop
print("Converting voice, Ctrl+C to stop conversion")
while True:
    chunk = stream_input.read(args.chunk)
    chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    chunk = torch.from_numpy(chunk).to(device)
    chunk = chunk / 32768
    
    chunk = gain(chunk, args.input_gain)
    chunk = stream_infer.audio_callback(chunk)
    chunk = gain(chunk, args.output_gain)

    chunk = chunk.cpu().numpy() * 32768
    chunk = chunk.astype(np.int16).tobytes()
    stream_output.write(chunk)
    if stream_loopback is not None:
        stream_loopback.write(chunk)
