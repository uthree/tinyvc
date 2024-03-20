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
from module.convertor import Convertor


parser = argparse.ArgumentParser(description="realtime inference")
parser.add_argument('-i', '--input', default=0, type=int)
parser.add_argument('-o', '--output', default=0, type=int)
parser.add_argument('-l', '--loopback', default=-1, type=int)
parser.add_argument('-p', '--pitch-shift', default=0, type=float)

parser.add_argument('-t', '--target', default=0, type=int)
parser.add_argument('-c', '--chunk', default=1920, type=int)
parser.add_argument('-b', '--buffer', default=4, type=int)
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-sr', '--sample-rate', default=24000, type=int)
parser.add_argument('-ig', '--input-gain', default=0, type=float)
parser.add_argument('-og', '--output-gain', default=0, type=float)

args = parser.parse_args()

device = torch.device(args.device)

encoder = Encoder().to(device).eval()
encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
convertor = Convertor(encoder, decoder).to(device)


if args.index == 'NONE':
    print("Loading target...")
    wf, sr = torchaudio.load(args.target)
    wf = wf.to(device)
    wf = resample(wf, sr, 24000)
    wf = wf[:1]
    print("Encoding...")
    tgt = convertor.encode_target(wf)
else:
    print("Loading index...")
    tgt = torch.load(args.index).to(device)

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

BUFFER_SIZE = args.buffer * args.chunk
CHUNK_SIZE = args.chunk

# encoder speaker
spk = convertor.get_speaker_embedding(args.target, device)

pitch_shift = args.pitch_shift

# initialize buffer
buffer = convertor.init_buffer(BUFFER_SIZE, device)

# inference loop
print("Converting voice, Ctrl+C to stop conversion")
while True:
    chunk = stream_input.read(CHUNK_SIZE)
    chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    chunk = torch.from_numpy(chunk).to(device)
    chunk = chunk.unsqueeze(0) / 32768
    
    chunk = gain(chunk, args.input_gain)
    chunk, buffer = convertor.convert(chunk, buffer, spk, pitch_shift, device)
    chunk = gain(chunk, args.output_gain)

    chunk = chunk.cpu().numpy() * 32768
    chunk = chunk.astype(np.int16).tobytes()
    stream_output.write(chunk)
    if stream_loopback is not None:
        stream_loopback.write(chunk)
