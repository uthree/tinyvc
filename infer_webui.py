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
from module.convertor import Convertor

import gradio as gr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', default="./inputs/")
    parser.add_argument('-o', '--outputs', default="./outputs/")
    parser.add_argument('-encp', '--encoder-path', default='./models/encoder.pt')
    parser.add_argument('-decp', '--decoder-path', default='./models/decoder.pt')
    parser.add_argument('-f0-est', '--f0-estimation', default='default')
    parser.add_argument('-d', '--device', default='auto')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        if torch.backends.mps.is_available():
            device = torch.device('mps')
    else:
        device = torch.device(args.device)
    encoder = Encoder().to(device).eval()
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
    convertor = Convertor(encoder, decoder).to(device)

    def audio_to_tensor(input_audio):
        input_sr, input_wf = input_audio
        input_wf = torch.from_numpy(input_wf).unsqueeze(0).to(torch.float)
        if input_wf.ndim == 3:
            input_wf = input_wf.sum(dim=2)
        input_wf = input_wf / input_wf.abs().max()
        input_wf = resample(input_wf, input_sr, 24000).to(device)
        return input_wf

    def convert(input_audio, target_audio, pitch_shift):
        input_wf = audio_to_tensor(input_audio)
        target_wf = audio_to_tensor(target_audio)

        tgt = convertor.encode_target(target_wf)
        output_wf = convertor.convert(input_wf, tgt, pitch_shift, device=device)

        output_wf = output_wf.clamp(-1.0, 1.0)
        output_wf = output_wf * 32768.0
        output_wf = output_wf.to(torch.int16).squeeze(0).cpu().numpy()
        return (24000, output_wf)
    
    demo = gr.Interface(
        convert,
        inputs=[
            gr.Audio(label="Input"),
            gr.Audio(label="Target"),
            gr.Slider(-24.0, 24.0, 0.0, label="Pitch Shift")
        ],
        outputs=[
            gr.Audio()
        ]
    )

    demo.launch()