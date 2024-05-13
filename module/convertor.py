import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder, match_features
from .common import estimate_energy
from .f0_estimation import estimate_f0

from tqdm import tqdm


# Realtime convertor
class Convertor(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode_target(self, waveform):
        z, f0 = self.encoder.infer(waveform)
        return z

    def init_buffer(self, buffer_size, extra_size=0, device=torch.device('cpu')):
        input_buffer = torch.zeros(1, buffer_size + extra_size, device=device)
        output_buffer = torch.zeros(1, buffer_size, device=device)

        return (input_buffer, output_buffer)

    def convert_chunk(self, wf, tgt, pitch_shift, device=torch.device('cpu'), f0_estimation='default'):
        frame_size = self.decoder.frame_size

        x = wf.to(device)

        with torch.inference_mode():
            # estimate energy, encode content, estimate pitch
            energy = estimate_energy(x, frame_size)
            z, f0 = self.encoder.infer(x)
            if f0_estimation != 'default':
                f0 = estimate_f0(x, algorithm=f0_estimation)

            # pitch shift
            scale = 12 * torch.log2(f0 / 440 + 1e-6)
            scale += pitch_shift
            f0 = 440 * (2 ** (scale / 12))

            # match features
            z = match_features(z, tgt)

            # synthesize new wave
            y = self.decoder.infer(z, energy, f0).squeeze(1)

            return y

    @torch.inference_mode()
    def streaming_convert(self, chunk, buffer, tgt, pitch_shift=0, device=torch.device('cpu'), f0_estimation='default'):
        chunk_size = chunk.shape[1]
        device = chunk.device

        input_buffer, output_buffer= buffer
        input_buffer_size = input_buffer.shape[1]
        output_buffer_size = output_buffer.shape[1]

        input_buffer = torch.cat([input_buffer, chunk], dim=1)[:, -input_buffer_size:]
        x = input_buffer

        y = self.convert_chunk(x, tgt, pitch_shift, device, f0_estimation)
        y = y[:, -output_buffer_size:]

        # cross fade
        y_prev = torch.cat([output_buffer, torch.zeros(1, chunk_size, device=device)], dim=1)[:, chunk_size:]
        window = torch.hann_window(y.shape[1], device=y.device).unsqueeze(0)

        output_buffer = y_prev + y * window

        # cut left
        chunk = output_buffer[:, :chunk_size]

        buffer = (input_buffer, output_buffer)
        return chunk, buffer

    def convert(self,
                wf,
                tgt,
                pitch_shift,
                device=torch.device('cpu'),
                f0_estimation='default',
                chunk_size=7680,
                buffer_size=15360,
                no_chunking=False):
        if wf.shape[1] < chunk_size or no_chunking:
            wf = wf.to(device)
            wf = self.convert_chunk(wf, tgt, pitch_shift, device, f0_estimation)
            wf = wf.cpu()
        else:
            wf = wf.cpu()
            chunks = torch.split(wf, chunk_size, dim=1)
            results = []
            buffer = self.init_buffer(buffer_size, device=device)
            for chunk in tqdm(chunks):
                if chunk.shape[1] < chunk_size:
                    chunk = torch.cat([chunk, torch.zeros(1, chunk_size - chunk.shape[1])], dim=1)
                chunk = chunk.to(device)
                out, buffer = self.streaming_convert(chunk, buffer, tgt, pitch_shift, device, f0_estimation)
                out = out.cpu()
                results.append(out)
            wf = torch.cat(results, dim=1)
        return wf