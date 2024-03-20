import torch
import torch.nn as nn
import torch.nn.functional as F


from .encoder import Encoder
from .decoder import Decoder, match_features
from .common import estimate_energy


# Realtime convertor
class Convertor(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode_target(self, waveform):
        z, f0 = self.encoder.infer(waveform)
        return z

    def init_buffer(self, buffer_size, device=torch.device('cpu')):
        input_buffer = torch.zeros(1, buffer_size, device=device)
        output_buffer = torch.zeros(1, buffer_size, device=device)
        return (input_buffer, output_buffer)

    def convert_without_chunking(self, wf, tgt, pitch_shift, device=torch.device('cpu')):
        frame_size = self.decoder.frame_size
        sample_rate = self.decoder.sample_rate

        x = wf.to(device)

        with torch.inference_mode():
            # estimate energy, encode content, estimate pitch
            energy = estimate_energy(x, frame_size)
            z, f0 = self.encoder.infer(x)

            # pitch shift
            scale = 12 * torch.log2(f0 / 440)
            scale += pitch_shift
            f0 = 440 * (2 ** (scale / 12))

            # match features
            z = match_features(z, tgt)

            # synthesize new wave
            y_dsp = self.decoder.source_net.synthesize(z, energy, f0)
            y = self.decoder.filter_net.synthesize(z, energy, y_dsp).squeeze(1)

            return y


    def convert(self, chunk, buffer, tgt, pitch_shift=0, device=torch.device('cpu')):
        frame_size = self.decoder.frame_size
        sample_rate = self.decoder.sample_rate
        chunk_size = chunk.shape[1]

        input_buffer, output_buffer = buffer
        buffer_size = input_buffer.shape[1]

        with torch.inference_mode():
            input_buffer = torch.cat([input_buffer, chunk], dim=1)[:, -buffer_size:]
            x = input_buffer

            # estimate energy, encode content, estimate pitch
            energy = estimate_energy(x, frame_size)
            z, f0 = self.encoder.infer(x)

            # pitch shift
            scale = 12 * torch.log2(f0 / 440)
            scale += pitch_shift
            f0 = 440 * (2 ** (scale / 12))

            # match features
            z = match_features(z, tgt)

            # synthesize new wave
            y_dsp = self.decoder.source_net.synthesize(z, energy, f0)
            y = self.decoder.filter_net.synthesize(z, energy, y_dsp).squeeze(1)

            # cross fade
            y_prev = torch.cat([output_buffer, torch.zeros(1, chunk_size, device=device)], dim=1)[:, chunk_size:]
            alpha = torch.cat([
                torch.zeros(buffer_size - chunk_size * 2, device=device),
                torch.linspace(0.0, 1.0, chunk_size, device=device),
                torch.ones(chunk_size, device=device)]).unsqueeze(0)

            output_buffer = y_prev * (1 - alpha) + y * alpha

            # cut center
            left = buffer_size // 2 - chunk_size // 2
            right = buffer_size // 2 + chunk_size // 2
            chunk = output_buffer[:, left:right]

        buffer = (input_buffer, output_buffer)
        return chunk, buffer

