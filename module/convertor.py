import torch
import torch.nn as nn
import torch.nn.functional as F


from .encoder import Encoder
from .decoder import Decoder, oscillate_harmonics
from .instance_norm import incremental_norm
from .common import estimate_energy


# Realtime convertor
class Convertor(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def get_speaker_embedding(self, spk_id: int, device=torch.device('cpu')):
        spk_id = torch.LongTensor([spk_id]).to(device)
        return self.decoder.speaker_embedding(spk_id)

    def init_buffer(self, buffer_size, device=torch.device('cpu')):
        input_buffer = torch.zeros(1, buffer_size, device=device)
        output_buffer = torch.zeros(1, buffer_size, device=device)
        norm_buffer = None
        return (input_buffer, output_buffer, norm_buffer)

    def convert(self, chunk, buffer, spk, pitch_shift=0, device=torch.device('cpu')):
        frame_size = self.decoder.synthesizer.frame_size
        num_harmonics = self.decoder.synthesizer.num_harmonics
        sample_rate = self.decoder.synthesizer.sample_rate
        chunk_size = chunk.shape[1]

        input_buffer, output_buffer, norm_buffer = buffer
        buffer_size = input_buffer.shape[1]

        with torch.inference_mode():
            input_buffer = torch.cat([input_buffer, chunk], dim=1)[:, -buffer_size:]
            x = input_buffer

            # estimate energy, encode content, estimate pitch
            energy = estimate_energy(x, frame_size)
            z, f0 = self.encoder.infer(x)

            # normalize content
            z, norm_buffer = incremental_norm(z, norm_buffer)

            # pitch shift
            scale = 12 * torch.log2(f0 / 440)
            scale += pitch_shift
            f0 = 440 * (2 ** (scale / 12))

            # synthesize new waveform
            src = oscillate_harmonics(f0, frame_size, sample_rate, num_harmonics)
            y = self.decoder.synthesizer(z, energy, spk, src)

            # cross fade
            y_prev = torch.cat([output_buffer, torch.zeros(1, chunk_size, device=device)], dim=1)[:, chunk_size:]
            alpha = torch.cat([
                torch.linspace(0.0, 1.0, buffer_size - chunk_size, device=device),
                torch.ones(chunk_size, device=device)]).unsqueeze(0)

            output = y_prev * (1 - alpha) + y * alpha
            output_buffer = y

            # cut center
            left = buffer_size // 2 - chunk_size // 2
            right = buffer_size // 2 + chunk_size // 2
            chunk = output[:, left:right]

        buffer = (input_buffer, output_buffer, norm_buffer)
        return chunk, buffer

