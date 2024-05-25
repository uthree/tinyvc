import torch
import torchaudio
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_path = 'dataset_cache'):
        super().__init__()
        self.dir_path = Path(dir_path)
        self.len = len(list(self.dir_path.glob("*.wav")))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        obj = torch.load(self.dir_path / f"{idx}.pt")
        wf, _ = torchaudio.load(self.dir_path / f"{idx}.wav")
        wf = wf.mean(dim=0)
        f0 = obj["f0"].squeeze(0)
        return wf, f0
