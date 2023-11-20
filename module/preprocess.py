import torch
import torch.nn as nn
import torchaudio


class MFCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
                sample_rate=48000,
                n_mfcc=80,
                log_mels=True,
                melkwargs={"n_fft": 3840, "hop_length": 960}
                )

    def forward(self, x):
        return self.mfcc(x)[:, :, 1:]


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=48000,
                n_fft=3840,
                hop_length=960,
                n_mels=80
                )
    
    def forward(self, x):
        return torch.log(self.to_mel(x)[:, :, 1:] + 1e-5)


