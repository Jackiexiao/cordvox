import torch
import torch.nn as nn
import torch.nn.functional as F

from module.common import CausalConvNeXtStack

# Spectrogram based f0 estimator
class F0Estimator(nn.Module):
    def __init__(
            self,
            n_fft=3840,
            hop_length=960,
            max_freqency=4096,
            channels=256,
            hidden_channels=512,
            kernel_size=7,
            num_layers=6,
            ):
        super().__init__()
        self.stack = CausalConvNeXtStack(
                n_fft//2+1,
                channels,
                hidden_channels,
                kernel_size,
                max_freqency,
                num_layers)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        return self.estimate_from_spec(self.spec(x))

    def estimate_from_spec(self, x):
        return self.stack(x)

    def spec(self, x):
        return torch.stft(x, self.n_fft, self.hop_length, return_complex=True).abs()[:, :, 1:]

    def estimate(self, x):
        return torch.argmax(self.forward(x), dim=1, keepdim=True).to(torch.float)
