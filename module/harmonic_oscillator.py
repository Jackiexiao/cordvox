import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from module.common import CausalConvNeXtStack


class DifferentiableOscillationControler(nn.Module):
    def __init__(self,
                 input_channels=80,
                 num_layers=6,
                 channels=256,
                 hidden_channels=512,
                 kernel_size=7,
                 num_harmonics=32,
                 segment_size=960,
                 sample_rate=48000):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.convnet = CausalConvNeXtStack(input_channels, channels, hidden_channels, kernel_size, num_harmonics+1, num_layers)

    def forward(self, x):
        x = self.convnet(x)
        musical_scale, magnitude = x.split([1, self.num_harmonics], dim=1)
        f0 = 440 * 2 ** musical_scale
        amp = torch.exp(matnitude.clamp_max(4.0))
        return f0, amp


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 num_harmonics=32,
                 segment_size=960,
                 sample_rate=48000):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.segment_size = segment_size
        self.sample_rate = sample_rate
    
    # f0: [N, 1, Lf], amps: [N, Nh, L]f, t0: float
    def forward(self, f0, amps, t0=0):
        N = amps.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = amps.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length
    
        # frequency multiplyer
        mul = (torch.arange(Nh, device=f0.device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)

        # Calculate formants
        formants = f0 * mul

        # Interpolate folmants
        formants = F.interpolate(formants, Lw, mode='linear')

        # Interpolate amp
        amps = F.interpolate(amps, Lw, mode='linear')

        # Generate harmonics
        dt = torch.cumsum(formants / self.sample_rate, dim=2)
        harmonics = torch.sin(2 * math.pi * dt + t0) * amps

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)

        return wave


