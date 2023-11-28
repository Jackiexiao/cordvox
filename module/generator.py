import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from module.common import CausalConv1d, CausalConvNeXtStack

LRELU_SLOPE = 0.1


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 input_channels=512,
                 num_harmonics=64,
                 segment_size=960,
                 sample_rate=48000,
                 f0_max=4000,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        
        self.to_amps = nn.Conv1d(input_channels, num_harmonics, 1)

    # x: [N, input_channels, Lf] t0: float
    def forward(self, x, f0, t0=0):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length

        # Ignore lower than 20Hz
        f0 = f0.clamp_min(20)

        # to amplitudes
        amps = self.to_amps(x)

        # magnitude to amplitude
        amps = torch.exp(amps)
    
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
        harmonics = torch.sin(2 * math.pi * dt) * amps

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)

        return wave


class ModulatedCausalConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, condition_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.conv = CausalConv1d(input_channels, output_channels, kernel_size, dilation)
        self.to_scale = nn.Conv1d(condition_channels, input_channels, 1)
        self.to_shift = nn.Conv1d(condition_channels, input_channels, 1)

    def forward(self, x, c):
        scale = self.to_scale(c)
        shift = self.to_shift(c)
        scale = F.interpolate(scale, x.shape[2], mode='linear')
        shift = F.interpolate(shift, x.shape[2], mode='linear')
        x = x * scale + shift
        x = self.conv(x)
        return x


class FilterBlock(nn.Module):
    def __init__(self, input_channels, output_channels, condition_channels, kernel_size=5, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([])
        self.convs.append(ModulatedCausalConv1d(input_channels, output_channels, condition_channels, kernel_size, 1))
        for d in range(num_layers-1):
            self.convs.append(ModulatedCausalConv1d(output_channels, output_channels, condition_channels, kernel_size, 2**(d+1)))

    def forward(self, x, c):
        for conv in self.convs:
            F.gelu(x)
            x = conv(x, c)
        return x


class Filter(nn.Module):
    def __init__(
            self,
            feat_channels=512,
            rates=[3, 5, 8, 8],
            channels=[32, 64, 128, 256],
            kernel_size=5,
            num_layers=3
            ):
        super().__init__()
        self.source_in = nn.Conv1d(1, channels[0], 7, 1, 3)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.blocks = nn.ModuleList([])

        channels_nexts = channels[1:] + [channels[-1]]
        for c, c_next, r in zip(channels, channels_nexts, rates):
            self.downs.append(nn.Conv1d(c, c_next, r, r, 0))

        channels = list(reversed(channels))
        rates = list(reversed(rates))
        channels_prevs = [channels[0]] + channels[:-1]
        
        for c, c_prev, r in zip(channels, channels_prevs, rates):
            self.ups.append(nn.ConvTranspose1d(c_prev, c, r, r, 0))
            self.blocks.append(FilterBlock(c, c, feat_channels, kernel_size, num_layers))

        self.source_out = nn.Conv1d(c, 1, 7, 1, 3)
    
    # x: [N, 1, Lw], c: [N, channels, Lf]
    def forward(self, x, c):
        skips = []
        x = self.source_in(x)
        for d in self.downs:
            x = d(x)
            skips.append(x)

        for u, b, s in zip(self.ups, self.blocks, reversed(skips)):
            x = u(x + s)
            x = b(x, c)
        x = self.source_out(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = CausalConvNeXtStack(80, 512, 1024, 7, 512, 6, 1)
        self.harmonic_oscillator = HarmonicOscillator()
        self.filter = Filter()

    def forward(self, x, f0, t0=0, harmonics_scale=1):
        x = self.feature_extractor(x)
        source = self.harmonic_oscillator(x, f0, t0) * harmonics_scale
        out = self.filter(source, x)
        out = out.squeeze(1)
        return out
