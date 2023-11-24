import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from module.common import CausalConvNeXtStack, CausalConv1d, DilatedCausalConvStack

LRELU_SLOPE = 0.1


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 channels=256,
                 num_harmonics=64,
                 segment_size=960,
                 sample_rate=48000,
                 f0_max=4000,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.segment_size = segment_size
        self.sample_rate = sample_rate

        self.to_amps = nn.Conv1d(channels, num_harmonics, 1)
    
    # x: [N, input_channels, Lf] t0: float
    def forward(self, x, f0, t0=0):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length

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
        harmonics = torch.sin(2 * math.pi * dt + t0) * amps

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)

        return wave


class NoiseGenerator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            upsample_rates=[10, 8, 4, 3],
            channels=[64, 32, 16, 8],
            kernel_size=5,
            num_layers=3
            ):
        super().__init__()
        c0 = channels[0]
        c_last = channels[-1]
        self.input_layer = CausalConv1d(input_channels, c0, kernel_size)
        self.to_gains = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.convs = nn.ModuleList([])
        for i in range(len(upsample_rates)):
            c = channels[i]
            r = upsample_rates[i]
            if i < len(upsample_rates) - 1:
                c_next = channels[i+1]
            else:
                c_next = channels[-1]
            self.ups.append(nn.ConvTranspose1d(c, c_next, r, r))
            self.convs.append(DilatedCausalConvStack(c_next, c_next, kernel_size, num_layers))
            self.to_gains.append(nn.Conv1d(c_next, 1, 1))
        self.post = CausalConv1d(c_last, 1, kernel_size)

    def forward(self, x):
        x = self.input_layer(x)
        for up, conv, to_gain in zip(self.ups, self.convs, self.to_gains):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            skip = x
            x = torch.randn_like(x) * to_gain(x)
            x = conv(x) + skip
        x = self.post(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=80, output_channels=256, internal_channels=256, num_layers=8):
        super().__init__()
        self.stack = CausalConvNeXtStack(
                input_channels,
                internal_channels,
                internal_channels*3,
                7,
                output_channels,
                num_layers)

    def forward(self, x):
        return self.stack(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.harmonic_oscillator = HarmonicOscillator()
        self.noise_generator = NoiseGenerator()

    def forward(self, x, f0, t0=0):
        x = self.feature_extractor(x)
        harmonics = self.harmonic_oscillator(x, f0)
        noise = self.noise_generator(x)
        wave = harmonics + noise
        wave = wave.squeeze(1)
        return wave
