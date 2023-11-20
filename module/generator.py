import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from module.common import CausalConvNeXtStack, ChannelNorm, DCC, CausalConv1d

LRELU_SLOPE = 0.1

class FeatureExtractor(nn.Module):
    def __init__(self,
                 input_channels=80,
                 internal_channels=256,
                 hidden_channels=512,
                 output_channels=256,
                 num_layers=6,
                 ):
        super().__init__()
        self.stack = CausalConvNeXtStack(input_channels, internal_channels, hidden_channels, 7, output_channels, num_layers)

    def forward(self, x):
        return self.stack(x)


class HarmonicOscillator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            sample_rate=48000,
            segment_size=960,
            num_harmonics=48,
            f0_min = 0,
            f0_max = 4096,
            ):
        super().__init__()
        self.to_mag = nn.Conv1d(input_channels, num_harmonics, 1)
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.num_harmonics = num_harmonics
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    # x = extracted features, phi = phase status
    def wave_formants(self, x, f0):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length
        
        mag = torch.exp(self.to_mag(x).clamp_max(4.0))

        # frequency multiplyer
        mul = (torch.arange(Nh, device=x.device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)

        # Calculate formants
        formants = f0 * mul

        # Interpolate folmants
        formants = F.interpolate(formants, Lw, mode='linear')

        # Interpolate mag
        mag = F.interpolate(mag, Lw, mode='linear')

        # Expand mags : [N, 1, Lw] -> [N, Nh, Lw]
        mag = mag.expand(N, Nh, Lw)
        
        # Generate harmonics (algorith based on DDSP)
        t = torch.cumsum(formants / self.sample_rate, dim=2)
        harmonics = torch.sin(2 * math.pi * t) * mag

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)
        return wave, formants

    def forward(self, x, f0):
        x, _ = self.wave_formants(x, f0)
        return x


class NoiseGenerator(nn.Module):
    def __init__(self,
                 input_channels=256,
                 n_fft=3840,
                 hop_length=960,
                 ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.norm = ChannelNorm(input_channels)
        self.output_layer = nn.Conv1d(input_channels, n_fft+2, 1)
        self.pad = nn.ReflectionPad1d([1, 0])

    def mag_phase(self, x):
        x = self.pad(x)
        x = self.norm(x)
        x = self.output_layer(x)
        mag, phase = torch.chunk(x, 2, dim=1)
        return mag, phase

    def forward(self, x):
        dtype = x.dtype
        mag, phase = self.mag_phase(x)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        return torch.istft(s, self.n_fft, hop_length=self.hop_length).unsqueeze(1)


class Reverb(nn.Module):
    def __init__(self, kernel_size=10, mid_channels=16, num_layers=4):
        super().__init__()
        self.wave_in = CausalConv1d(1, mid_channels, kernel_size)
        self.dcc = DCC(mid_channels, mid_channels, kernel_size, num_layers)
        self.wave_out = CausalConv1d(mid_channels, 1, kernel_size)

    def forward(self, x):
        x = self.wave_in(x)
        F.leaky_relu(x, LRELU_SLOPE)
        x = self.dcc(x)
        x = self.wave_out(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.harmonic_oscillator = HarmonicOscillator()
        self.noise_generator = NoiseGenerator()
        self.reverb = Reverb()

    def forward(self, x, f0):
        x, _ = self.wave_formants(x, f0)
        return x

    def wave_formants(self, x, f0):
        x = self.feature_extractor(x)
        h, formants = self.harmonic_oscillator.wave_formants(x, f0)
        n = self.noise_generator(x)
        x = h #+ n
        x = self.reverb(x)

        mu = x.mean(dim=2, keepdim=True)
        x = x - mu

        x = x.squeeze(1)
        return x, formants
