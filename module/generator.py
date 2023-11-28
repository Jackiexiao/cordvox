import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from module.common import CausalConvNeXtStack, CausalConv1d, DilatedCausalConvStack

LRELU_SLOPE = 0.1


class HarmonicOscillator(nn.Module):
    def __init__(self,
                 input_channels=80,
                 num_layers=4,
                 channels=256,
                 num_harmonics=16,
                 segment_size=960,
                 sample_rate=48000,
                 f0_max=4000,
                 ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        
        self.to_amps = CausalConvNeXtStack(input_channels, channels, channels*2, 7, num_harmonics)
    
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


class Filter(nn.Module):
    def __init__(self,
                 channels=512,
                 n_fft=3840,
                 hop_length=960,
                 kernel_size=7,
                 num_layers=8):
        super().__init__()
        self.stack = CausalConvNeXtStack(n_fft//2+1, channels, channels*2, kernel_size, n_fft+2, num_layers)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.pad = nn.ReflectionPad1d([1, 0])

    def spectrogram(self, x):
        return torch.stft(x, self.n_fft, self.hop_length, return_complex=True).abs()[:, :, 1:]

    def mag_phase(self, x):
        x = self.pad(x)
        x = self.stack(x)
        return x.chunk(2, dim=1)

    def forward(self, x):
        dtype = x.dtype
        x = self.spectrogram(x)
        mag, phase = self.mag_phase(x)
        mag = mag.to(torch.float)
        phase = phase.to(torch.float)
        mag = torch.clamp_max(mag, 6.0)
        mag = torch.exp(mag)
        phase = torch.cos(phase) + 1j * torch.sin(phase)
        s = mag * phase
        wf = torch.istft(s, self.n_fft, hop_length=self.hop_length)
        return wf


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.harmonic_oscillator = HarmonicOscillator()
        self.filter = Filter()

    def forward(self, x, f0, t0=0, harmonics_scale=1, noise_scale=1):
        x = self.harmonic_oscillator(x, f0, t0)
        x = x.squeeze(1)
        x = self.filter(x)
        return x
