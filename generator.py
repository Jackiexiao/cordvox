import torch
import torch.nn as nn
import torch.nn.functional as F
import math

LRELU_SLOPE = 0.1

class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class CausalConvNeXt1d(nn.Module):
    def __init__(self, channels=256, hidden_channels=512, kernel_size=7, scale=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, groups=channels)
        self.pad = nn.ReflectionPad1d([kernel_size-1, 0])
        self.norm = ChannelNorm(channels)
        self.pw_conv1 = nn.Conv1d(channels, hidden_channels, 1)
        self.pw_conv2 = nn.Conv1d(hidden_channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * scale)

    def forward(self, x):
        res = x
        x = self.pad(x)
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.relu(x, LRELU_SLOPE)
        x = self.pw_conv2(x)
        x = x * self.scale
        return x + res


class CausalConvNeXtStack(nn.Module):
    def __init__(self,
                 input_channels=80,
                 channels=256,
                 hidden_channels=512,
                 kernel_size=7,
                 output_channels=256,
                 num_layers=6):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.output_layer = nn.Conv1d(channels, output_channels, 1)
        self.mid_layers = nn.Sequential(*[CausalConvNeXt1d(channels, hidden_channels, kernel_size) for _ in range(num_layers)])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.mid_layers(x)
        x = self.output_layer(x)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=5, dilation=1):
        super().__init__()
        self.pad = nn.ReflectionPad1d([kernel_size*dilation-dilation, 0])
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(self.pad(x))


class DilatedCausalConvSatck(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=5, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([])
        self.convs.append(CausalConv1d(input_channels, output_channels, kernel_size, 1))
        for d in range(num_layers-1):
            self.convs.append(CausalConv1d(output_channels, output_channels, kernel_size, d+2))
    
    def forward(self, x):
        for c in self.convs:
            F.leaky_relu(x, LRELU_SLOPE)
            x = c(x)
        return x


# Sinewave based harmonic generator
class HarmonicGenerator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            sample_rate=48000,
            segment_size=960,
            num_harmonics=8,
            base_frequency=220,
            min_frequency=20,
            ):
        super().__init__()
        self.to_mag = nn.Conv1d(input_channels, num_harmonics, 1)
        self.to_octave = nn.Conv1d(input_channels, 1, 1)
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.base_frequency = base_frequency
        self.num_harmonics = num_harmonics
        self.min_frequency = min_frequency
    
    # x = extracted features, phi = phase status
    def forward(self, x, phi):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length
        
        # Estimate f0 and magnitudes of each harmonics 
        octave = self.to_octave(x)
        mag = torch.exp(self.to_mag(x).clamp_max(6.0))
        f0 = self.base_frequency * 2 ** octave # to Hz

        # frequency multiplyer
        mul = (torch.arange(Nh, device=x.device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lf)

        # Calculate formants
        formants = f0 * mul

        # Interpolate folmants
        formants = F.interpolate(f0, Lw, mode='linear')

        # Interpolate mag
        mag = F.interpolate(mag, Lw, mode='linear')

        # Expand mags : [N, 1, Lw] -> [N, Nh, Lw]
        mag = mag.expand(N, Nh, Lw)
        
        # Generate harmonics (algorith based on DDSP)
        t = torch.cumsum(formants / self.sample_rate, dim=2) + phi
        harmonics = torch.sin(2 * math.pi * t) * mag

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)
        return wave


# Tiny upsample-based noise generator
class NoiseGenerator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            upsample_rates=[10, 8, 4, 3],
            channels=[64, 32, 16, 8],
            kernel_size=5,
            ):
        super().__init__()
        c0 = channels[0]
        c_last = channels[-1]
        self.input_layer = CausalConv1d(input_channels, c0, kernel_size)
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
            self.convs.append(DilatedCausalConvSatck(c_next, c_next, kernel_size, 3))
        self.post = CausalConv1d(c_last, 1, kernel_size)

    def forward(self, x):
        x = self.input_layer(x)
        for up, conv in zip(self.ups, self.convs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = conv(x)
        x = self.post(x)
        return x


class ConvFilter(nn.Module):
    def __init__(
            self,
            input_channels=256,
            mid_channels=8,
            segment_size=960,
            kernel_size=40,
            ):
        super().__init__()
        self.feature2scale = nn.ConvTranspose1d(input_channels, mid_channels, segment_size, segment_size)
        self.wave_in = CausalConv1d(1, mid_channels, kernel_size)
        self.wave_out = CausalConv1d(mid_channels, 1, kernel_size)
    
    # x: extracted features [N, 1, Lf], w: generated waves [N, 1, Lw]
    # Output: [N, 1, Lw]
    def forward(self, x, w, alpha=1):
        res = w
        s = self.feature2scale(x)
        w = self.wave_in(w) * s
        w = F.leaky_relu(w, LRELU_SLOPE)
        w = self.wave_out(w)
        return w * (1 - alpha) + res * alpha


class Generator(nn.Module):
    def __init__(
            self,
            input_channels=80,
            segment_size=960,
            sample_rate=48000
            ):
        super().__init__()
        self.segment_size = segment_size
        self.sample_rate = sample_rate

        self.feature_extractor = CausalConvNeXtStack(input_channels)
        self.harmonic_generator = HarmonicGenerator(segment_size=segment_size)
        self.noise_generator = NoiseGenerator()
        self.post_filter = ConvFilter()
    
    # x: input features, phi: [N, Nh, 1]
    # Output: [N, Lw]
    def forward(self, x, phi, noise_scale=1, harmonics_scale=1):
        x = self.feature_extractor(x)
        h = self.harmonic_generator(x, phi)
        n = self.noise_generator(x)
        w = h * harmonics_scale + n * noise_scale
        w = self.post_filter(x, w)
        w = w.squeeze(1)
        return w, phi
    
    # forward without argument 'phi'. for training.
    def forward_without_phi(self, x, noise_scale=1, harmonics_scale=1):
        N = x.shape[0]
        Nh = self.harmonic_generator.num_harmonics
        phi = torch.zeros(N, Nh, 1, device=x.device)
        wf, phi = self.forward(x, phi, noise_scale, harmonics_scale)
        return wf

