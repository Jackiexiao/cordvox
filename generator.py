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
            self.convs.append(CausalConv1d(output_channels, output_channels, kernel_size, 2**(d+1)))
    
    def forward(self, x):
        for c in self.convs:
            F.leaky_relu(x, LRELU_SLOPE)
            x = c(x)
        return x


class F0Estimator(nn.Module):
    def __init__(self, channels=256, f0_min=20, f0_max=2000):
        super().__init__()
        self.conv = nn.Conv1d(channels, 2, 1)
        self.s1 = nn.Parameter(torch.ones(1, 1, 1) * 16)
        self.s2 = nn.Parameter(torch.ones(1, 1, 1) * 440)
        self.f0_max = f0_max
        self.f0_min = f0_min

    def forward(self, x):
        x = self.conv(x)
        a, b = x.chunk(2, dim=1)
        x = (torch.atan(a / (b + 1e-4)) / (math.pi * 2))
        x = x * self.s1
        x = 2 ** x # Convert log to linear scale
        x = x * self.s2
        x = x.clamp(self.f0_min, self.f0_max)
        return x


# Sinewave based harmonic Oscillator
class HarmonicOscillator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            sample_rate=48000,
            segment_size=960,
            num_harmonics=32,
            f0_min = 20.0,
            f0_max = 2000.0,
            ):
        super().__init__()
        self.to_mag = nn.Conv1d(input_channels, num_harmonics, 1)
        self.f0_estimator = F0Estimator(input_channels, f0_min, f0_max)
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.num_harmonics = num_harmonics
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    # x = extracted features, phi = phase status
    def wave_formants(self, x):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length
        
        # Estimate f0 and magnitudes of each harmonics 
        f0 = self.f0_estimator(x)
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

    def forward(self, x):
        x, _ = self.wave_formants(x)
        return x


# Tiny upsample-based noise generator
class NoiseGenerator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            upsample_rates=[10, 8, 4, 3],
            channels=[32, 16, 8, 4],
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
            self.convs.append(DilatedCausalConvSatck(c_next, c_next, kernel_size, num_layers))
            self.to_gains.append(nn.Conv1d(c_next, 1, 1))
        self.post = CausalConv1d(c_last, 1, kernel_size)

    def forward(self, x):
        x = self.input_layer(x)
        for up, conv, to_gain in zip(self.ups, self.convs, self.to_gains):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            skip = x
            x = x + torch.randn_like(x) * to_gain(x)
            x = conv(x) + x
        x = self.post(x)
        return x


class ConvFilter(nn.Module):
    def __init__(
            self,
            input_channels=256,
            mid_channels=16,
            segment_size=960,
            kernel_size=5,
            num_layers=6,
            ):
        super().__init__()
        self.wave_in = CausalConv1d(1, mid_channels, kernel_size)
        self.mid_layers = DilatedCausalConvSatck(mid_channels, mid_channels, kernel_size, num_layers)
        self.wave_out = CausalConv1d(mid_channels, 1, kernel_size)
    
    # x: extracted features [N, 1, Lf], w: generated waves [N, 1, Lw]
    # Output: [N, 1, Lw]
    def forward(self, x, alpha=1):
        res = x
        x = self.wave_in(x)
        x = self.mid_layers(x)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.wave_out(x)
        return x * (1 - alpha) + res * alpha


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
        self.harmonic_oscillator = HarmonicOscillator(segment_size=segment_size)
        self.noise_generator = NoiseGenerator()
        self.post_filter = ConvFilter()
    
    # x: input features
    # Output: [N, Lw]
    def wave_formants(self, x, noise_scale=1, harmonics_scale=1):
        x = self.feature_extractor(x)
        h, formants = self.harmonic_oscillator.wave_formants(x)
        n = self.noise_generator(x)
        x = h * harmonics_scale + n * noise_scale
        x = self.post_filter(x)
        mu = x.mean(dim=2, keepdim=True)
        x = x - mu
        x = x.squeeze(1)
        return x, formants

    def forward(self, x, noise_scale=1, harmonics_scale=1):
        x, fs = self.wave_formants(x, noise_scale, harmonics_scale)
        return x

    def extract_feature(self, x):
        return self.feature_extractor(x)

    def feat_loss(self, fake, real):
        with torch.no_grad():
            real_feat = self.extract_feature(real)
        fake_feat = self.extract_feature(fake)
        return (real_feat - fake_feat).abs().mean()

