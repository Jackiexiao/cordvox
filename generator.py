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
    def __init__(self, input_channels, output_channels, kernel_size=5):
        super().__init__()
        self.pad = nn.ReflectionPad1d([kernel_size-1, 0])
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)

    def forward(self, x):
        return self.conv(self.pad(x))


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
        self.to_mag = nn.Conv1d(input_channels, num_harmonics*2, 1)
        self.to_pitch = nn.Conv1d(input_channels, 1, 1)
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.base_frequency = base_frequency
        self.num_harmonics = num_harmonics
        self.min_frequency = min_frequency
    
    # x = extracted features, t = time(seconds)
    def forward(self, x, t):
        N = x.shape[0] # batch size
        Nh = self.num_harmonics # number of harmonics
        Lf = x.shape[2] # frame length
        Lw = Lf * self.segment_size # wave length
        
        # Estimate f0 and magnitudes of each harmonics 
        pitch = self.to_pitch(x)
        mag = self.to_mag(x)
        f0 = self.base_frequency * 2 ** pitch
        f0 = f0.clamp_min(self.min_frequency)
        mag_cos, mag_sin = torch.chunk(mag, 2, dim=1)
        
        # frequency multiplyer
        mul = (torch.arange(Nh, device=x.device) + 1).unsqueeze(0).unsqueeze(2).expand(N, Nh, Lw)

        # Interpolate F0
        f0 = F.interpolate(f0, Lw, mode='linear')

        # Expand f0: [N, 1, Lw] -> [N, Nh, Lw]
        f0 = t.expand(N, Nh, Lw)

        # Expand t: [N, 1, Lw] -> [N, Nh, Lw]
        t = t.expand(N, Nh, Lw)

        # Interpolate mag_sin, mag_cos
        mag_sin = F.interpolate(mag_sin, Lw, mode='linear')
        mag_cos = F.interpolate(mag_cos, Lw, mode='linear')

        # Expand mag_sin, mag_cos : [N, 1, Lw] -> [N, Nh, Lw]
        mag_sin = mag_sin.expand(N, Nh, Lw)
        mag_cos = mag_cos.expand(N, Nh, Lw)

        sin_part = torch.sin(t * 2 * math.pi * f0 * mul) * mag_sin
        cos_part = torch.cos(t * 2 * math.pi * f0 * mul) * mag_cos
        harmonics = sin_part + cos_part

        # Sum all harmonics
        wave = harmonics.mean(dim=1, keepdim=True)
        return wave


# Tiny upsample-based noise generator
class NoiseGenerator(nn.Module):
    def __init__(
            self,
            input_channels=256,
            upsample_rates=[10, 8, 4, 3],
            channels=[32, 16, 8, 4],
            kernel_size=15,
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
            self.convs.append(CausalConv1d(c_next, c_next, kernel_size))
        self.post = CausalConv1d(c_last, 1, kernel_size)

    def forward(self, x):
        x = self.input_layer(x)
        for up, conv in zip(self.ups, self.convs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
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
    def forward(self, x, w):
        res = w
        s = self.feature2scale(x)
        w = self.wave_in(w) * s
        w = F.leaky_relu(w, LRELU_SLOPE)
        w = self.wave_out(w)
        return w


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
    
    # x: input features, t: time(seconds) [N, 1, Lw]
    # Output: [N, Lw]
    def forward(self, x, t, noise_scale=1, harmonics_scale=1):
        x = self.feature_extractor(x)
        h = self.harmonic_generator(x, t)
        n = self.noise_generator(x)
        w = h * harmonics_scale + n * noise_scale
        w = self.post_filter(x, w)
        return w.squeeze(1)
    
    # forward without argument 't'. for training.
    def forward_without_t(self, x, noise_scale=1, harmonics_scale=1):
        N = x.shape[0] # batch size
        Lf = x.shape[2] # length(frame based)
        Lw = Lf * self.segment_size

        t = torch.linspace(0, Lw, Lw) / self.sample_rate
        t = t.expand(N, 1, Lw)
        t = t.to(x.device)
        return self.forward(x, t, noise_scale, harmonics_scale)

