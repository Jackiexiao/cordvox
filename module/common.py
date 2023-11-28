import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


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
    def __init__(self, channels=256, hidden_channels=512, kernel_size=7, scale=1, dilation=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation)
        self.pad = nn.ReflectionPad1d([kernel_size*dilation-dilation, 0])
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
                 num_layers=6,
                 dilation=1):
        super().__init__()
        self.input_layer = nn.Conv1d(input_channels, channels, 1)
        self.output_layer = nn.Conv1d(channels, output_channels, 1)
        self.mid_layers = nn.Sequential(
                *[CausalConvNeXt1d(channels, hidden_channels, kernel_size, dilation=dilation) for i in range(num_layers)])

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


class DilatedCausalConvStack(nn.Module):
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


class AdaptiveChannelNorm(nn.Module):
    def __init__(self, channels, condition_emb, eps=1e-4):
        super().__init__()
        self.shift = nn.Conv1d(condition_emb, channels, 1, 1, 0)
        self.scale = nn.Conv1d(condition_emb, channels, 1, 1, 0)
        self.eps = eps

    def forward(self, x, p):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale(p) + self.shift(p)
        return x


class AdaptiveConvNeXt1d(nn.Module):
    def __init__(self, channels=512, hidden_channels=1536, condition_emb=512, kernel_size=7, scale=1, dilation=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, dilation), groups=channels)
        self.norm = AdaptiveChannelNorm(channels, condition_emb)
        self.pw_conv1 = nn.Conv1d(channels, hidden_channels, 1)
        self.pw_conv2 = nn.Conv1d(hidden_channels, channels, 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1) * scale)

    def forward(self, x, p):
        res = x
        x = self.dw_conv(x)
        x = self.norm(x, p)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = x * self.scale
        return x + res
