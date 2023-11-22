import torch
import torchaudio

from module.harmonic_oscillator import HarmonicOscillator


Lf = 100
Nh = 32

harmonics = torch.randn(Nh).abs()
oscillator = HarmonicOscillator(num_harmonics=Nh)

f0 = torch.ones(1, 1, Lf) * 123

harmonics = torch.log10(torch.Tensor(harmonics) + 1e-6) 
print(harmonics)

amps = harmonics.unsqueeze(0).unsqueeze(2)
amps = amps.expand(1, Nh, Lf)

wave = oscillator(f0, amps).squeeze(1)
wave = wave / (wave.abs().max() + 1e-6)

torchaudio.save("output.wav", wave, sample_rate=48000)
