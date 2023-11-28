import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np

from module.generator import Generator
from module.preprocess import LogMelSpectrogram
from module.dataset import compute_f0

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-genp', '--generator-path', default="generator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-c', '--chunk', default=240000, type=int)
parser.add_argument('-norm', '--normalize', default=False, type=bool)
parser.add_argument('--harmonics', default=1, type=float)
parser.add_argument('-g', '--gain', default=0)

args = parser.parse_args()

device = torch.device(args.device)

G = Generator().to(device)
G.load_state_dict(torch.load(args.generator_path, map_location=device))

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)


log_mel = LogMelSpectrogram().to(device)

mel_hq = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_fft=3840,
        hop_length=240,
        n_mels=256
        ).to(device)

def log_mel_hq(x, eps=1e-5):
    return torch.log(mel_hq(x) + eps)[:, :, 1:]

# Plot spectrogram
def plot_spec(x, save_path="./spectrogram.png", log=False):
    plt.figure()
    x = x[0]
    x = x.flip(dims=(0,))
    plt.imshow(x)
    plt.savefig(save_path, dpi=200)
    plt.close()


paths = []
for fmt in ["wav", "mp3", "ogg"]:
    paths += glob.glob(os.path.join(args.inputs, f"**/*.{fmt}"), recursive=True)
n_files = len(paths)
scores = []
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = wf.to('cpu')
    wf = torchaudio.functional.resample(wf, sr, 48000)
    wf = wf / wf.abs().max()
    wf = wf.mean(dim=0, keepdim=True)
    wf_in = wf
    total_length = wf.shape[1]
    
    wf = torch.cat([wf, torch.zeros(1, (args.chunk * 3))], dim=1)

    wf = wf.unsqueeze(1).unsqueeze(1)
    wf = F.pad(wf, (args.chunk, args.chunk, 0, 0))

    chunks = F.unfold(wf, (1, args.chunk*3), stride=args.chunk)
    chunks = chunks.transpose(1, 2).split(1, dim=1)

    result = []
    with torch.no_grad():
        tqdm.write(f"{i} / {n_files} : Inferencing {path}")
        bar = tqdm(total=len(chunks))
        t0 = 0
        for chunk in chunks:
            chunk = chunk.squeeze(1)

            if chunk.shape[1] < args.chunk:
                chunk = torch.cat([chunk, torch.zeros(1, args.chunk - chunk.shape[1])], dim=1)
            chunk = chunk.to(device)

            chunk_in = chunk[:, args.chunk:-args.chunk]

            f0 = compute_f0(chunk)
            chunk = G(log_mel(chunk), f0, t0, args.harmonics)

            chunk = chunk[:, args.chunk:-args.chunk]

            score = (log_mel(chunk_in) - log_mel(chunk)).abs().mean().item()
            scores.append(score)

            result.append(chunk.to('cpu'))
            bar.set_description(f"Mel loss: {score}")
            bar.update(1)
            
            t0 += chunk.shape[1] / 48000

        wf = torch.cat(result, dim=1)[:, :total_length]

        wf_out = wf

        wf = torchaudio.functional.resample(wf, 48000, sr)
        wf = torchaudio.functional.gain(wf, args.gain)
    wf = wf.cpu().detach()
    if args.normalize:
        wf = wf / wf.abs().max()
    file_name = f"{i}_{os.path.splitext(os.path.basename(path))[0]}"
    wf_in = wf_in.to(device)
    wf_out = wf_out.to(device)
    plot_spec(log_mel_hq(wf_in).cpu(), os.path.join(args.outputs, f"{file_name}_input.png"))
    plot_spec(log_mel_hq(wf_out).cpu(), os.path.join(args.outputs, f"{file_name}_output.png"))
    plot_spec((log_mel_hq(wf_in).cpu() - log_mel_hq(wf_out).cpu()).abs(), os.path.join("./outputs/", f"{file_name}_diff.png"))
    torchaudio.save(os.path.join(args.outputs, f"{file_name}.wav"), src=wf, sample_rate=sr)

mean_scores = sum(scores) / len(scores)
print("Complete!")

time.sleep(0.1)

print("")
print("-" * 80)
print("")
print(f"Total Valid. Mel loss: {mean_scores}")
