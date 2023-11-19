import argparse
import sys
import json
import torchaudio
import os
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm

from generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputs', default="./inputs/")
parser.add_argument('-o', '--outputs', default="./outputs/")
parser.add_argument('-genp', '--generator-path', default="generator.pt")
parser.add_argument('-d', '--device', default='cpu')
parser.add_argument('-c', '--chunk', default=65536, type=int)
parser.add_argument('-norm', '--normalize', default=False, type=bool)
parser.add_argument('--noise', default=1, type=float)
parser.add_argument('--harmonics', default=1, type=float)
parser.add_argument('-g', '--gain', default=0)

args = parser.parse_args()

device = torch.device(args.device)

G = Generator().to(device)
G.load_state_dict(torch.load(args.generator_path, map_location=device))

if not os.path.exists(args.outputs):
    os.mkdir(args.outputs)

mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_fft=3840,
        hop_length=960,
        n_mels=80
        ).to(device)

def log_mel(x, eps=1e-5):
    return torch.log(mel(x) + eps)[:, :, 1:]

paths = glob.glob(os.path.join(args.inputs, "*"))
for i, path in enumerate(paths):
    wf, sr = torchaudio.load(path)
    wf = wf.to('cpu')
    wf = torchaudio.functional.resample(wf, sr, 48000)
    wf = wf / wf.abs().max()
    wf = wf.mean(dim=0, keepdim=True)
    total_length = wf.shape[1]
    
    wf = torch.cat([wf, torch.zeros(1, (args.chunk * 3))], dim=1)

    wf = wf.unsqueeze(1).unsqueeze(1)
    wf = F.pad(wf, (args.chunk, args.chunk, 0, 0))
    chunks = F.unfold(wf, (1, args.chunk*3), stride=args.chunk)
    chunks = chunks.transpose(1, 2).split(1, dim=1)

    result = []
    with torch.no_grad():
        print(f"converting {path}")
        for chunk in tqdm(chunks):
            chunk = chunk.squeeze(1)

            if chunk.shape[1] < args.chunk:
                chunk = torch.cat([chunk, torch.zeros(1, args.chunk - chunk.shape[1])], dim=1)
            chunk = chunk.to(device)

            chunk = G.forward_without_t(log_mel(chunk), args.noise, args.harmonics)
            
            chunk = chunk[:, args.chunk:-args.chunk]

            result.append(chunk.to('cpu'))
        wf = torch.cat(result, dim=1)[:, :total_length]
        wf = torchaudio.functional.resample(wf, 48000, sr)
        wf = torchaudio.functional.gain(wf, args.gain)
    wf = wf.cpu().detach()
    if args.normalize:
        wf = wf / wf.abs().max()
    torchaudio.save(os.path.join("./outputs/", f"{os.path.splitext(os.path.basename(path))[0]}.wav"), src=wf, sample_rate=sr)
