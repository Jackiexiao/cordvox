import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

from tqdm import tqdm

from dataset import WaveFileDirectory
from generator import Generator
from discriminator import Discriminator

parser = argparse.ArgumentParser(description="train Vocoder")

parser.add_argument('dataset')

parser.add_argument('-genp', '--generator-path', default="generator.pt")
parser.add_argument('-disp', '--discriminator-path', default="discriminator.pt")
parser.add_argument('-d', '--device', default='cuda')
parser.add_argument('-e', '--epoch', default=1000, type=int)
parser.add_argument('-b', '--batch-size', default=4, type=int)
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float)
parser.add_argument('-len', '--length', default=24000, type=int)
parser.add_argument('-m', '--max-data', default=-1, type=int)
parser.add_argument('-fp16', default=False, type=bool)
parser.add_argument('--feature-matching', default=2, type=float)
parser.add_argument('--mel', default=45, type=float)
parser.add_argument('--content', default=1, type=float)

args = parser.parse_args()


def inference_mode(model):
    for param in model.parameters():
        param.requires_grad = False


def load_or_init_models(device=torch.device('cpu')):
    dis = Discriminator().to(device)
    gen = Generator().to(device)
    if os.path.exists(args.generator_path):
        gen.load_state_dict(torch.load(args.generator_path, map_location=device))
    if os.path.exists(args.discriminator_path):
        dis.load_state_dict(torch.load(args.discriminator_path, map_location=device))
    return gen, dis


def save_models(gen, dis):
    print("Saving Models...")
    torch.save(gen.state_dict(), args.generator_path)
    torch.save(dis.state_dict(), args.discriminator_path)
    print("complete!")


def cut_center(x):
    length = x.shape[2]
    center = length // 2
    size = length // 4
    return x[:, :, center-size:center+size]


def cut_center_wav(x):
    length = x.shape[1]
    center = length // 2
    size = length // 4
    return x[:, center-size:center+size]


device = torch.device(args.device)
G, D = load_or_init_models(device)

ds = WaveFileDirectory(
        [args.dataset],
        length=args.length,
        max_files=args.max_data
        )

dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

OptG = optim.AdamW(G.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))
OptD = optim.AdamW(D.parameters(), lr=args.learning_rate, betas=(0.8, 0.99))

SchedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(OptG, 5000)
SchedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(OptD, 5000)

step_count = 0

mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        n_fft=1920,
        hop_length=480,
        n_mels=80
        ).to(device)

def log_mel(x, eps=1e-5):
    return torch.log(mel(x) + eps)[:, :, 1:]

for epoch in range(args.epoch):
    tqdm.write(f"Epoch #{epoch}")
    bar = tqdm(total=len(ds))
    for batch, wave in enumerate(dl):
        wave = wave.to(device) * (torch.rand(wave.shape[0], 1, device=device) * 2)
        spec = log_mel(wave)
        
        # Train G.
        OptG.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            wave_fake = G.forward_without_t(spec)
            
            loss_adv = 0
            logits = D.logits(cut_center_wav(wave_fake))
            for logit in logits:
                loss_adv += (logit ** 2).mean()

            loss_mel = (log_mel(wave_fake) - spec).abs().mean()
            loss_feat = D.feat_loss(cut_center_wav(wave_fake), cut_center_wav(wave))

            loss_g = loss_mel * args.mel + loss_feat * args.feature_matching  + loss_adv

        scaler.scale(loss_g).backward()
        scaler.step(OptG)

        # Train D.
        OptD.zero_grad()
        wave_fake = wave_fake.detach()
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits_fake = D.logits(cut_center_wav(wave_fake))
            logits_real = D.logits(cut_center_wav(wave))
            loss_d = 0
            for logit in logits_real:
                loss_d += (logit ** 2).mean()
            for logit in logits_fake:
                loss_d += ((logit - 1) ** 2).mean()
        scaler.scale(loss_d).backward()
        scaler.step(OptD)

        scaler.update()
        SchedulerD.step()
        SchedulerG.step()

        step_count += 1
        
        tqdm.write(f"Step {step_count}, D: {loss_d.item():.4f}, Adv.: {loss_adv.item():.4f}, Mel.: {loss_mel.item():.4f}, Feat.: {loss_feat.item():.4f}")

        N = wave.shape[0]
        bar.update(N)

        if batch % 150 == 0:
            save_models(G, D)

print("Training Complete!")
save_models(G, D)

