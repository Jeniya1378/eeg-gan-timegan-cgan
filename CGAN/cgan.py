#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cgan.py — CGAN for EEG (14 × 768) with posture-conditional structure losses.

- Two discriminators (Global full seq + Local random crop), spectral norm
- Projection discriminator + ACGAN posture head on both Ds
- Hinge loss + R1 (on real) + TTUR (G stronger than D by default)
- DiffAugment-1D, instance noise decay, feature matching, EMA generator
- **Posture-conditional** PSD / Coherence / Channel-Cov losses for G
- FFT work in float32; no complex .abs() (NVRTC-safe)
- **Generator step in fp32** (no AMP) to avoid Half/Float mismatch
- **Discriminator step uses AMP** for speed
- Balanced per-posture sampling each batch
- Per-epoch diagnostics (D real/fake acc, ACGAN accs, losses) -> metrics.csv
- Trains both conditions by default; saves EMA generator *_best.pth

Assumes preprocessed .npz files like: posture{1..9}_{with_exo|no_exo}.npz
X shape (N, T, C) scaled to [0,1]; channels=14, T=768.
"""
import os, csv, json, time, argparse
from pathlib import Path
from glob import glob
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

# ---------------- CLI ----------------
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=False)

    # Train
    p.add_argument("--data-dir", type=str, default="./preprocessed")
    p.add_argument("--save-root", type=str, default="./cgan_runs")
    p.add_argument("--condition", type=str, default="both", choices=["both", "with_exo", "no_exo"])
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--noise-dim", type=int, default=100)

    # Tuned defaults to help G
    p.add_argument("--lr-g", type=float, default=3e-4)
    p.add_argument("--lr-d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--d-steps", type=int, default=1)
    p.add_argument("--proj-scale", type=float, default=0.25)

    # Loss family
    p.add_argument("--loss", type=str, default="hinge", choices=["hinge", "wgan-gp", "bce"])
    p.add_argument("--gp-weight", type=float, default=10.0)

    # Label supervision
    p.add_argument("--acgan-weight", type=float, default=1.5)
    p.add_argument("--g-acgan-weight", type=float, default=2.0)

    # Stabilizers
    p.add_argument("--r1-gamma", type=float, default=0.5)
    p.add_argument("--r1-every", type=int, default=8)
    p.add_argument("--inst-noise-start", type=float, default=0.20)
    p.add_argument("--inst-noise-end", type=float, default=0.02)
    p.add_argument("--use-diffaugment", action="store_true", default=True)
    p.add_argument("--diffaugment-p", type=float, default=0.25)

    # Structure losses (posture-conditional)
    p.add_argument("--psd-weight", type=float, default=0.5)
    p.add_argument("--coh-weight", type=float, default=0.25)
    p.add_argument("--cov-weight", type=float, default=0.25)
    p.add_argument("--local-crop", type=int, default=256)

    # >>> Missing before; added now <<<
    p.add_argument("--fm-weight", type=float, default=15.0, help="Feature matching loss weight")

    # EMA / Schedules / AMP-D
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--lr-decay", type=float, default=1.0)
    p.add_argument("--lr-decay-step", type=int, default=200)
    p.add_argument("--amp-d", action="store_true", default=True, help="Use AMP for discriminator step")

    # IO / misc
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    # Generate
    g = sub.add_parser("generate", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g.add_argument("--data-dir", type=str, default="./preprocessed")
    g.add_argument("--save-root", type=str, default="./cgan_runs")
    g.add_argument("--condition", type=str, required=True, choices=["with_exo", "no_exo"])
    g.add_argument("--model-path", type=str, default="")
    g.add_argument("--noise-dim", type=int, default=100)
    g.add_argument("--num-per-posture", type=int, default=100)
    g.add_argument("--inverse-scale", action="store_true")
    g.add_argument("--seed", type=int, default=123)
    return p.parse_args()

# --------------- constants / utils ---------------
NUM_CLASSES = 9
C, T = 14, 768

def set_seed(s: int):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_device():
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_condition_dataset(data_dir: str, condition: str):
    files = sorted(glob(str(Path(data_dir) / f"posture*_{condition}.npz")))
    if not files:
        raise SystemExit(f"No files found like posture*_{condition}.npz in {data_dir}")
    Xs, ys = [], []
    meta: Dict[int, dict] = {}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        X = z["X"].astype(np.float32)      # (N,T,C) in [0,1]
        posture = int(z["posture"])
        X = X.transpose(0, 2, 1)           # -> (N,C,T)
        y = np.full((X.shape[0],), posture, dtype=np.int64)
        Xs.append(X); ys.append(y)
        meta[posture] = {
            "file": fp,
            "scale_min": z["scale_min"].astype(np.float32),
            "scale_range": z["scale_range"].astype(np.float32),
            "ch_names": z["ch_names"],
            "fs": float(z["fs"]),
        }
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    perm = np.random.permutation(X_all.shape[0])
    return X_all[perm], y_all[perm], meta

def build_index_by_label(y: np.ndarray):
    return {k: np.where(y == k)[0] for k in range(1, NUM_CLASSES+1)}

def sample_balanced_batch(X: np.ndarray, y: np.ndarray, idx_by_label: dict, batch_size: int, device):
    labels_np = np.random.randint(1, NUM_CLASSES+1, size=batch_size)
    reals = [X[np.random.choice(idx_by_label[lab])] for lab in labels_np]
    real = torch.from_numpy(np.stack(reals, axis=0)).float().to(device)
    labels = torch.from_numpy(labels_np - 1).long().to(device)  # 0..8
    return real, labels

def random_crop(x: torch.Tensor, L: int):
    L = min(L, x.size(2))
    if x.size(2) == L: return x
    start = torch.randint(0, x.size(2) - L + 1, (1,), device=x.device).item()
    return x[:, :, start:start+L]

def autocast(enabled: bool):
    try:
        return torch.amp.autocast('cuda', enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)

# --------------- DiffAugment-1D ---------------
def diffaugment_1d(x, p=0.25):
    if torch.rand(1, device=x.device).item() < p:  # time shift
        shift = torch.randint(-8, 9, (1,), device=x.device).item()
        x = torch.roll(x, shifts=shift, dims=2)
    if torch.rand(1, device=x.device).item() < p:  # amplitude jitter
        scale = (0.9 + 0.2*torch.rand(x.size(0),1,1, device=x.device))
        bias  = 0.02*torch.randn(x.size(0),1,1, device=x.device)
        x = torch.clamp(x*scale + bias, 0.0, 1.0)
    if torch.rand(1, device=x.device).item() < p:  # time cutout
        L = x.size(2); w = max(1, int(0.05*L))
        start = torch.randint(0, L-w, (x.size(0),), device=x.device)
        mask = torch.ones_like(x)
        for i,s in enumerate(start):
            mask[i,:,s:s+w] = 0
        x = x*mask
    return x

# --------------- Models ---------------
class CBN1d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features*2)
        nn.init.ones_(self.embed.weight[:, :num_features])
        nn.init.zeros_(self.embed.weight[:, num_features:])
    def forward(self, x, y):
        h = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1); beta = beta.unsqueeze(-1)
        return gamma * h + beta

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.cbn = CBN1d(out_ch, num_classes)
    def forward(self, x, labels):
        x = self.up(x)
        x = self.conv(x)
        return F.relu(self.cbn(x, labels), inplace=True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=9):
        super().__init__()
        self.noise_dim = noise_dim; self.num_classes = num_classes
        self.init_ch = 512; self.init_len = 24
        self.proj = nn.Linear(noise_dim + num_classes, self.init_ch * self.init_len)
        self.up1 = UpsampleBlock(512,256,num_classes)
        self.up2 = UpsampleBlock(256,128,num_classes)
        self.up3 = UpsampleBlock(128, 64,num_classes)
        self.up4 = UpsampleBlock(64,  32,num_classes)
        self.up5 = UpsampleBlock(32,  16,num_classes)
        self.to_out = nn.Conv1d(16, C, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()
    def forward(self, z, labels):
        oh = F.one_hot(labels, num_classes=self.num_classes).float()
        h = self.proj(torch.cat([z, oh], dim=1)).view(-1, self.init_ch, self.init_len)
        h = self.up1(h, labels); h = self.up2(h, labels); h = self.up3(h, labels)
        h = self.up4(h, labels); h = self.up5(h, labels)
        return self.out_act(self.to_out(h))  # (B,14,768)

class DiscBase(nn.Module):
    def __init__(self, num_classes=9, proj_scale=0.25):
        super().__init__()
        self.proj_scale = proj_scale
        self.c1 = spectral_norm(nn.Conv1d(C,   32,4,2,1))
        self.c2 = spectral_norm(nn.Conv1d(32,  64,4,2,1))
        self.c3 = spectral_norm(nn.Conv1d(64, 128,4,2,1))
        self.c4 = spectral_norm(nn.Conv1d(128,256,4,2,1))
        self.c5 = spectral_norm(nn.Conv1d(256,512,4,2,1))
        self.fc = spectral_norm(nn.Linear(512, 1))
        self.embed = nn.Embedding(num_classes, 512)
        self.cls = spectral_norm(nn.Linear(512, num_classes))
        self.std_weight = nn.Parameter(torch.zeros(1))
    def extract(self, x):
        h = F.leaky_relu(self.c1(x), 0.2)
        h = F.leaky_relu(self.c2(h), 0.2)
        h = F.leaky_relu(self.c3(h), 0.2)
        h = F.leaky_relu(self.c4(h), 0.2)
        h = F.leaky_relu(self.c5(h), 0.2)
        return torch.mean(h, dim=2)  # (B,512)
    def forward(self, x, labels):
        f = self.extract(x)
        std = torch.sqrt(torch.var(f, dim=0, unbiased=False) + 1e-8)
        mb = std.mean().view(1).expand(f.size(0), 1)
        proj = torch.sum(f * self.embed(labels), dim=1, keepdim=True)
        score = self.fc(f) + self.proj_scale * proj + self.std_weight * mb
        logits = self.cls(f)
        return score, logits, f

class GlobalD(DiscBase): pass
class LocalD (DiscBase): pass

# --------------- losses / EMA ---------------
def d_hinge(real_scores, fake_scores):
    return torch.mean(F.relu(1.0 - real_scores) + F.relu(1.0 + fake_scores))
def g_hinge(fake_scores):
    return -torch.mean(fake_scores)

def r1_penalty(D, x, labels):
    x.requires_grad_(True)
    scores, _, _ = D(x, labels)
    grad = torch.autograd.grad(outputs=scores.sum(), inputs=x, create_graph=True)[0]
    return 0.5 * torch.mean(grad.view(grad.size(0), -1).pow(2).sum(1))

@torch.no_grad()
def ema_update(model_src, model_tgt, decay=0.999):
    for p_s, p_t in zip(model_src.parameters(), model_tgt.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=(1.0 - decay))
def make_ema(model, device):
    ema = Generator(noise_dim=model.noise_dim, num_classes=model.num_classes).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters(): p.requires_grad_(False)
    return ema

# --------------- structure helpers (fp32; no complex abs) ---------------
def _mag(z):  # magnitude without torch.abs on complex
    return torch.sqrt(z.real*z.real + z.imag*z.imag)

def _psd_loss_basic(real, fake):
    r32 = real.float(); f32 = fake.float()
    Fr = torch.fft.rfft(r32, dim=2)          # (B,C,F)
    Ff = torch.fft.rfft(f32, dim=2)
    P_r = (Fr.real**2 + Fr.imag**2).mean(0)  # (C,F)
    P_f = (Ff.real**2 + Ff.imag**2).mean(0)
    return F.l1_loss(P_f, P_r)

def _coh_loss_basic(real, fake, pairs):
    r32 = real.float(); f32 = fake.float()
    def coh(a, b):
        A = torch.fft.rfft(a, dim=2)
        B = torch.fft.rfft(b, dim=2)
        num = _mag(A * torch.conj(B))
        den = torch.sqrt((A.real**2 + A.imag**2) * (B.real**2 + B.imag**2) + 1e-8)
        return (num / den).mean(0)
    loss = 0.0
    for (i,j) in pairs:
        cr = coh(r32[:, i:i+1, :], r32[:, j:j+1, :])
        cf = coh(f32[:, i:i+1, :], f32[:, j:j+1, :])
        loss = loss + F.l1_loss(cf, cr)
    return loss / len(pairs)

def _cov_loss_basic(real, fake):
    def covmat(x):
        x = x - x.mean(dim=2, keepdim=True)
        cov = torch.matmul(x, x.transpose(1,2)) / (x.size(2) - 1)
        return cov.mean(0)
    return F.mse_loss(covmat(fake), covmat(real))

def posture_conditional_losses(real, fake, labels, psd_w, coh_w, cov_w):
    """
    Compute PSD/Coherence/Cov **per posture present in the batch** and average.
    """
    if (psd_w + coh_w + cov_w) == 0:
        return real.new_tensor(0.0)

    pairs = [(0,13), (6,7), (9,10), (1,12)]  # AF3-AF4, O1-O2, T8-FC6, F7-F8
    losses = []
    for lab in torch.unique(labels):
        mask = (labels == lab)
        r = real[mask]; f = fake[mask]
        if r.size(0) == 0:
            continue
        l = 0.0
        if psd_w > 0: l = l + psd_w * _psd_loss_basic(r, f)
        if coh_w > 0: l = l + coh_w * _coh_loss_basic(r, f, pairs)
        if cov_w > 0: l = l + cov_w * _cov_loss_basic(r, f)
        losses.append(l.float())
    return torch.stack(losses).mean() if losses else real.new_tensor(0.0)

# --------------- training ---------------
def train_one_condition(args, condition: str):
    device = get_device(); set_seed(args.seed)

    X_all, y_all, meta = load_condition_dataset(args.data_dir, condition)
    idx_by_label = build_index_by_label(y_all)

    G  = Generator(noise_dim=args.noise_dim).to(device)
    Dg = GlobalD(proj_scale=args.proj_scale).to(device)
    Dl = LocalD (proj_scale=args.proj_scale).to(device)

    optG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optD = optim.Adam(list(Dg.parameters()) + list(Dl.parameters()), lr=args.lr_d, betas=(args.beta1, args.beta2))

    schG = optim.lr_scheduler.StepLR(optG, step_size=args.lr_decay_step, gamma=args.lr_decay) if args.lr_decay < 1.0 else None
    schD = optim.lr_scheduler.StepLR(optD, step_size=args.lr_decay_step, gamma=args.lr_decay) if args.lr_decay < 1.0 else None

    scalerD = torch.cuda.amp.GradScaler(enabled=args.amp_d)
    ema_G = make_ema(G, device) if args.ema else None

    save_dir = Path(args.save_root) / condition
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "hparams.json", "w") as f:
        hp = vars(args).copy(); hp["condition"] = condition; json.dump(hp, f, indent=2)

    # metrics CSV
    metrics_csv = save_dir / "metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch","g_loss","d_loss",
                "d_g_real_acc","d_g_fake_acc","d_l_real_acc","d_l_fake_acc",
                "acgan_real_global","acgan_fake_global","acgan_real_local","acgan_fake_local"
            ])

    start_epoch = 0
    best_g = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        G.load_state_dict(ck["G_state"]); Dg.load_state_dict(ck["Dg_state"]); Dl.load_state_dict(ck["Dl_state"])
        optG.load_state_dict(ck["optim_G_state"]); optD.load_state_dict(ck["optim_D_state"])
        start_epoch = ck.get("epoch", 0); best_g = ck.get("g_loss", best_g)
        if ema_G and "G_ema_state" in ck: ema_G.load_state_dict(ck["G_ema_state"]); ema_G.to(device)
        print(f"[{condition}] Resumed from {args.resume} @ epoch {start_epoch}")

    steps_per_epoch = max(1, X_all.shape[0] // args.batch_size)
    print(f"[{condition}] Training {args.epochs} epochs, steps/epoch ≈ {steps_per_epoch}")
    ce = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        G.train(); Dg.train(); Dl.train()
        t = epoch / max(1, args.epochs - 1)
        sigma = (1 - t) * args.inst_noise_start + t * args.inst_noise_end

        # running diagnostics
        d_g_real_acc = d_g_fake_acc = d_l_real_acc = d_l_fake_acc = 0.0
        acgan_real_g = acgan_fake_g = acgan_real_l = acgan_fake_l = 0.0
        diag_count = 0

        for step in range(steps_per_epoch):
            # ---------- D updates (AMP on) ----------
            for _ in range(args.d_steps):
                real, labels = sample_balanced_batch(X_all, y_all, idx_by_label, args.batch_size, device)
                z = torch.randn(args.batch_size, args.noise_dim, device=device)
                with autocast(args.amp_d):
                    fake = G(z, labels)

                real_in = real if sigma == 0 else torch.clamp(real + sigma * torch.randn_like(real), 0, 1)
                fake_in = fake.detach() if sigma == 0 else torch.clamp(fake.detach() + sigma * torch.randn_like(fake), 0, 1)
                if args.use_dffaugment if False else args.use_diffaugment:  # safety; supports older args
                    real_in = diffaugment_1d(real_in, args.diffaugment_p)
                    fake_in = diffaugment_1d(fake_in, args.diffaugment_p)

                with autocast(args.amp_d):
                    rs_g, rlog_g, _ = Dg(real_in, labels)
                    fs_g, flog_g, _ = Dg(fake_in, labels)
                    dloss_g = d_hinge(rs_g, fs_g) + args.acgan_weight * ce(rlog_g, labels)

                real_loc = random_crop(real_in, args.local_crop)
                fake_loc = random_crop(fake_in, args.local_crop)
                with autocast(args.amp_d):
                    rs_l, rlog_l, _ = Dl(real_loc, labels)
                    fs_l, flog_l, _ = Dl(fake_loc, labels)
                    dloss_l = d_hinge(rs_l, fs_l) + args.acgan_weight * ce(rlog_l, labels)

                dloss = dloss_g + dloss_l

                if (step % max(1, args.r1_every)) == 0 and args.loss == "hinge" and args.r1_gamma > 0:
                    with autocast(False):
                        r1g = r1_penalty(Dg, real_in, labels)
                        r1l = r1_penalty(Dl, real_loc, labels)
                        dloss = dloss + args.r1_gamma * (r1g + r1l)

                optD.zero_grad(set_to_none=True)
                scalerD.scale(dloss).backward()
                scalerD.step(optD)
                scalerD.update()

                # diagnostics on this step
                with torch.no_grad():
                    d_g_real_acc += (rs_g > 0).float().mean().item()
                    d_g_fake_acc += (fs_g < 0).float().mean().item()
                    d_l_real_acc += (rs_l > 0).float().mean().item()
                    d_l_fake_acc += (fs_l < 0).float().mean().item()
                    acgan_real_g += (rlog_g.argmax(1) == labels).float().mean().item()
                    acgan_fake_g += (flog_g.argmax(1) == labels).float().mean().item()
                    acgan_real_l += (rlog_l.argmax(1) == labels).float().mean().item()
                    acgan_fake_l += (flog_l.argmax(1) == labels).float().mean().item()
                    diag_count += 1

            # ---------- G update (FULL FP32; no AMP) ----------
            real_g, labels_g = sample_balanced_batch(X_all, y_all, idx_by_label, args.batch_size, device)
            z2 = torch.randn(args.batch_size, args.noise_dim, device=device)

            fake2 = G(z2, labels_g)                         # fp32
            fake2_in = fake2 if sigma == 0 else torch.clamp(fake2 + sigma * torch.randn_like(fake2), 0, 1)
            if args.use_diffaugment:
                fake2_in = diffaugment_1d(fake2_in, args.diffaugment_p)

            gs_g, glog_g, ffeat_g = Dg(fake2_in, labels_g)  # fp32
            gs_l, glog_l, _       = Dl(random_crop(fake2_in, args.local_crop), labels_g)

            gloss = g_hinge(gs_g) + g_hinge(gs_l)
            gloss = gloss + args.g_acgan_weight * (ce(glog_g, labels_g) + ce(glog_l, labels_g))

            rfeat = Dg.extract(real_g).detach().mean(0)
            gloss = gloss + args.fm_weight * F.mse_loss(ffeat_g.mean(0), rfeat)

            # posture-conditional structure losses
            gloss = gloss + posture_conditional_losses(
                real_g, fake2, labels_g,
                psd_w=args.psd_weight, coh_w=args.coh_weight, cov_w=args.cov_weight
            )

            optG.zero_grad(set_to_none=True)
            gloss.backward()
            optG.step()

            if ema_G is not None:
                ema_update(G, ema_G, decay=args.ema_decay)

        if schG: schG.step(); schD.step()

        # per-epoch logging
        d_g_real_acc /= max(diag_count, 1); d_g_fake_acc /= max(diag_count, 1)
        d_l_real_acc /= max(diag_count, 1); d_l_fake_acc /= max(diag_count, 1)
        acgan_real_g /= max(diag_count, 1); acgan_fake_g /= max(diag_count, 1)
        acgan_real_l /= max(diag_count, 1); acgan_fake_l /= max(diag_count, 1)

        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            print(f"[{condition}] Ep {epoch+1}/{args.epochs} | "
                  f"D={dloss.item():.4f} G={gloss.item():.4f} | "
                  f"Dg(R/F)={d_g_real_acc:.2f}/{d_g_fake_acc:.2f} Dl(R/F)={d_l_real_acc:.2f}/{d_l_fake_acc:.2f} | "
                  f"ACGAN G (R/F)={acgan_real_g:.2f}/{acgan_fake_g:.2f} L (R/F)={acgan_real_l:.2f}/{acgan_fake_l:.2f}")

        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch+1, float(gloss.item()), float(dloss.item()),
                d_g_real_acc, d_g_fake_acc, d_l_real_acc, d_l_fake_acc,
                acgan_real_g, acgan_fake_g, acgan_real_l, acgan_fake_l
            ])

        # save checkpoints + best (by G objective)
        if (epoch + 1) % args.save_every == 0:
            ck = {"epoch": epoch+1,
                  "G_state": G.state_dict(), "Dg_state": Dg.state_dict(), "Dl_state": Dl.state_dict(),
                  "optim_G_state": optG.state_dict(), "optim_D_state": optD.state_dict(),
                  "g_loss": float(gloss.item()), "d_loss": float(dloss.item())}
            if ema_G: ck["G_ema_state"] = ema_G.state_dict()
            torch.save(ck, save_dir / f"checkpoint_epoch{epoch+1}.pt")
            torch.save(G.state_dict(), save_dir / f"CGAN_generator_{condition}_epoch{epoch+1}.pth")

        if gloss.item() < best_g:
            best_g = gloss.item()
            torch.save((ema_G.state_dict() if ema_G else G.state_dict()), save_dir / f"CGAN_generator_{condition}_best.pth")
            torch.save(Dg.state_dict(), save_dir / f"CGAN_globalD_{condition}_best.pth")
            torch.save(Dl.state_dict(), save_dir / f"CGAN_localD_{condition}_best.pth")

    torch.save((ema_G.state_dict() if ema_G else G.state_dict()), save_dir / f"CGAN_generator_{condition}_last.pth")
    print(f"[{condition}] ✅ Done. Best G loss: {best_g:.4f}")

# --------------- generation ---------------
@torch.no_grad()
def generate_for_condition(args):
    device = get_device()
    _, _, meta = load_condition_dataset(args.data_dir, args.condition)

    G = Generator(noise_dim=args.noise_dim).to(device)
    gpath = Path(args.model_path) if args.model_path else (Path(args.save_root)/args.condition/f"CGAN_generator_{args.condition}_best.pth")
    state = torch.load(gpath, map_location=device)
    G.load_state_dict(state); G.eval()
    print(f"[{args.condition}] Loaded generator: {gpath}")

    out_dir = Path(args.save_root)/args.condition/f"generated_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for posture in range(1, NUM_CLASSES+1):
        n = args.num_per_posture
        z = torch.randn(n, args.noise_dim, device=device)
        labels = torch.full((n,), posture-1, dtype=torch.long, device=device)
        synth = G(z, labels).cpu().numpy()
        minv = meta[posture]["scale_min"][None,:,None]
        rngv = meta[posture]["scale_range"][None,:,None]
        X_out = synth * rngv + minv if args.inverse_scale else synth
        np.savez_compressed(out_dir/f"synth_posture{posture}_{args.condition}.npz",
            X=X_out.transpose(0,2,1).astype(np.float32),
            posture=np.int32(posture),
            condition=str(args.condition),
            ch_names=np.array(meta[posture]["ch_names"], dtype=object),
            fs=np.float32(meta[posture]["fs"]),
            note="CGAN generation")
        print(f"[{args.condition}] Saved {n} -> {out_dir}/synth_posture{posture}_{args.condition}.npz")
    print(f"[{args.condition}] ✅ Generation complete: {out_dir}")

# ---------------- main ----------------
def main():
    args = get_args()
    set_seed(args.seed)
    if args.cmd == "generate":
        generate_for_condition(args); return
    if args.condition in ("both","with_exo"): train_one_condition(args, "with_exo")
    if args.condition in ("both","no_exo"):   train_one_condition(args, "no_exo")

if __name__ == "__main__":
    main()
