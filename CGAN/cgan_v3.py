#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cgan_posture.py — Train 9 posture-specialist CGANs.
Each model: condition is binary {no_exo=0, with_exo=1}.

Patches included:
- Two discriminators (Global/Local) with spectral norm, projection + ACGAN head
- Hinge + light R1 on real, TTUR (G lr > D lr)
- DiffAugment-1D, instance noise schedule
- Feature Matching (FM), PSD/Cov/Coherence (random channel pairs), per-channel μ/σ calibration
- Pre-warm G (structure/FM/amp) for a few epochs before adversarial starts
- AMP on D only; G kept in fp32
- EMA for the generator
- Balanced batches across conditions
"""

import os, json, csv, argparse, itertools
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import spectral_norm

# ---------------- config ----------------
NUM_CLASSES_COND = 2   # 0=no_exo, 1=with_exo
C, T = 14, 768
ALL_PAIRS = [(i, j) for i, j in itertools.combinations(range(C), 2)]

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # IO
    p.add_argument("--data-dir", type=str, default="./preprocessed")
    p.add_argument("--runs-root", type=str, default="./cgan_runs_posture")
    p.add_argument("--posture", type=str, default="all", help="'all' or an int 1..9")
    # Train
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--prewarm", type=int, default=5, help="epochs where G trains only structure/FM/amp")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--noise-dim", type=int, default=100)
    # TTUR + stabilizers
    p.add_argument("--lr-g", type=float, default=6e-4)
    p.add_argument("--lr-d", type=float, default=8e-5)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--d-steps", type=int, default=1)
    p.add_argument("--proj-scale", type=float, default=0.10)
    p.add_argument("--r1-gamma", type=float, default=0.5)
    p.add_argument("--r1-every", type=int, default=8)
    p.add_argument("--inst-noise-start", type=float, default=0.20)
    p.add_argument("--inst-noise-end", type=float, default=0.06)
    p.add_argument("--use-diffaugment", action="store_true", default=True)
    p.add_argument("--diffaugment-p", type=float, default=0.5)
    p.add_argument("--amp-d", action="store_true", default=True)
    # Loss weights
    p.add_argument("--acgan-weight", type=float, default=1.25)
    p.add_argument("--g-acgan-weight", type=float, default=1.5)
    p.add_argument("--fm-weight", type=float, default=50.0)
    p.add_argument("--psd-weight", type=float, default=0.3)
    p.add_argument("--coh-weight", type=float, default=0.8)
    p.add_argument("--cov-weight", type=float, default=0.3)
    p.add_argument("--amp-weight", type=float, default=0.5)   # per-channel mean/std calibration
    p.add_argument("--coh-pairs", type=int, default=24)
    # EMA / saves
    p.add_argument("--ema", action="store_true", default=True)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------- utils ---------------
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def autocast(enabled):
    try: return torch.amp.autocast(device_type='cuda', enabled=enabled)
    except TypeError: return torch.cuda.amp.autocast(enabled=enabled)

# --------------- data ---------------
def load_posture_both_conditions(data_dir: str, posture: int):
    files = {
        0: Path(data_dir)/f"posture{posture}_no_exo.npz",
        1: Path(data_dir)/f"posture{posture}_with_exo.npz"
    }
    Xs, ys = [], []
    meta = {}
    for cond, fp in files.items():
        if not fp.exists():
            raise SystemExit(f"Missing file: {fp}")
        z = np.load(fp, allow_pickle=True)
        X = z["X"].astype(np.float32).transpose(0,2,1)  # (N,C,T)
        y = np.full((X.shape[0],), cond, dtype=np.int64)
        Xs.append(X); ys.append(y)
        if not meta:
            meta = dict(ch_names=z["ch_names"], fs=float(z["fs"]),
                        scale_min=z["scale_min"].astype(np.float32),
                        scale_range=z["scale_range"].astype(np.float32))
    X = np.concatenate(Xs,0); y = np.concatenate(ys,0)
    perm = np.random.permutation(len(y))
    return X[perm], y[perm], meta

def build_index_by_cond(y: np.ndarray):
    return {c: np.where(y == c)[0] for c in (0,1)}

def sample_balanced_by_cond(X, y, idx_by_cond, bs, dev):
    half = bs // 2
    idx0 = np.random.choice(idx_by_cond[0], size=half, replace=len(idx_by_cond[0])<half)
    idx1 = np.random.choice(idx_by_cond[1], size=bs-half, replace=len(idx_by_cond[1])<bs-half)
    idx = np.concatenate([idx0, idx1]); np.random.shuffle(idx)
    real = torch.from_numpy(X[idx]).float().to(dev)
    labels = torch.from_numpy(y[idx]).long().to(dev)  # 0 or 1
    return real, labels

# --------------- DiffAugment-1D ---------------
def diffaugment_1d(x, p=0.5):
    if torch.rand(1, device=x.device).item() < p:
        shift = torch.randint(-8, 9, (1,), device=x.device).item()
        x = torch.roll(x, shifts=shift, dims=2)
    if torch.rand(1, device=x.device).item() < p:
        scale = (0.9 + 0.2*torch.rand(x.size(0),1,1, device=x.device))
        bias  = 0.02*torch.randn(x.size(0),1,1, device=x.device)
        x = torch.clamp(x*scale + bias, 0, 1)
    if torch.rand(1, device=x.device).item() < p:
        L = x.size(2); w = max(1, int(0.05*L))
        start = torch.randint(0, L-w, (x.size(0),), device=x.device)
        mask = torch.ones_like(x)
        for i,s in enumerate(start): mask[i,:,s:s+w] = 0
        x = x*mask
    return x

# --------------- models ---------------
class CBN1d(nn.Module):
    def __init__(self, nf, ncls):
        super().__init__()
        self.bn = nn.BatchNorm1d(nf, affine=False)
        self.emb = nn.Embedding(ncls, nf*2)
        nn.init.ones_(self.emb.weight[:, :nf]); nn.init.zeros_(self.emb.weight[:, nf:])
    def forward(self, x, y):
        h = self.bn(x)
        g,b = self.emb(y).chunk(2, dim=1)
        return g.unsqueeze(-1)*h + b.unsqueeze(-1)

class Ups(nn.Module):
    def __init__(self, ci, co, ncls):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(ci, co, 3, 1, 1)
        self.cbn = CBN1d(co, ncls)
    def forward(self, x, y): return F.relu(self.cbn(self.conv(self.up(x)), y), True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=2):
        super().__init__()
        self.nd = noise_dim; self.nc = num_classes
        self.ic, self.il = 512, 24
        self.proj = nn.Linear(noise_dim+num_classes, self.ic*self.il)
        self.up1=Ups(512,256,num_classes); self.up2=Ups(256,128,num_classes)
        self.up3=Ups(128,64,num_classes);  self.up4=Ups(64,32,num_classes); self.up5=Ups(32,16,num_classes)
        self.to_out = nn.Conv1d(16, C, 3, 1, 1); self.act = nn.Sigmoid()
    def forward(self, z, labels):
        oh = F.one_hot(labels, num_classes=self.nc).float()
        h = self.proj(torch.cat([z, oh], 1)).view(-1, self.ic, self.il)
        for blk in [self.up1,self.up2,self.up3,self.up4,self.up5]: h = blk(h, labels)
        return self.act(self.to_out(h))

class DiscBase(nn.Module):
    def __init__(self, ncls=2, proj_scale=0.10):
        super().__init__()
        self.proj_scale = proj_scale
        self.c1 = spectral_norm(nn.Conv1d(C,   32,4,2,1))
        self.c2 = spectral_norm(nn.Conv1d(32,  64,4,2,1))
        self.c3 = spectral_norm(nn.Conv1d(64, 128,4,2,1))
        self.c4 = spectral_norm(nn.Conv1d(128,256,4,2,1))
        self.c5 = spectral_norm(nn.Conv1d(256,512,4,2,1))
        self.fc = spectral_norm(nn.Linear(512,1))
        self.embed = nn.Embedding(ncls, 512)
        self.cls = spectral_norm(nn.Linear(512, ncls))
        self.std_weight = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=0.1)
    def features(self, x):
        h = F.leaky_relu(self.c1(x),0.2); h=F.leaky_relu(self.c2(h),0.2)
        h = F.leaky_relu(self.c3(h),0.2); h=F.leaky_relu(self.c4(h),0.2)
        h = F.leaky_relu(self.c5(h),0.2)
        return torch.mean(h, dim=2)
    def forward(self, x, labels):
        f = self.dropout(self.features(x))
        proj = torch.sum(f * self.embed(labels), dim=1, keepdim=True)
        std = torch.sqrt(torch.var(f, dim=0, unbiased=False) + 1e-8).mean().view(1,1).expand(f.size(0),1)
        score = self.fc(f) + self.proj_scale*proj + 0.1*std
        logits = self.cls(f)
        return score, logits, f

class GlobalD(DiscBase): pass
class LocalD (DiscBase): pass

# --------------- losses ---------------
def d_hinge(rs, fs): return torch.mean(F.relu(1.0 - rs) + F.relu(1.0 + fs))
def g_hinge(fs): return -torch.mean(fs)

def r1_penalty(D, x, labels):
    x.requires_grad_(True)
    s, _, _ = D(x, labels)
    g = torch.autograd.grad(outputs=s.sum(), inputs=x, create_graph=True)[0]
    return 0.5 * torch.mean(g.view(g.size(0), -1).pow(2).sum(1))

def _psd_loss_basic(real, fake):
    r32, f32 = real.float(), fake.float()
    Fr = torch.fft.rfft(r32, dim=2); Ff = torch.fft.rfft(f32, dim=2)
    P_r = (Fr.real**2 + Fr.imag**2).mean(0)
    P_f = (Ff.real**2 + Ff.imag**2).mean(0)
    return F.l1_loss(P_f, P_r)

def _coh_loss_random(real, fake, num_pairs=24):
    r32, f32 = real.float(), fake.float()
    idx = torch.randperm(len(ALL_PAIRS), device=real.device)[:num_pairs].tolist()
    pairs = [ALL_PAIRS[k] for k in idx]
    loss = 0.0
    for (i,j) in pairs:
        Ar = torch.fft.rfft(r32[:, i:i+1, :], dim=2); Br = torch.fft.rfft(r32[:, j:j+1, :], dim=2)
        Af = torch.fft.rfft(f32[:, i:i+1, :], dim=2); Bf = torch.fft.rfft(f32[:, j:j+1, :], dim=2)
        num_r = torch.sqrt((Ar*torch.conj(Br)).real**2 + (Ar*torch.conj(Br)).imag**2)
        den_r = torch.sqrt((Ar.real**2 + Ar.imag**2)*(Br.real**2 + Br.imag**2) + 1e-8)
        coh_r = (num_r/den_r).mean(0)
        num_f = torch.sqrt((Af*torch.conj(Bf)).real**2 + (Af*torch.conj(Bf)).imag**2)
        den_f = torch.sqrt((Af.real**2 + Af.imag**2)*(Bf.real**2 + Bf.imag**2) + 1e-8)
        coh_f = (num_f/den_f).mean(0)
        loss += F.l1_loss(coh_f, coh_r)
    return loss / max(1, len(pairs))

def _cov_loss_basic(real, fake):
    def cov(x):
        x = x - x.mean(dim=2, keepdim=True)
        return (x @ x.transpose(1,2) / (x.size(2)-1)).mean(0)
    return F.mse_loss(cov(fake), cov(real))

def amp_calib_loss(real, fake):
    mu_r, mu_f = real.mean((0,2)), fake.mean((0,2))
    sd_r, sd_f = real.std((0,2)),  fake.std((0,2))
    return F.l1_loss(mu_f, mu_r) + F.l1_loss(sd_f, sd_r)

@torch.no_grad()
def ema_update(src, tgt, decay=0.999):
    for ps, pt in zip(src.parameters(), tgt.parameters()):
        pt.data.mul_(decay).add_(ps.data, alpha=(1.0-decay))

def make_ema(G, dev):
    ema = Generator(noise_dim=G.nd, num_classes=NUM_CLASSES_COND).to(dev)
    ema.load_state_dict(G.state_dict()); [p.requires_grad_(False) for p in ema.parameters()]
    return ema

# --------------- training loop ---------------
def train_one_posture(args, posture:int):
    dev = device(); set_seed(args.seed)
    X, cond, meta = load_posture_both_conditions(args.data_dir, posture)
    idx_by_cond = build_index_by_cond(cond)

    G  = Generator(noise_dim=args.noise_dim, num_classes=NUM_CLASSES_COND).to(dev)
    Dg = GlobalD(ncls=NUM_CLASSES_COND, proj_scale=args.proj_scale).to(dev)
    Dl = LocalD (ncls=NUM_CLASSES_COND, proj_scale=args.proj_scale).to(dev)

    optG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1,args.beta2))
    optD = optim.Adam(list(Dg.parameters())+list(Dl.parameters()), lr=args.lr_d, betas=(args.beta1,args.beta2))
    scalerD = torch.cuda.amp.GradScaler(enabled=args.amp_d)
    ema_G = make_ema(G, dev) if args.ema else None
    ce = nn.CrossEntropyLoss()

    save_dir = Path(args.runs_root)/f"posture{posture}"; save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir/"hparams.json","w") as f: json.dump({**vars(args), "posture":posture}, f, indent=2)

    steps = max(1, X.shape[0]//args.batch_size)
    print(f"[posture {posture}] epochs={args.epochs}, steps/epoch≈{steps}")

    best_g = float("inf")
    metrics_csv = save_dir/"metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","g_loss","d_loss","Dg_R","Dg_F","Dl_R","Dl_F","ACg_R","ACg_F","ACl_R","ACl_F"])

    total_epochs = args.prewarm + args.epochs
    for ep in range(total_epochs):
        prewarm = (ep < args.prewarm)
        t = (ep) / max(1, total_epochs-1)
        sigma = (1-t)*args.inst_noise_start + t*args.inst_noise_end

        d_diag = {"dg_r":0.,"dg_f":0.,"dl_r":0.,"dl_f":0.,"acg_r":0.,"acg_f":0.,"acl_r":0.,"acl_f":0.}; d_count=0
        for _ in range(steps):
            # ---- D ----
            if not prewarm:
                real, labels = sample_balanced_by_cond(X, cond, idx_by_cond, args.batch_size, dev)
                z = torch.randn(args.batch_size, args.noise_dim, device=dev)
                with autocast(args.amp_d): fake = G(z, labels)

                real_in = real if sigma==0 else torch.clamp(real + sigma*torch.randn_like(real), 0, 1)
                fake_in = fake.detach() if sigma==0 else torch.clamp(fake.detach() + sigma*torch.randn_like(fake), 0, 1)
                if args.use_diffaugment:
                    real_in = diffaugment_1d(real_in, args.diffaugment_p)
                    fake_in = diffaugment_1d(fake_in, args.diffaugment_p)

                with autocast(args.amp_d):
                    rs_g, rlog_g, _ = Dg(real_in, labels); fs_g, flog_g, _ = Dg(fake_in, labels)
                    rs_l, rlog_l, _ = Dl(real_in[:,:, :256], labels); fs_l, flog_l, _ = Dl(fake_in[:,:, :256], labels)
                    dloss = d_hinge(rs_g, fs_g) + d_hinge(rs_l, fs_l) \
                            + args.acgan_weight*(ce(rlog_g,labels)+ce(rlog_l,labels))
                if (d_count % max(1,args.r1_every))==0 and args.r1_gamma>0:
                    with autocast(False):
                        dloss = dloss + args.r1_gamma*(r1_penalty(Dg, real_in, labels) + r1_penalty(Dl, real_in[:,:, :256], labels))

                optD.zero_grad(set_to_none=True); scalerD.scale(dloss).backward(); scalerD.step(optD); scalerD.update()

                with torch.no_grad():
                    d_diag["dg_r"] += (rs_g>0).float().mean().item(); d_diag["dg_f"] += (fs_g<0).float().mean().item()
                    d_diag["dl_r"] += (rs_l>0).float().mean().item(); d_diag["dl_f"] += (fs_l<0).float().mean().item()
                    d_diag["acg_r"] += (rlog_g.argmax(1)==labels).float().mean().item()
                    d_diag["acg_f"] += (flog_g.argmax(1)==labels).float().mean().item()
                    d_diag["acl_r"] += (rlog_l.argmax(1)==labels).float().mean().item()
                    d_diag["acl_f"] += (flog_l.argmax(1)==labels).float().mean().item()
                    d_count += 1

            # ---- G ---- (fp32)
            real_g, labels_g = sample_balanced_by_cond(X, cond, idx_by_cond, args.batch_size, dev)
            z2 = torch.randn(args.batch_size, args.noise_dim, device=dev)
            fake2 = G(z2, labels_g)
            fake2_in = fake2 if sigma==0 else torch.clamp(fake2 + sigma*torch.randn_like(fake2), 0, 1)
            if args.use_diffaugment: fake2_in = diffaugment_1d(fake2_in, args.diffaugment_p)

            gs_g, glog_g, ffeat = Dg(fake2_in, labels_g)
            gs_l, glog_l, _     = Dl(fake2_in[:, :, :256], labels_g)

            gloss = 0.0
            if not prewarm:
                gloss = gloss + g_hinge(gs_g) + g_hinge(gs_l) + args.g_acgan_weight*(ce(glog_g,labels_g)+ce(glog_l,labels_g))
            # structure + FM + amp always
            rfeat = Dg.features(real_g).detach().mean(0)
            gloss = gloss + args.fm_weight*F.mse_loss(ffeat.mean(0), rfeat)
            gloss = gloss + args.psd_weight*_psd_loss_basic(real_g, fake2)
            gloss = gloss + args.coh_weight*_coh_loss_random(real_g, fake2, num_pairs=args.coh_pairs)
            gloss = gloss + args.cov_weight*_cov_loss_basic(real_g, fake2)
            gloss = gloss + args.amp_weight*amp_calib_loss(real_g, fake2)

            optG.zero_grad(set_to_none=True); gloss.backward(); optG.step()
            if args.ema: ema_update(G, ema_G, decay=args.ema_decay)

        # epoch end
        for k in d_diag: d_diag[k] /= max(d_count,1)

        if (ep+1) % 10 == 0 or ep == 0:
            print(f"[posture {posture}] ep {ep+1}/{total_epochs} "
                  f"| G={gloss.item():.4f} D={float(dloss.item()) if not prewarm else 0:.4f} "
                  f"| Dg R/F={d_diag['dg_r']:.2f}/{d_diag['dg_f']:.2f} Dl R/F={d_diag['dl_r']:.2f}/{d_diag['dl_f']:.2f}")

        with open(metrics_csv,"a",newline="") as f:
            csv.writer(f).writerow([ep+1, float(gloss.item()), float(dloss.item()) if not prewarm else 0.0,
                                    d_diag['dg_r'], d_diag['dg_f'], d_diag['dl_r'], d_diag['dl_f'],
                                    d_diag['acg_r'], d_diag['acg_f'], d_diag['acl_r'], d_diag['acl_f']])

        # save best by G objective (after prewarm)
        if not prewarm and gloss.item() < best_g:
            best_g = gloss.item()
            torch.save((ema_G.state_dict() if args.ema else G.state_dict()), save_dir/f"CGAN_generator_posture{posture}_best.pth")
        if (ep+1) % args.save_every == 0:
            torch.save(G.state_dict(), save_dir/f"CGAN_generator_posture{posture}_epoch{ep+1}.pth")

    torch.save((ema_G.state_dict() if args.ema else G.state_dict()), save_dir/f"CGAN_generator_posture{posture}_last.pth")
    print(f"[posture {posture}] ✅ Done. Best G loss {best_g:.4f}")

# --------------- main ---------------
def main():
    args = get_args(); set_seed(args.seed)
    postures = range(1,10) if args.posture=="all" else [int(args.posture)]
    for p in postures:
        train_one_posture(args, p)

if __name__ == "__main__":
    main()
