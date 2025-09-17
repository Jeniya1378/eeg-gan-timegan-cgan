#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone visualization for posture-specialist CGAN runs.
- Generates synthetic sequences per posture & condition using your trained CGAN.
- Builds PCA and t-SNE embeddings on PSD features (same evaluation style).
- Uses the SAME color scheme & legend style:
    • 36 unique colors (posture × condition × {Real,Synthetic})
    • Markers: Real='o', Synthetic='x'
    • Legend formatted as 18 rows × 2 columns with paired labels
    • Posture-2 biased to darker palette indices

Outputs:
  <OUT_DIR>/pca_36.png
  <OUT_DIR>/tsne_36.png

Run:
  python visualize_cgan_36clusters.py \
      --data_dir ./preprocessed \
      --runs_root ./cgan_runs_posture \
      --out ./cgan_viz_out \
      --samples_per_cond match \
      --noise_dim 100 \
      --tsne_perplexity 30.0 --tsne_iter 1000
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.nn.functional as F

# ----------------- Defaults -----------------
C, T = 14, 768
NUM_COND = 2  # 0=no_exo, 1=with_exo

# ----------------- IO helpers -----------------
def load_real_posture(data_dir: str, posture: int):
    """Load real data for a posture; returns dict{cond: X(N,C,T)} and meta."""
    paths = {
        0: Path(data_dir)/f"posture{posture}_no_exo.npz",
        1: Path(data_dir)/f"posture{posture}_with_exo.npz",
    }
    real = {}
    meta_any = None
    for cond, fp in paths.items():
        if not fp.exists():
            raise SystemExit(f"Missing real data file: {fp}")
        z = np.load(fp, allow_pickle=True)
        X = z["X"].astype(np.float32).transpose(0,2,1)  # (N,C,T)
        real[cond] = X
        if meta_any is None:
            meta_any = dict(
                ch_names=list(z["ch_names"]),
                fs=float(z["fs"]),
                scale_min=z["scale_min"].astype(np.float32),
                scale_range=z["scale_range"].astype(np.float32),
            )
    return real, meta_any

# ----------------- CGAN Generator (matches training naming) -----------------
class CBN1d(nn.Module):
    def __init__(self, nf, ncls):
        super().__init__()
        self.bn = nn.BatchNorm1d(nf, affine=False)
        self.emb = nn.Embedding(ncls, nf*2)
        nn.init.ones_(self.emb.weight[:, :nf]); nn.init.zeros_(self.emb.weight[:, nf:])
    def forward(self, x, y):
        h = self.bn(x); g,b = self.emb(y).chunk(2, dim=1)
        return g.unsqueeze(-1)*h + b.unsqueeze(-1)

class UpsampleBlock(nn.Module):
    def __init__(self, ci, co, ncls):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(ci, co, 3, 1, 1)
        self.cbn = CBN1d(co, ncls)
    def forward(self, x, y):
        return F.relu(self.cbn(self.conv(self.up(x)), y), inplace=True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=2):
        super().__init__()
        self.nd, self.nc = noise_dim, num_classes
        self.init_ch, self.init_len = 512, 24
        self.proj = nn.Linear(noise_dim+num_classes, self.init_ch*self.init_len)
        self.up1 = UpsampleBlock(512,256,num_classes)
        self.up2 = UpsampleBlock(256,128,num_classes)
        self.up3 = UpsampleBlock(128, 64,num_classes)
        self.up4 = UpsampleBlock(64,  32,num_classes)
        self.up5 = UpsampleBlock(32,  16,num_classes)
        self.to_out = nn.Conv1d(16, C, 3, 1, 1)
        self.out_act = nn.Sigmoid()
    def forward(self, z, labels):
        oh = F.one_hot(labels, num_classes=self.nc).float()
        h = self.proj(torch.cat([z,oh],1)).view(-1, self.init_ch, self.init_len)
        h = self.up1(h,labels); h=self.up2(h,labels); h=self.up3(h,labels); h=self.up4(h,labels); h=self.up5(h,labels)
        return self.out_act(self.to_out(h))

def safe_load_generator(G: nn.Module, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    try:
        G.load_state_dict(sd, strict=True)
    except RuntimeError:
        new_sd={}
        for k,v in sd.items():
            nk = k.replace("u1.","up1.").replace("u2.","up2.").replace("u3.","up3.")\
                  .replace("u4.","up4.").replace("u5.","up5.").replace("out.","to_out.")
            new_sd[nk]=v
        G.load_state_dict(new_sd, strict=False)

@torch.no_grad()
def synthesize_for_posture(runs_root, posture, noise_dim, n_per_cond, device):
    gpath = Path(runs_root)/f"posture{posture}"/f"CGAN_generator_posture{posture}_best.pth"
    if not gpath.exists():
        gpath = Path(runs_root)/f"posture{posture}"/f"CGAN_generator_posture{posture}_last.pth"
    if not gpath.exists():
        raise SystemExit(f"Missing generator checkpoint for posture {posture}: {gpath.parent}")
    G = Generator(noise_dim=noise_dim, num_classes=NUM_COND).to(device).eval()
    safe_load_generator(G, gpath, device)
    fakes = {}
    for cond in (0,1):
        z = torch.randn(n_per_cond, noise_dim, device=device)
        y = torch.full((n_per_cond,), cond, dtype=torch.long, device=device)
        x = G(z,y).cpu().numpy()   # (N,C,T) in [0,1]
        fakes[cond] = x
    return fakes

# ----------------- Features -----------------
def psd_features(X, n_bins=64, eps=1e-6):
    N,C,T = X.shape
    F = np.fft.rfft(X.astype(np.float32), axis=2)
    P = (F.real**2 + F.imag**2) / (T/2.0 + 1e-8)
    P = np.log(P + eps)
    Fbins = P.shape[2]
    if n_bins < Fbins:
        pool = Fbins // n_bins
        P = P[:, :, :pool*n_bins].reshape(N, C, n_bins, pool).mean(-1)
    else:
        pad = n_bins - Fbins
        P = np.pad(P, ((0,0),(0,0),(0,max(0,pad))), mode="edge")[:, :, :n_bins]
    feats = P.reshape(N, C*n_bins)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ----------------- Color scheme (36 clusters) -----------------
def make_palette(n, cmap_name="hsv"):
    cmap = plt.cm.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]

def cluster_id(posture, cond, domain, color_scheme="36"):
    """Map (posture, condition, domain) → color index.
    posture: 1..9, cond: 0=no_exo,1=with_exo, domain: 1=real,0=gen
    """
    p = posture - 1
    c = 1 if cond==1 else 0  # keep 0=no_exo, 1=with_exo for clarity
    d = 1 if domain == 1 else 0
    if color_scheme == "36":
        idx = p*4 + (1-c)*2 + (0 if d==1 else 1)  # 'no_exo' first if desired
        # Bias posture-2 to low (darker) indices (0..3 bucket)
        if posture == 2:
            idx = (1-c)*2 + (0 if d==1 else 1)
        return idx
    else:
        idx = p*2 + (1-c)
        return idx

def add_paired_legend(ax, colors, fontsize=5.6, color_scheme="36"):
    """18 rows × 2 cols: for each posture×cond, show (Real, Synthetic)."""
    handles, labels = [], []
    for p in range(1,10):
        for cond in [0,1]:  # 0=no_exo, 1=with_exo
            # Real
            gid_r = cluster_id(p, cond, 1, color_scheme=color_scheme)
            handles.append(Line2D([], [], linestyle='None', marker='o', markersize=5, color=colors[gid_r]))
            labels.append(f"P-{p} {'No exo' if cond==0 else 'With exo'} Real")
            # Synthetic
            gid_g = cluster_id(p, cond, 0, color_scheme=color_scheme)
            handles.append(Line2D([], [], linestyle='None', marker='x', markersize=5, color=colors[gid_g]))
            labels.append(f"P-{p} {'No exo' if cond==0 else 'With exo'} Syn")
    lg = ax.legend(handles, labels, ncol=2, fontsize=fontsize, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.02, 1),
                   borderaxespad=0.0, columnspacing=1.0, handlelength=1.2,
                   handletextpad=0.5, markerscale=1.0)
    return lg

# ----------------- Embedding & plotting -----------------
def plot_embeddings(Fr, Fg, P, Cc, D, out_path, title, tsne=False, tsne_perp=30.0, tsne_iter=1000, seed=123):
    X = np.vstack([Fr, Fg])
    # Colors
    n_colors = 36
    colors = make_palette(n_colors, cmap_name="hsv")

    # Dimensionality reduction
    if tsne:
        # PCA→t-SNE
        K = min(50, X.shape[1]-1)
        X50 = PCA(n_components=K, random_state=seed).fit_transform(np.nan_to_num(X,0,0,0))
        try:
            tsne = TSNE(n_components=2, perplexity=tsne_perp, n_iter=tsne_iter,
                        learning_rate="auto", init="pca", random_state=seed, verbose=0)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=tsne_perp, learning_rate=200.0,
                        init="pca", random_state=seed)
        Z = tsne.fit_transform(X50)
    else:
        Z = PCA(n_components=2, random_state=seed).fit_transform(np.nan_to_num(X,0,0,0))

    # Build arrays for all points (order: [real; synthetic])
    P_all = np.array(P, dtype=int)
    C_all = np.array(Cc, dtype=int)
    D_all = np.array(D, dtype=int)  # 1 real, 0 gen

    # Scatter by (p,cond,domain)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for p in range(1,10):
        for cond in [0,1]:
            for dom in [1,0]:
                m = (P_all==p) & (C_all==cond) & (D_all==dom)
                if not np.any(m): continue
                gid = cluster_id(p, cond, dom, color_scheme="36")
                marker = 'o' if dom==1 else 'x'
                ax.scatter(Z[m,0], Z[m,1],
                           s=9, alpha=0.8, marker=marker,
                           c=[colors[gid]], label=None)

    ax.set_title(title)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.subplots_adjust(right=0.78)
    add_paired_legend(ax, colors, fontsize=5.6, color_scheme="36")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", type=str, default="./preprocessed")
    ap.add_argument("--runs_root", type=str, default="./cgan_runs_posture")
    ap.add_argument("--out", type=str, default="./cgan_viz_out")
    ap.add_argument("--postures", type=int, nargs="*", default=list(range(1,10)))
    ap.add_argument("--samples_per_cond", type=str, default="match", help='"match" or an integer')
    ap.add_argument("--noise_dim", type=int, default=100)
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Fr_list, Fg_list = [], []
    P_all, C_all, D_all = [], [], []  # posture, cond(0=no_exo,1=with_exo), domain (1=real,0=gen)

    for p in args.postures:
        # Load real
        real_dict, _ = load_real_posture(args.data_dir, p)
        # Decide number of fakes
        if isinstance(args.samples_per_cond, str) and args.samples_per_cond.lower()=="match":
            n_synth = min(real_dict[0].shape[0], real_dict[1].shape[0])
        else:
            n_synth = int(args.samples_per_cond)

        # Synthesize
        fakes = synthesize_for_posture(args.runs_root, p, args.noise_dim, n_synth, device)

        # Balance counts across both conds
        n = min(real_dict[0].shape[0], real_dict[1].shape[0], fakes[0].shape[0], fakes[1].shape[0])

        # Build features per condition
        for cond in (0,1):
            R = real_dict[cond][:n]  # (n,C,T)
            G = fakes[cond][:n]      # (n,C,T)
            Fr = psd_features(R); Fg = psd_features(G)

            Fr_list.append(Fr); Fg_list.append(Fg)

            # posture/cond/domain metadata
            P_all += [p]*len(Fr) + [p]*len(Fg)
            C_all += [cond]*len(Fr) + [cond]*len(Fg)
            D_all += [1]*len(Fr) + [0]*len(Fg)

    # Stack
    Fr_all = np.vstack(Fr_list) if Fr_list else np.zeros((0, C*64), np.float32)
    Fg_all = np.vstack(Fg_list) if Fg_list else np.zeros((0, C*64), np.float32)

    # Global standardize features (optional but helpful)
    scaler = StandardScaler()
    X_all = np.vstack([Fr_all, Fg_all])
    X_all = np.nan_to_num(scaler.fit_transform(X_all), 0, 0, 0)
    Fr_all = X_all[:len(Fr_all)]
    Fg_all = X_all[len(Fr_all):]

    # PCA plot (36 clusters)
    plot_embeddings(Fr_all, Fg_all, P_all, C_all, D_all,
                    out_path=out_dir/"pca_36.png",
                    title="PCA: 9 Postures × 2 Conditions × {Real, Synthetic}",
                    tsne=False, seed=args.seed)

    # t-SNE plot (36 clusters)
    plot_embeddings(Fr_all, Fg_all, P_all, C_all, D_all,
                    out_path=out_dir/"tsne_36.png",
                    title="t-SNE: Posture×Condition (● Real, × Gen)",
                    tsne=True, tsne_perp=args.tsne_perplexity, tsne_iter=args.tsne_iter, seed=args.seed)

    print(f"Saved PCA and t-SNE to {out_dir}")

if __name__ == "__main__":
    np.random.seed(0)
    main()
