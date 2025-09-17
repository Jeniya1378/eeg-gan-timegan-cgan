#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global PCA & t-SNE + automatic per-(posture×condition) zooms.

This revision preserves the original structure but:
1) Uses the SAME color mapping (including posture-2 darker bias in 36-color mode)
   as our paired-legend script.
2) Formats the global legend as 18 rows × 2 columns with labels:
   "Posture-i No exo Real" and "Posture-i No exo Synthetic" (then With exo).

Run:
  python visualization_v2_paired_colors.py --real_dir ./preprocessed --synth_dir ./timegan_runs --out ./eval_out_plots
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------- file discovery ----------
def _find_synth_npz(run_dir: Path):
    for name in ["synthetic_long.npz", "synthetic.npz"]:
        fp = run_dir / name
        if fp.exists(): return fp
    npzs = sorted(run_dir.glob("*.npz"))
    return npzs[0] if npzs else None

def load_pairs(real_dir: Path, synth_dir: Path):
    pairs = {}
    for p in range(1, 10):
        for cond in ["with_exo", "no_exo"]:
            rfp = real_dir / f"posture{p}_{cond}.npz"
            sdir = synth_dir / f"posture{p}_{cond}"
            sfp  = _find_synth_npz(sdir)
            if rfp.exists() and sfp and sfp.exists():
                r = np.load(rfp)["X"].astype(np.float32)
                f = np.load(sfp)["X"].astype(np.float32)
                m = min(len(r), len(f))
                if m > 0:
                    pairs[(p, cond)] = (r[:m], f[:m])
    return pairs

# ---------- preprocessing ----------
def flatten(X_list):
    X = np.concatenate(X_list, axis=0)
    return X.reshape(len(X), -1)

def winsorize(X, lo=0.005, hi=0.995):
    low = np.quantile(X, lo, axis=0); high = np.quantile(X, hi, axis=0)
    return np.clip(X, low, high)

def zscore(X):
    return StandardScaler().fit_transform(X)

def balanced_subsample(X, P, D, C, max_total=6000, seed=0):
    if len(X) <= max_total: return X, P, D, C
    rng = np.random.RandomState(seed)
    idxs = []
    posts = np.unique(P)
    for p in posts:
        for c in ["with_exo", "no_exo"]:
            for d in [1,0]:
                m = (P==p) & (C==c) & (D==d)
                if not np.any(m): continue
                take = max_total // (len(posts)*2*2)
                take = min(take, m.sum())
                idxs.append(rng.choice(np.where(m)[0], size=take, replace=False))
    idx = np.concatenate(idxs) if idxs else np.arange(len(X))
    return X[idx], P[idx], D[idx], C[idx]

# ---------- colors & plotting ----------
def make_palette(n, cmap_name="hsv"):
    cmap = plt.cm.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]

def cluster_id(posture, cond, domain, color_scheme="36"):
    p = posture - 1
    c = 0 if cond == "with_exo" else 1
    d = 1 if domain == 1 else 0  # 1=real, 0=gen
    if color_scheme == "36":
        idx = p*4 + c*2 + (0 if d==1 else 1)  # 0..35
        # Bias posture-2 to darker (low) indices to match other script
        if posture == 2:
            idx = c*2 + (0 if d==1 else 1)  # 0..3
        return idx
    else:
        idx = p*2 + c  # 0..17
        # Uncomment to also bias posture-2 in 18-color mode:
        # if posture == 2: idx = c
        return idx

def scatter(ax, emb, P, D, C, colors, color_scheme="36"):
    # Use proxy legend; avoid automatic labels here
    for p in np.unique(P):
        for c in ["with_exo","no_exo"]:
            for d in [1,0]:
                m = (P==p) & (C==c) & (D==d)
                if not np.any(m): continue
                gid = cluster_id(p, c, d, color_scheme=color_scheme)
                ax.scatter(emb[m,0], emb[m,1], s=9, alpha=0.8,
                           marker=('o' if d==1 else 'x'), c=[colors[gid]], label=None)

def add_paired_legend(ax, colors, color_scheme="36", fontsize=5.6):
    """2 columns × 18 rows legend with explicit paired labels (Real/Synthetic)."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for p in range(1, 10):
        for cond in ["no_exo", "with_exo"]:  # No exo first per request
            # Real
            gid_r = cluster_id(p, cond, 1, color_scheme=color_scheme)
            handles.append(Line2D([], [], linestyle='None', marker='o', markersize=5, color=colors[gid_r]))
            labels.append(f"Posture-{p} {'No exo' if cond=='no_exo' else 'With exo'} Real")
            # Synthetic
            gid_g = cluster_id(p, cond, 0, color_scheme=color_scheme)
            handles.append(Line2D([], [], linestyle='None', marker='x', markersize=5, color=colors[gid_g]))
            labels.append(f"Posture-{p} {'No exo' if cond=='no_exo' else 'With exo'} Synthetic")
    lg = ax.legend(handles, labels, ncol=2, fontsize=fontsize, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.02, 1),
                   borderaxespad=0.0, columnspacing=1.0, handlelength=1.2,
                   handletextpad=0.5, markerscale=1.0)
    return lg

def scatter_two(ax, emb, is_real, colors, p, cond, color_scheme="36"):
    gid_r = cluster_id(p, cond, 1, color_scheme=color_scheme)
    gid_g = cluster_id(p, cond, 0, color_scheme=color_scheme)
    if np.any(is_real):
        ax.scatter(emb[is_real,0], emb[is_real,1], s=14, alpha=0.9, marker='o',
                   c=[colors[gid_r]], label=f"P{p} {'W' if cond=='with_exo' else 'N'} R")
    if np.any(~is_real):
        ax.scatter(emb[~is_real,0], emb[~is_real,1], s=14, alpha=0.9, marker='x',
                   c=[colors[gid_g]], label=f"P{p} {'W' if cond=='with_exo' else 'N'} G")

def zoom_one_cluster(X, P, D, C, colors, p, cond, out_dir, args):
    ms = (P==p) & (C==cond)
    if not np.any(ms): return False
    Xs, Ds = X[ms], D[ms]

    # PCA (subset)
    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    scatter_two(ax, pca2, Ds==1, colors, p, cond, color_scheme=args.color_scheme)
    ax.set_title(f"ZOOM PCA: P{p} – {cond} (●R, ×G)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir/f"zoom_p{p}_{cond}_pca.png", dpi=180); plt.close(fig)

    # t-SNE (subset)
    K = min(args.pca_keep, Xs.shape[1])
    Xred = PCA(n_components=K, random_state=args.seed).fit_transform(Xs)
    perp = min(args.tsne_perplexity, max(5, len(Xs)//3))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=args.seed)
    ts2 = tsne.fit_transform(Xred)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    scatter_two(ax, ts2, Ds==1, colors, p, cond, color_scheme=args.color_scheme)
    ax.set_title(f"ZOOM t-SNE: P{p} – {cond} (●R, ×G)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir/f"zoom_p{p}_{cond}_tsne.png", dpi=180); plt.close(fig)
    return True

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--real_dir",  type=str, default="./preprocessed")
    ap.add_argument("--synth_dir", type=str, default="./timegan_runs")
    ap.add_argument("--out",       type=str, default="./eval_out_plots_final")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_max",  type=int, default=6000)
    ap.add_argument("--pca_keep",  type=int, default=50)
    ap.add_argument("--winsor_low",  type=float, default=0.005)
    ap.add_argument("--winsor_high", type=float, default=0.995)
    ap.add_argument("--color_scheme", type=str, choices=["36","18"], default="36",
                    help="36=unique color for domain; 18=same color for real/gen.")
    ap.add_argument("--cmap", type=str, default="hsv", help="Matplotlib colormap name.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(Path(args.real_dir), Path(args.synth_dir))
    if not pairs: raise SystemExit("No matching (posture, condition) pairs found.")

    # Build arrays
    X_blocks, P, D, C = [], [], [], []
    for (p, c), (r, f) in sorted(pairs.items()):
        X_blocks.append(np.concatenate([r, f], axis=0))
        P += [p]*(len(r)+len(f))
        D += [1]*len(r) + [0]*len(f)
        C += [c]*(len(r)+len(f))
    P = np.array(P); D = np.array(D); C = np.array(C)

    # Preprocess globally
    X = flatten(X_blocks)
    X = winsorize(X, args.winsor_low, args.winsor_high)
    X = zscore(X)

    # Colors
    n_colors = 36 if args.color_scheme == "36" else 18
    colors = make_palette(n_colors, cmap_name=args.cmap)

    # Global PCA
    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    scatter(ax, pca2, P, D, C, colors, color_scheme=args.color_scheme)
    ax.set_title("Combined PCA: Posture×Condition (● Real, × Gen)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    fig.subplots_adjust(right=0.78)
    add_paired_legend(ax, colors, color_scheme=args.color_scheme, fontsize=5.6)
    fig.savefig(out_dir/"pca_combined_v2.png", dpi=170); plt.close(fig)

    # Global t-SNE
    Xb, Pb, Db, Cb = balanced_subsample(X, P, D, C, max_total=args.tsne_max, seed=args.seed)
    K = min(args.pca_keep, Xb.shape[1])
    Xred = PCA(n_components=K, random_state=args.seed).fit_transform(Xb)
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, init="pca", random_state=args.seed)
    ts2 = tsne.fit_transform(Xred)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    scatter(ax, ts2, Pb, Db, Cb, colors, color_scheme=args.color_scheme)
    ax.set_title("Combined t-SNE: Posture×Condition (● Real, × Gen)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    fig.subplots_adjust(right=0.78)
    add_paired_legend(ax, colors, color_scheme=args.color_scheme, fontsize=5.6)
    fig.savefig(out_dir/"tsne_combined_v2.png", dpi=170); plt.close(fig)

    # Per-(posture×condition) zooms
    saved_any = False
    for p in range(1, 10):
        for cond in ["with_exo","no_exo"]:
            ok = zoom_one_cluster(X, P, D, C, colors, p, cond, out_dir, args)
            saved_any = saved_any or ok

    print(f"Saved global and zoom plots to {out_dir}")
    if not saved_any:
        print("Warning: no per-(posture×condition) zooms saved (likely missing data for all pairs).");

if __name__ == "__main__":
    np.random.seed(0)
    main()
