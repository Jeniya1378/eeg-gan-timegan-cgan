#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined PCA & t-SNE with compact legend, posture-2 darker colors, and
standalone legend image export.

- Color scheme:
    "36" (default): 36 distinct colors (posture×condition×domain)
    "18": 18 colors (posture×condition); domain shown by marker (real='o', gen='x')
- Marker: real='o' (dot), gen='x'
- Legend: compact font, 3 columns, abbreviated labels.
- NEW: Save legends as separate PNGs (pca_legend.png, tsne_legend.png).

Run example:
  python visualization_color_scheme_with_legends.py \      --real_dir ./preprocessed \      --synth_dir ./timegan_runs \      --out ./eval_out_plots
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------- data discovery ----------
def _find_synth_npz(run_dir: Path):
    for name in ["synthetic_long.npz", "synthetic.npz"]:
        fp = run_dir / name
        if fp.exists():
            return fp
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
    low = np.quantile(X, lo, axis=0)
    high = np.quantile(X, hi, axis=0)
    return np.clip(X, low, high)

def zscore(X):
    return StandardScaler().fit_transform(X)

def balanced_subsample(X, P, D, C, max_total=6000, seed=0):
    if len(X) <= max_total:
        return X, P, D, C
    rng = np.random.RandomState(seed)
    idxs = []
    posts = np.unique(P)
    for p in posts:
        for c in ["with_exo", "no_exo"]:
            for d in [1, 0]:
                m = (P == p) & (C == c) & (D == d)
                if not np.any(m):
                    continue
                take = max_total // (len(posts) * 2 * 2)
                take = min(take, m.sum())
                idxs.append(rng.choice(np.where(m)[0], size=take, replace=False))
    idx = np.concatenate(idxs) if idxs else np.arange(len(X))
    return X[idx], P[idx], D[idx], C[idx]

# ---------- colors & plotting ----------
def make_palette(n, cmap_name="hsv"):
    cmap = plt.cm.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]

def cluster_id(posture, cond, domain, color_scheme="36"):
    # posture: 1..9 -> 0..8 internally
    p = posture - 1
    c = 0 if cond == "with_exo" else 1
    d = 1 if domain == 1 else 0  # 1=real, 0=gen (order doesn't matter for color indexing)
    if color_scheme == "36":
        idx = p*4 + c*2 + (0 if d==1 else 1)  # 0..35
        # Force posture 2 into low indices (darker colors on common colormaps)
        if posture == 2:
            idx = c*2 + (0 if d==1 else 1)  # 0..3 bucket
        return idx
    else:
        idx = p*2 + c  # 0..17
        if posture == 2:
            idx = c  # keep it lower
        return idx

def scatter(ax, emb, P, D, C, colors, color_scheme="36"):
    # markers: real='o', gen='x'
    seen_labels = set()
    for p in np.unique(P):
        for c in ["with_exo", "no_exo"]:
            for d in [1, 0]:
                m = (P == p) & (C == c) & (D == d)
                if not np.any(m):
                    continue
                gid = cluster_id(p, c, d, color_scheme=color_scheme)
                label = f"P{p} {'W' if c=='with_exo' else 'N'} {'R' if d==1 else 'G'}"
                # avoid duplicate legend entries
                add_label = label not in seen_labels
                if add_label:
                    seen_labels.add(label)
                ax.scatter(
                    emb[m, 0], emb[m, 1],
                    s=9, alpha=0.8,
                    marker=('o' if d==1 else 'x'),
                    c=[colors[gid]],
                    label=(label if add_label else None),
                )

def add_compact_legend(ax, ncols=3, fontsize=5.5):
    lg = ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        ncol=ncols, fontsize=fontsize, frameon=False,
        borderaxespad=0.0, columnspacing=0.7, handlelength=1.2,
        handletextpad=0.3, markerscale=0.9
    )
    return lg

def save_legend(ax, out_path, ncols=3, fontsize=5.5, fig_size=(8, 2)):
    """Save the legend from an existing Axes to a standalone image."""
    handles, labels = ax.get_legend_handles_labels()
    fig_leg = plt.figure(figsize=fig_size)
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(
        handles, labels,
        ncol=ncols, fontsize=fontsize, frameon=False, loc="center",
        handlelength=1.2, handletextpad=0.3, columnspacing=0.7,
        markerscale=0.9
    )
    fig_leg.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig_leg)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--real_dir",  type=str, default="./preprocessed")
    ap.add_argument("--synth_dir", type=str, default="./timegan_runs")
    ap.add_argument("--out",       type=str, default="./eval_out_plots_color_scheme")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_max",  type=int, default=6000)
    ap.add_argument("--pca_keep",  type=int, default=50, help="PCs fed to t-SNE.")
    ap.add_argument("--winsor_low",  type=float, default=0.005)
    ap.add_argument("--winsor_high", type=float, default=0.995)
    ap.add_argument("--color_scheme", type=str, choices=["36","18"], default="36",
                    help="36 = each cluster unique color; 18 = color by (posture×cond)." )
    ap.add_argument("--cmap", type=str, default="hsv", help="Matplotlib colormap name.")
    ap.add_argument("--save_legends", action="store_true",
                    help="If set, save legends as standalone images.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(Path(args.real_dir), Path(args.synth_dir))
    if not pairs:
        raise SystemExit("No matching (posture, condition) pairs found.")

    X_blocks, P, D, C = [], [], [], []
    for (p, c), (r, f) in sorted(pairs.items()):
        X_blocks.append(np.concatenate([r, f], axis=0))
        P += [p]*(len(r)+len(f))
        D += [1]*len(r) + [0]*len(f)
        C += [c]*(len(r)+len(f))
    P = np.array(P); D = np.array(D); C = np.array(C)

    # preprocess
    X = flatten(X_blocks)
    X = winsorize(X, args.winsor_low, args.winsor_high)
    X = zscore(X)

    # color palette
    n_colors = 36 if args.color_scheme == "36" else 18
    colors = make_palette(n_colors, cmap_name=args.cmap)

    # ---------- PCA ----------
    pca2 = PCA(n_components=2, random_state=args.seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    scatter(ax, pca2, P, D, C, colors, color_scheme=args.color_scheme)
    ax.set_title("Combined PCA: Posture×Condition (● Real, × Gen)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    fig.subplots_adjust(right=0.78)  # leave room for big legend
    add_compact_legend(ax, ncols=3, fontsize=5.2)
    fig.savefig(out_dir/"pca_combined.png", dpi=170)
    if args.save_legends:
        save_legend(ax, out_dir/"pca_legend.png", ncols=3, fontsize=5.2, fig_size=(10, 2.2))
    plt.close(fig)

    # ---------- t-SNE ----------
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
    add_compact_legend(ax, ncols=3, fontsize=5.2)
    fig.savefig(out_dir/"tsne_combined.png", dpi=170)
    if args.save_legends:
        save_legend(ax, out_dir/"tsne_legend.png", ncols=3, fontsize=5.2, fig_size=(10, 2.2))
    plt.close(fig)

    print(f"Saved PCA and t-SNE (and legends if requested) to {out_dir}")

if __name__ == "__main__":
    np.random.seed(0)
    main()
