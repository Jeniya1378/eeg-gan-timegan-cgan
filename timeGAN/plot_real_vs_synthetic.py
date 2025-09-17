#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-detect real & synthetic EEG for the first available posture×condition
and plot quick comparisons — no CLI args needed.

Outputs:
  ./rvsg_plots/samples_<stem>.png
  ./rvsg_plots/summary_<stem>.png
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------- config & paths (mirror your training defaults) ----------------
def load_config_or_defaults():
    cfg_path = Path("timegan_config.json")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        data_dir = Path(cfg.get("data_dir", "./preprocessed"))
        synth_dir = Path(cfg.get("out_dir", "./timegan_runs"))
    else:
        data_dir = Path("./preprocessed")
        synth_dir = Path("./timegan_runs")
    return data_dir, synth_dir

def find_pair(data_dir: Path, synth_root: Path):
    # Pick the first posture*_*.npz that also has a matching synthetic folder.
    files = sorted(data_dir.glob("posture*_*.npz"))
    if not files:
        raise SystemExit(f"No NPZs found in {data_dir}")
    for fp in files:
        run_dir = synth_root / fp.stem
        if run_dir.exists():
            # Prefer synthetic_long.npz, then synthetic.npz, else first *.npz
            cand = [run_dir / "synthetic_long.npz", run_dir / "synthetic.npz"]
            synth = None
            for c in cand:
                if c.exists():
                    synth = c
                    break
            if synth is None:
                npzs = sorted(run_dir.glob("*.npz"))
                synth = npzs[0] if npzs else None
            if synth is not None and synth.exists():
                return fp, synth
    raise SystemExit(f"No matching synthetic run found under {synth_root} for any posture*_*.npz")

# ---------------- helpers ----------------
def standardize_per_seq(X):
    """Channel-wise z-score per sequence over time."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd

def smooth_ma_1d(x, k=1):
    if k <= 1: return x
    k = int(k) + (int(k) % 2 == 0)  # odd
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    ker = np.ones(k, dtype=float) / k
    return np.convolve(xp, ker, mode="valid")

def maybe_smooth(X, k=1):
    if k <= 1: return X
    N, T, C = X.shape
    Y = np.empty_like(X)
    for n in range(N):
        for c in range(C):
            Y[n, :, c] = smooth_ma_1d(X[n, :, c], k)
    return Y

# ---------------- plotting ----------------
def samples_grid(real, fake, ch_idx, n_samples, fs, out_path, title):
    N, T, _ = real.shape
    n_samples = min(n_samples, N)
    rng = np.random.RandomState(0)
    idx = rng.choice(N, size=n_samples, replace=False)
    t = (np.arange(T) / fs) if fs > 0 else np.arange(T)

    rows = len(ch_idx); cols = n_samples
    fig, axes = plt.subplots(rows, cols, figsize=(1.9*cols + 1.5, 1.2*rows + 1.2), sharex=True)
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[np.newaxis, :]
    elif cols == 1: axes = axes[:, np.newaxis]

    for r, c in np.ndindex(rows, cols):
        ax = axes[r, c]
        ch = ch_idx[r]
        n = idx[c]
        ax.plot(t, real[n, :, ch], lw=1.0, alpha=0.9, label="real")
        ax.plot(t, fake[n, :, ch], lw=1.0, alpha=0.9, linestyle="--", label="synth")
        if r == 0:
            ax.set_title(f"sample #{n}", fontsize=9)
        if c == 0:
            ax.set_ylabel(f"ch{ch}", fontsize=9)
        if r == rows-1:
            ax.set_xlabel("time (s)" if fs>0 else "t", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0,0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title + " — samples", fontsize=11)
    fig.tight_layout(rect=[0, 0.00, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def summary_plot(real, fake, ch_idx, fs, out_path, title):
    N, T, _ = real.shape
    t = (np.arange(T) / fs) if fs > 0 else np.arange(T)
    rows = len(ch_idx)
    fig, axes = plt.subplots(rows, 1, figsize=(7.8, 1.8*rows + 1.2), sharex=True)
    if rows == 1: axes = [axes]

    for r, ch in enumerate(ch_idx):
        ax = axes[r]
        r_mu, r_sd = real[:, :, ch].mean(axis=0), real[:, :, ch].std(axis=0)
        f_mu, f_sd = fake[:, :, ch].mean(axis=0), fake[:, :, ch].std(axis=0)
        ax.plot(t, r_mu, lw=1.4, label="real μ")
        ax.fill_between(t, r_mu - r_sd, r_mu + r_sd, alpha=0.15, label="real ±σ")
        ax.plot(t, f_mu, lw=1.4, linestyle="--", label="synth μ")
        ax.fill_between(t, f_mu - f_sd, f_mu + f_sd, alpha=0.15, label="synth ±σ")
        ax.set_ylabel(f"ch{ch}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if r == 0:
            ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    axes[-1].set_xlabel("time (s)" if fs>0 else "t", fontsize=10)

    fig.suptitle(title + " — mean ± std", fontsize=11)
    fig.tight_layout(rect=[0, 0.00, 1, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

# ---------------- main (no args) ----------------
def main():
    # Defaults you can tweak here if you want
    FS = 128.0          # sampling rate for x-axis (seconds). Set 0 to use sample index.
    FIRST_K = 4         # number of channels to show
    N_SAMPLES = 4       # how many random trials to overlay
    STANDARDIZE = True  # per-sequence z-score (helps shape comparison)
    SMOOTH = 1          # moving average window (odd int). 1 disables smoothing.
    OUT_DIR = Path("./rvsg_plots")

    data_dir, synth_root = load_config_or_defaults()
    real_fp, synth_fp = find_pair(data_dir, synth_root)

    real = np.load(real_fp)["X"].astype(np.float32)   # (N,T,C)
    fake = np.load(synth_fp)["X"].astype(np.float32)  # (N,T,C)
    m = min(len(real), len(fake))
    real, fake = real[:m], fake[:m]

    # Optional standardization & smoothing
    if STANDARDIZE:
        real = standardize_per_seq(real)
        fake = standardize_per_seq(fake)
    if SMOOTH > 1:
        real = maybe_smooth(real, SMOOTH)
        fake = maybe_smooth(fake, SMOOTH)

    C = real.shape[2]
    ch_idx = list(range(min(FIRST_K, C)))
    stem = real_fp.stem  # postureX_with_exo
    title = f"{stem} (N={len(real)}, T={real.shape[1]}, C={real.shape[2]})"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples_grid(real, fake, ch_idx, N_SAMPLES, FS, OUT_DIR / f"samples_{stem}.png", title)
    summary_plot(real, fake, ch_idx, FS, OUT_DIR / f"summary_{stem}.png", title)
    print(f"Saved:\n  {OUT_DIR / f'samples_{stem}.png'}\n  {OUT_DIR / f'summary_{stem}.png'}")

if __name__ == "__main__":
    np.random.seed(0)
    main()
