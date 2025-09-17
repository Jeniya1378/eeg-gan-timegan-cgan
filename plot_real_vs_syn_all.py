#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
FS_FALLBACK = 128.0    # used only if 'fs' not in NPZ
PLOT_ALL_CHANNELS = True
FIRST_K = 4            # used only if PLOT_ALL_CHANNELS=False
N_SAMPLES = 4
STANDARDIZE = True     # per-sequence z-score
SMOOTH = 1             # odd moving-average; 1 disables
OUT_DIR = Path("./rvsg_plots")

# ---- Auto-trimming controls (per posture×condition) ----
AUTO_TRIM   = True
BASE_WIN    = 16
MAX_TRIM    = 64
FIXED_TRIM  = 16       # used only if AUTO_TRIM=False
TRIM_BOTH   = True
# =======================================================

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

def pick_synth_file(run_dir: Path):
    preferred = [run_dir / "synthetic_long.npz", run_dir / "synthetic.npz"]
    for c in preferred:
        if c.exists(): return c
    npzs = sorted([p for p in run_dir.glob("*.npz") if p.suffix == ".npz"])
    return npzs[0] if npzs else None

def to_str_list(arr, C):
    try:
        lst = [x.decode() if hasattr(x, "decode") else str(x) for x in list(arr)]
        if len(lst) == C:
            return lst
    except Exception:
        pass
    return [f"ch{i}" for i in range(C)]

def get_channel_names(npz_obj, C):
    for key in ("ch_names", "channels", "channel_names"):
        if key in npz_obj.files:
            return to_str_list(npz_obj[key], C)
    return [f"ch{i}" for i in range(C)]

def get_fs(npz_obj):
    if "fs" in npz_obj.files:
        try:
            v = np.array(npz_obj["fs"]).astype(float)
            return float(np.median(v))
        except Exception:
            pass
    return float(FS_FALLBACK)

def standardize_per_seq(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd

def smooth_ma_1d(x, k=1):
    if k <= 1: return x
    k = int(k) + (int(k) % 2 == 0)  # force odd
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

def estimate_prefix_to_trim(fake, base_win=16, max_trim=64,
                            tol_mean=0.18, tol_step=0.15, use_first_C=None):
    """Estimate how many initial samples to trim from *synthetic*."""
    X = standardize_per_seq(fake.copy())
    N, T, C = X.shape
    if T < 3: return 0
    W = max(4, min(base_win, max(2, T // 6)))
    Csel = min(C, use_first_C if use_first_C is not None else C)
    max_L = min(max_trim, max(0, T - 2*W - 1)) if T >= 2*W + 2 else 0
    best = 0
    for L in range(0, max_L + 1):
        seg1 = X[:, L:L+W, :Csel].mean()
        seg2 = X[:, L+W:L+2*W, :Csel].mean()
        if abs(seg1) <= tol_mean and abs(seg1 - seg2) <= tol_step:
            best = L
            break
    if best == 0 and max_L > 0:
        best = min(base_win, max_trim, max_L)
    return max(best, 0)

def samples_grid(real, fake, ch_names, n_samples, fs, out_path, title):
    N, T, C = real.shape
    n_samples = min(n_samples, N)
    rng = np.random.RandomState(0)
    idx = rng.choice(N, size=n_samples, replace=False)
    t = (np.arange(T) / fs) if fs > 0 else np.arange(T)

    rows = len(ch_names); cols = n_samples
    fig, axes = plt.subplots(rows, cols, figsize=(2.0*cols + 1.8, 1.0*rows + 1.8), sharex=True)
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes[np.newaxis, :]
    elif cols == 1: axes = axes[:, np.newaxis]

    for r, c in np.ndindex(rows, cols):
        ax = axes[r, c]
        ch = r; n = idx[c]
        ax.plot(t, real[n, :, ch], lw=1.0, alpha=0.9, label="real")
        ax.plot(t, fake[n, :, ch], lw=1.0, alpha=0.9, linestyle="--", label="synth")
        if r == 0: ax.set_title(f"sample #{n}", fontsize=9)
        if c == 0: ax.set_ylabel(ch_names[ch], fontsize=9)
        if r == rows-1: ax.set_xlabel("time (s)" if fs>0 else "t", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.margins(x=0)  # no x padding

    axes[0,0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title + " — samples", fontsize=11)
    fig.tight_layout(rect=[0, 0.00, 1, 0.95])
    fig.savefig(out_path, dpi=155)
    plt.close(fig)

def summary_plot(real, fake, ch_names, fs, out_path, title):
    N, T, C = real.shape
    t = (np.arange(T) / fs) if fs > 0 else np.arange(T)
    rows = len(ch_names)
    fig, axes = plt.subplots(rows, 1, figsize=(8.2, 1.35*rows + 1.6), sharex=True)
    if rows == 1: axes = [axes]

    for r in range(rows):
        ax = axes[r]
        r_mu, r_sd = real[:, :, r].mean(axis=0), real[:, :, r].std(axis=0)
        f_mu, f_sd = fake[:, :, r].mean(axis=0), fake[:, :, r].std(axis=0)
        ax.plot(t, r_mu, lw=1.4, label="real μ")
        ax.fill_between(t, r_mu - r_sd, r_mu + r_sd, alpha=0.15, label="real ±σ")
        ax.plot(t, f_mu, lw=1.4, linestyle="--", label="synth μ")
        ax.fill_between(t, f_mu - f_sd, f_mu + f_sd, alpha=0.15, label="synth ±σ")
        ax.set_ylabel(ch_names[r], fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.margins(x=0)  # remove left-edge padding
        # NOTE: removed vertical t=0 line to avoid the “hurting” stripe

        if r == 0:
            ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")

    axes[-1].set_xlabel("time (s)" if fs>0 else "t", fontsize=10)
    fig.suptitle(title + " — mean ± std", fontsize=11)
    fig.tight_layout(rect=[0, 0.00, 1, 0.95])
    fig.savefig(out_path, dpi=165)
    plt.close(fig)

def process_one(real_fp: Path, synth_fp: Path):
    stem = real_fp.stem

    try:
        with np.load(real_fp, allow_pickle=True) as dreal:
            real = dreal["X"].astype(np.float32)
            fs = get_fs(dreal)
            ch_names = get_channel_names(dreal, real.shape[2])
    except Exception as e:
        print(f"[skip] Failed to load real {real_fp.name}: {e}")
        return False

    try:
        with np.load(synth_fp, allow_pickle=True) as dfake:
            fake = dfake["X"].astype(np.float32)
    except Exception as e:
        print(f"[skip] Failed to load synthetic {synth_fp}: {e}")
        return False

    # Match batch count
    m = min(len(real), len(fake))
    real, fake = real[:m], fake[:m]

    # Auto/fixed trim
    if AUTO_TRIM:
        L = estimate_prefix_to_trim(fake, base_win=BASE_WIN, max_trim=MAX_TRIM,
                                    tol_mean=0.18, tol_step=0.15, use_first_C=min(8, fake.shape[2]))
        reason = "auto"
    else:
        L = int(FIXED_TRIM); reason = "fixed"

    L = max(0, min(L, fake.shape[1]-2))
    if L > 0:
        if TRIM_BOTH: real = real[:, L:, :]
        fake = fake[:, L:, :]
        print(f"[trim-{reason}] {stem}: trimmed {L} samples")
    else:
        print(f"[trim-{reason}] {stem}: no trimming")

    # Normalize/smooth AFTER trimming
    if STANDARDIZE:
        real = standardize_per_seq(real)
        fake = standardize_per_seq(fake)
    if SMOOTH > 1:
        real = maybe_smooth(real, SMOOTH)
        fake = maybe_smooth(fake, SMOOTH)

    # Channel selection
    C = real.shape[2]
    ch_plot_names = ch_names[:C] if PLOT_ALL_CHANNELS else ch_names[:min(FIRST_K, C)]

    title = f"{stem} (N={len(real)}, T={real.shape[1]}, C={C})"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_grid(real, fake, ch_plot_names, N_SAMPLES, fs, OUT_DIR / f"samples_{stem}.png", title)
    summary_plot(real, fake, ch_plot_names, fs, OUT_DIR / f"summary_{stem}.png", title)
    print(f"[ok] {stem} -> plots saved.")
    return True

def main():
    data_dir, synth_root = load_config_or_defaults()
    real_files = sorted(data_dir.glob("posture*_*.npz"))
    if not real_files:
        print(f"[exit] No NPZs found in {data_dir.resolve()}")
        return

    processed = 0
    skipped = 0
    for fp in real_files:
        run_dir = synth_root / fp.stem
        if not run_dir.exists():
            print(f"[skip] No synthetic folder for {fp.stem} at {run_dir}")
            skipped += 1; continue
        synth_fp = pick_synth_file(run_dir)
        if synth_fp is None:
            print(f"[skip] No synthetic .npz inside {run_dir}")
            skipped += 1; continue
        ok = process_one(fp, synth_fp)
        processed += int(ok); skipped += int(not ok)

    print(f"\nDone. Processed: {processed}  Skipped: {skipped}")
    print(f"Outputs in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    np.random.seed(0)
    main()
