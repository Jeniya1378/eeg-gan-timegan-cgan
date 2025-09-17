# %%
# PSD & Bandpower Batch Plotter for EEG .npz files (Emotiv 14-ch style)
#
# Folder layout expected:
# /mnt/data/all npz/
#   ├── real/
#   │     ├── posture1_no_exo.npz
#   │     └── ...
#   └── synthetic/
#         ├── posture1_no_exo.npz
#         └── ...
#
# What this script does:
# - Loads every .npz in both "real" and "synthetic"
# - Computes per-channel PSD using Welch, averaged across epochs
# - Computes band powers for Delta/Theta/Alpha/Beta/Gamma
# - Saves 3 outputs per file:
#     (1) PSD curve plot (mean across channels with ±SEM shaded)
#     (2) Bandpower bar chart (per channel)
#     (3) bandpowers CSV (per channel)
# - Also creates two summary CSVs across all files (one for real, one for synthetic)
#
# No CLI args; paths are set below. Adjust ROOT_DIR if needed.

import os
import io
import math
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

# ---------------------
# Configuration
# ---------------------
ROOT_DIR = "all npz"  # adjust if your folder differs
SETS = ["real", "synthetic"]
OUT_DIR = "psd_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Welch params
N_PER_SEG = None  # auto: uses 1-second windows if None (we'll set from fs)
N_OVERLAP = None  # auto
NFFT = None       # auto

# EEG bands (Hz)
BANDS = {
    "Delta (1–4 Hz)": (1.0, 4.0),
    "Theta (4–8 Hz)": (4.0, 8.0),
    "Alpha (8–13 Hz)": (8.0, 13.0),
    "Beta (13–30 Hz)": (13.0, 30.0),
    "Gamma (30–45 Hz)": (30.0, 45.0),
}

# ---------------------
# Helpers
# ---------------------

def load_npz_safe(path):
    """Load npz allowing for pickled fields (e.g., ch_names)."""
    npz = np.load(path, allow_pickle=True)

    # Required data
    if "X" not in npz:
        raise ValueError("No EEG array 'X' in file")
    X = npz["X"]

    # Sampling rate
    if "fs" in npz:
        fs = npz["fs"]
        if isinstance(fs, np.ndarray):
            fs = fs.item()
        fs = float(fs)
    else:
        print(f"[INFO] No 'fs' key found in {path}, defaulting to 128 Hz")
        fs = 128.0  # <-- set default

    # Channels
    ch_names = None
    if "ch_names" in npz:
        ch_names = npz["ch_names"]
        if isinstance(ch_names, np.ndarray) and ch_names.dtype == object:
            ch_names = [str(x) for x in ch_names.tolist()]
        else:
            ch_names = ch_names.tolist()
    else:
        C = X.shape[-1]
        ch_names = [f"Ch{idx+1}" for idx in range(C)]

    return X, fs, ch_names


def compute_psd_per_channel(X, fs):
    """
    Compute PSD with Welch per epoch & channel, then average over epochs.
    X: (N, T, C)
    Returns: f (F,), psd_mean (C, F), psd_sem (C, F)
    """
    N, T, C = X.shape
    # determine Welch window defaults: 1-second windows if possible
    nperseg = int(fs) if N_PER_SEG is None else N_PER_SEG
    noverlap = int(nperseg // 2) if N_OVERLAP is None else N_OVERLAP

    # Collect PSDs for each epoch and channel
    # We'll accumulate mean & sem robustly
    all_psd = []  # list of arrays (C, F) per epoch
    f = None
    for i in range(N):
        epoch = X[i]  # (T, C)
        # scipy expects (nperseg,) per channel; so iterate channels
        epoch_psd = []
        for c in range(C):
            freqs, Pxx = welch(epoch[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=NFFT)
            if f is None:
                f = freqs
            epoch_psd.append(Pxx)
        epoch_psd = np.stack(epoch_psd, axis=0)  # (C, F)
        all_psd.append(epoch_psd)

    all_psd = np.stack(all_psd, axis=0)  # (N, C, F)
    psd_mean = np.mean(all_psd, axis=0)  # (C, F)
    psd_sem = np.std(all_psd, axis=0, ddof=1) / math.sqrt(max(1, N))  # (C, F)
    return f, psd_mean, psd_sem

def bandpower_from_psd(f, psd, band):
    """Integrate PSD over a frequency band. psd shape: (C, F)."""
    f_lo, f_hi = band
    idx = np.logical_and(f >= f_lo, f <= f_hi)
    if not np.any(idx):
        return np.zeros(psd.shape[0])
    # Trapezoidal integration across selected freqs
    return np.trapz(psd[:, idx], f[idx], axis=1)

def make_filename_safe(name):
    return "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", ".", " ")).rstrip()

def plot_mean_psd(f, psd_mean, psd_sem, title, out_path):
    """
    Plot mean PSD across channels (mean over channels, +/- SEM across channels).
    One figure per file to comply with single-plot rule.
    """
    ch_mean = psd_mean.mean(axis=0)            # (F,)
    ch_sem  = psd_sem.mean(axis=0)             # (F,)

    plt.figure(figsize=(8, 5))
    plt.plot(f, ch_mean, label="Mean PSD (across channels)")
    # Shaded SEM
    upper = ch_mean + ch_sem
    lower = ch_mean - ch_sem
    plt.fill_between(f, lower, upper, alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_bandpower_bars(ch_names, bandpower_dict, title, out_path):
    """
    Bar chart of bandpower per channel for all bands (stacked by band).
    To comply with 'single-plot per chart', we create one chart per file.
    """
    # We'll plot bands as grouped bars per channel by stacking with small offsets.
    bands = list(bandpower_dict.keys())
    C = len(ch_names)
    x = np.arange(C)

    width = 0.8 / max(1, len(bands))  # total width split among bands

    plt.figure(figsize=(10, 5))
    for i, band in enumerate(bands):
        vals = bandpower_dict[band]  # (C,)
        plt.bar(x + i * width, vals, width=width, label=band)
    plt.xticks(x + (len(bands)-1)*width/2, ch_names, rotation=45, ha="right")
    plt.ylabel("Band power (integrated PSD)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------
# Main processing
# ---------------------
summary_records = {"real": [], "synthetic": []}

for subset in SETS:
    in_dir = os.path.join(ROOT_DIR, subset)
    if not os.path.isdir(in_dir):
        continue
    out_dir_subset = os.path.join(OUT_DIR, subset)
    os.makedirs(out_dir_subset, exist_ok=True)

    for fname in sorted(os.listdir(in_dir)):
        if not fname.lower().endswith(".npz"):
            continue
        fpath = os.path.join(in_dir, fname)
        try:
            X, fs, ch_names = load_npz_safe(fpath)
        except Exception as e:
            print(f"[WARN] Could not load {fpath}: {e}")
            continue

        # Compute PSD
        f, psd_mean, psd_sem = compute_psd_per_channel(X, fs)

        # Compute bandpowers per channel
        bandpowers = {}
        for band_name, (lo, hi) in BANDS.items():
            bandpowers[band_name] = bandpower_from_psd(f, psd_mean, (lo, hi))

        # Save per-file CSV of bandpowers
        bp_df = pd.DataFrame(bandpowers, index=ch_names)
        csv_name = make_filename_safe(os.path.splitext(fname)[0]) + "_bandpowers.csv"
        bp_csv_path = os.path.join(out_dir_subset, csv_name)
        bp_df.to_csv(bp_csv_path)

        # Make PSD plot
        psd_plot_name = make_filename_safe(os.path.splitext(fname)[0]) + "_mean_PSD.png"
        psd_plot_path = os.path.join(out_dir_subset, psd_plot_name)
        plot_mean_psd(f, psd_mean, psd_sem, title=f"{subset.capitalize()} | {fname} | Mean PSD", out_path=psd_plot_path)

        # Make bandpower plot
        band_plot_name = make_filename_safe(os.path.splitext(fname)[0]) + "_bandpowers.png"
        band_plot_path = os.path.join(out_dir_subset, band_plot_name)
        plot_bandpower_bars(ch_names, bandpowers, title=f"{subset.capitalize()} | {fname} | Bandpowers per Channel", out_path=band_plot_path)

        # Add to summary records (mean over channels for each band)
        for band_name, vals in bandpowers.items():
            summary_records[subset].append({
                "file": fname,
                "fs": fs,
                "band": band_name,
                "mean_bandpower_across_channels": float(np.mean(vals)),
                "std_bandpower_across_channels": float(np.std(vals, ddof=1)),
                "n_channels": len(vals),
                "n_epochs": X.shape[0],
                "n_samples_per_epoch": X.shape[1],
            })

# Save summary CSVs
for subset in SETS:
    if summary_records[subset]:
        df = pd.DataFrame(summary_records[subset])
        out_csv = os.path.join(OUT_DIR, f"{subset}_summary_bandpowers.csv")
        df.to_csv(out_csv, index=False)

# List generated outputs
generated = []
for root, dirs, files in os.walk(OUT_DIR):
    for f in files:
        generated.append(os.path.join(root, f))

len(generated), sorted(generated)[:10]
