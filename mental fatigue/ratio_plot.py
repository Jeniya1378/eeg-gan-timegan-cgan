# %%
# EEG Mental Workload & Fatigue – FULL SCRIPT with grouped-by-region channel plots
#
# Folder layout (no CLI args; edit the three paths below if needed):
#   ROOT_DIR = "all npz"
#     ├── real/*.npz
#     └── synthetic/*.npz
#   OUT_DIR  = "psd_outputs"
#
# What this script does (per .npz file):
#   1) Loads X (N,T,C), fs (defaults 128 if missing), ch_names (fallback Ch1..ChC)
#   2) Welch PSD → per-channel bandpowers (Delta/Theta/Alpha/Beta/Gamma)
#   3) Computes indices:
#        - Workload θf/αp  (frontal-theta / parietal–occipital-alpha)
#        - Theta/Alpha, TBR (Theta/Beta), TABR ((Theta+Alpha)/Beta), ABR (Alpha/Beta)
#   4) Saves CSVs:
#        - <name>_bandpowers.csv (per channel)
#        - <name>_indices_per_channel.csv (per channel ratios)
#        - <name>_indices_summary.csv (file-level scalars including θf/αp)
#   5) Saves PLOTS (grouped-by-region across x-axis for ALL channels):
#        - PSD_mean.png (reference)
#        - MF_TABR.png   (TABR per channel, grouped by regions)
#        - TBR.png       (per channel)
#        - ABR.png       (per channel)
#        - ThetaAlpha.png(per channel)
#        - Alpha.png     (per channel alpha power, for context)
#        - Workload_thetaF_over_alphaPO.png (scalar bar for θf/αp)
#
# Notes:
#  - Regional groups assume Emotiv EPOC+ naming. If some channels are missing,
#    plots still render with the channels that exist (and global fallback for θf/αp).
#
import os
import math
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt

# ---------------------
# Config (edit if needed)
# ---------------------
ROOT_DIR = "all npz"
SETS = ["real", "synthetic"]
OUT_DIR = "psd_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Welch params (auto: 1s windows @ fs, 50% overlap)
N_PER_SEG = None
N_OVERLAP = None
NFFT = None

# EEG bands
BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# Region ordering for Emotiv 14ch
REGION_ORDER = {
    "Frontal":   ["AF3", "AF4", "F3", "F4", "FC5", "FC6", "F7", "F8"],
    "Temporal":  ["T7", "T8"],
    "Parietal":  ["P7", "P8"],
    "Occipital": ["O1", "O2"],
}

# Prefer numpy.trapezoid if present
trapz_fn = getattr(np, "trapezoid", np.trapz)

# ---------------------
# Helpers
# ---------------------
def make_filename_safe(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-", ".", " ")).rstrip()

def load_npz_safe(path):
    npz = np.load(path, allow_pickle=True)
    if "X" not in npz:
        raise ValueError("No EEG array 'X' in file")
    X = npz["X"]
    if "fs" in npz:
        fs = npz["fs"]
        if isinstance(fs, np.ndarray):
            fs = fs.item()
        fs = float(fs)
    else:
        print(f"[INFO] No 'fs' in {path}; defaulting to 128 Hz")
        fs = 128.0
    if "ch_names" in npz:
        ch = npz["ch_names"]
        try:
            ch_names = ch.tolist() if hasattr(ch, "tolist") else list(ch)
            ch_names = [str(x) for x in ch_names]
        except Exception:
            C = X.shape[-1]
            ch_names = [f"Ch{idx+1}" for idx in range(C)]
    else:
        C = X.shape[-1]
        ch_names = [f"Ch{idx+1}" for idx in range(C)]
    return X, fs, ch_names

def compute_psd_per_channel(X, fs):
    N, T, C = X.shape
    nperseg = int(fs) if N_PER_SEG is None else N_PER_SEG
    noverlap = int(nperseg // 2) if N_OVERLAP is None else N_OVERLAP
    f = None
    all_psd = []
    for i in range(N):
        epoch = X[i]  # (T, C)
        ch_list = []
        for c in range(C):
            freqs, Pxx = welch(epoch[:, c], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=NFFT)
            if f is None:
                f = freqs
            ch_list.append(Pxx)
        all_psd.append(np.stack(ch_list, axis=0))  # (C, F)
    all_psd = np.stack(all_psd, axis=0)  # (N, C, F)
    psd_mean = all_psd.mean(axis=0)      # (C, F)
    psd_sem  = all_psd.std(axis=0, ddof=1) / math.sqrt(max(1, N))
    return f, psd_mean, psd_sem

def bandpower_from_psd(f, psd, lo, hi):
    idx = (f >= lo) & (f <= hi)
    if not np.any(idx):
        return np.zeros(psd.shape[0])
    return trapz_fn(psd[:, idx], f[idx], axis=1)  # (C,)

def compute_bandpowers(f, psd_mean):
    out = {}
    for name, (lo, hi) in BANDS.items():
        out[name] = bandpower_from_psd(f, psd_mean, lo, hi)
    return out  # dict band -> (C,)

def order_channels_grouped(ch_names):
    """
    Returns: ordered_names, ordered_indices, boundaries
      - ordered_names: list of channel names in region-grouped order
      - ordered_indices: indices in original ch_names corresponding to ordered_names
      - boundaries: list of (region, start_idx, end_idx) for annotating regions
    """
    present = set(ch_names)
    ordered_names = []
    ordered_indices = []
    boundaries = []
    cursor = 0
    for region, group in REGION_ORDER.items():
        group_present = [ch for ch in group if ch in present]
        if group_present:
            start = cursor
            for ch in group_present:
                ordered_names.append(ch)
                ordered_indices.append(ch_names.index(ch))
                cursor += 1
            end = cursor  # exclusive
            boundaries.append((region, start, end))
    # Add any remaining channels not in REGION_ORDER at the end
    for i, ch in enumerate(ch_names):
        if ch not in ordered_names:
            boundaries.append(("Other", len(ordered_names), len(ordered_names)+1))
            ordered_names.append(ch)
            ordered_indices.append(i)
    return ordered_names, ordered_indices, boundaries

def grouped_bar_plot(ch_names, values, y_label, title, out_path):
    """
    Plot all channels on x-axis, grouped by region (Frontal/Temporal/Parietal/Occipital).
    """
    ordered_names, idxs, boundaries = order_channels_grouped(ch_names)
    vals = [values[i] for i in idxs]
    x = np.arange(len(ordered_names))

    plt.figure(figsize=(12, 6))
    plt.bar(x, vals)
    plt.xticks(x, ordered_names, rotation=45, ha="right")
    plt.ylabel(y_label); plt.title(title)

    # Add region dividers and labels
    if boundaries:
        ymax = max(vals) if len(vals) else 1.0
        for region, start, end in boundaries:
            # vertical line at boundary between regions (except at start 0)
            if start > 0:
                plt.axvline(start - 0.5, linestyle="--", alpha=0.5)
            # region label centered above group
            cx = (start + end - 1) / 2.0
            plt.text(cx, ymax * 1.05, region, ha="center", va="bottom", fontsize=10)

    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def scalar_bar(value, label, title, out_path):
    plt.figure(figsize=(5, 5))
    plt.bar([label], [value])
    plt.ylabel(label); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def select_indices(names, wanted_set):
    return [i for i, n in enumerate(names) if n in wanted_set]

def safe_mean(vals, idxs):
    if len(idxs) == 0:
        return float(np.mean(vals))  # fallback to global
    return float(np.mean([vals[i] for i in idxs]))

# ---------------------
# Processing
# ---------------------
for subset in SETS:
    in_dir = os.path.join(ROOT_DIR, subset)
    if not os.path.isdir(in_dir):
        print(f"[WARN] Missing folder: {in_dir}")
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

        # PSD + bandpowers
        f, psd_mean, psd_sem = compute_psd_per_channel(X, fs)
        bp = compute_bandpowers(f, psd_mean)  # dict band -> (C,)

        # Save bandpowers
        pd.DataFrame(bp, index=ch_names).to_csv(
            os.path.join(out_dir_subset, make_filename_safe(os.path.splitext(fname)[0]) + "_bandpowers.csv")
        )

        # Indices per channel
        eps = 1e-12
        T = bp["Theta"]; A = bp["Alpha"]; B = bp["Beta"]
        TBR = T / (B + eps)
        TABR = (T + A) / (B + eps)
        ABR = A / (B + eps)
        TA  = T / (A + eps)
        idx_df = pd.DataFrame({
            "TBR (Theta/Beta)": TBR,
            "TABR ((Theta+Alpha)/Beta)": TABR,
            "ABR (Alpha/Beta)": ABR,
            "Theta/Alpha": TA,
        }, index=ch_names)
        idx_df.to_csv(os.path.join(out_dir_subset, make_filename_safe(os.path.splitext(fname)[0]) + "_indices_per_channel.csv"))

        # Region indices for workload θf/αp
        FRONTAL_SET = {"AF3", "AF4", "F3", "F4"}
        PARIETAL_OCC_SET = {"P7", "P8", "O1", "O2"}
        frontal_idx = select_indices(ch_names, FRONTAL_SET)
        po_idx = select_indices(ch_names, PARIETAL_OCC_SET)
        theta_frontal_mean = safe_mean(T, frontal_idx)
        alpha_po_mean      = safe_mean(A, po_idx)
        workload_ratio = theta_frontal_mean / (alpha_po_mean + eps)

        # Save per-file summary
        summary = {
            "file": fname,
            "fs": fs,
            "n_epochs": X.shape[0],
            "n_samples_per_epoch": X.shape[1],
            "n_channels": X.shape[2],
            "Workload (thetaF/alphaPO)": workload_ratio,
            "Global Theta/Alpha": float(np.mean(T) / (np.mean(A) + eps)),
            "Global TBR": float(np.mean(T) / (np.mean(B) + eps)),
            "Global TABR": float((np.mean(T) + np.mean(A)) / (np.mean(B) + eps)),
            "Global ABR": float(np.mean(A) / (np.mean(B) + eps)),
        }
        pd.DataFrame([summary]).to_csv(
            os.path.join(out_dir_subset, make_filename_safe(os.path.splitext(fname)[0]) + "_indices_summary.csv"),
            index=False
        )

        # PLOTS (grouped-by-region per channel)
        base = os.path.join(out_dir_subset, make_filename_safe(os.path.splitext(fname)[0]))

        # Reference PSD (mean ± SEM; not grouped by channel because it's frequency-domain mean across channels)
        ch_mean = psd_mean.mean(axis=0); ch_sem = psd_sem.mean(axis=0)
        upper = ch_mean + ch_sem; lower = ch_mean - ch_sem
        plt.figure(figsize=(8, 5))
        plt.plot(f, ch_mean, label="Mean PSD (across channels)")
        plt.fill_between(f, lower, upper, alpha=0.3)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD"); plt.title(f"{subset.capitalize()} | {fname} | Mean PSD")
        plt.tight_layout(); plt.savefig(base + "_PSD_mean.png", dpi=200); plt.close()

        # Grouped-by-region channel plots
        grouped_bar_plot(ch_names, TABR, "TABR ((Theta+Alpha)/Beta)",
                         f"{subset.capitalize()} | {fname} | Mental Fatigue (TABR)", base + "_MF_TABR.png")
        grouped_bar_plot(ch_names, TBR, "TBR (Theta/Beta)",
                         f"{subset.capitalize()} | {fname} | TBR", base + "_TBR.png")
        grouped_bar_plot(ch_names, ABR, "ABR (Alpha/Beta)",
                         f"{subset.capitalize()} | {fname} | ABR", base + "_ABR.png")
        grouped_bar_plot(ch_names, TA, "Theta/Alpha",
                         f"{subset.capitalize()} | {fname} | Theta/Alpha", base + "_ThetaAlpha.png")
        grouped_bar_plot(ch_names, A, "Alpha Power (integrated PSD)",
                         f"{subset.capitalize()} | {fname} | Alpha power (context)", base + "_Alpha.png")

        # Workload scalar (θf/αp)
        scalar_bar(workload_ratio, "θf/αp",
                   f"{subset.capitalize()} | {fname} | Workload (Frontal θ / ParOcc α)",
                   base + "_Workload_thetaF_over_alphaPO.png")

print(f"Done. Outputs saved under: {OUT_DIR}")
