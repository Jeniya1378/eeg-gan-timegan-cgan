# %%
# Add CSV export of t-test results for the median-scaled 4-group TBR plots.
# For each posture k, this writes:
#   tbr_channel_bars_4groups_scaled_median/posture{k}_ttest_results.csv
# Columns:
#   Channel, t_stat_real, p_val_real, t_stat_synth, p_val_synth, SN_scale, SW_scale
#
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

ROOT_DIR = "all npz"
OUT_DIR  = "tbr_channel_bars_4groups_scaled_median_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Welch params
N_PER_SEG = None
N_OVERLAP = None
NFFT = None

# Bands
THETA = (4.0, 8.0)
BETA  = (13.0, 30.0)

# Channel order (grouped by region for readability)
REGION_ORDER = {
    "Frontal":   ["AF3","AF4","F3","F4","FC5","FC6","F7","F8"],
    "Temporal":  ["T7","T8"],
    "Parietal":  ["P7","P8"],
    "Occipital": ["O1","O2"],
}
ORDERED_CH = sum(REGION_ORDER.values(), [])

def load_npz_safe(path):
    npz = np.load(path, allow_pickle=True)
    if "X" not in npz:
        raise ValueError("No EEG array 'X' in file")
    X = npz["X"]
    if "fs" in npz:
        fs = npz["fs"]
        if isinstance(fs, np.ndarray): fs = fs.item()
        fs = float(fs)
    else:
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

def welch_bandpower(x, fs, band):
    lo, hi = band
    nperseg = int(fs) if N_PER_SEG is None else N_PER_SEG
    noverlap = int(nperseg // 2) if N_OVERLAP is None else N_OVERLAP
    freqs, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=NFFT)
    idx = (freqs >= lo) & (freqs <= hi)
    if not np.any(idx): return 0.0
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(Pxx[idx], freqs[idx]))

def compute_tbr_matrix(fpath):
    X, fs, ch_names = load_npz_safe(fpath)  # X: (N,T,C)
    N, T, C = X.shape
    tbr = np.zeros((N, C), dtype=np.float64)
    for i in range(N):
        for c in range(C):
            theta = welch_bandpower(X[i, :, c], fs, THETA)
            beta  = welch_bandpower(X[i, :, c], fs, BETA)
            tbr[i, c] = theta / (beta + 1e-12)
    return tbr, ch_names

def scan_files(root):
    buckets = {}  # posture -> {RN:[], RW:[], SN:[], SW:[]}
    for subset, code in [("real","RN"), ("real","RW"), ("synthetic","SN"), ("synthetic","SW")]:
        base = os.path.join(root, subset)
        if not os.path.isdir(base): continue
        for fname in os.listdir(base):
            if not fname.lower().endswith(".npz"): continue
            low = fname.lower()
            m = re.search(r"posture\s*(\d+)", low)
            if not m: continue
            posture = int(m.group(1))
            if "no_exo" in low or "no-exo" in low or "noexo" in low:
                ok = code in ("RN","SN")
            elif "with_exo" in low or "with-exo" in low or "withexo" in low or "with" in low:
                ok = code in ("RW","SW")
            else:
                continue
            if not ok: continue
            buckets.setdefault(posture, {"RN":[], "RW":[], "SN":[], "SW":[]})
            buckets[posture][code].append(os.path.join(base, fname))
    return buckets

def reorder_by_region(ch_names, arr_values):
    order = []
    for ch in ORDERED_CH:
        if ch in ch_names:
            order.append(ch_names.index(ch))
    for i, ch in enumerate(ch_names):
        if i not in order: order.append(i)
    if arr_values.ndim == 1:
        return [ch_names[i] for i in order], arr_values[order]
    else:
        return [ch_names[i] for i in order], arr_values[order, ...]

def median_scale(target_vals, source_vals):
    t = np.nanmedian(target_vals)
    s = np.nanmedian(source_vals)
    if s <= 0 or not np.isfinite(s) or not np.isfinite(t) or t <= 0:
        return 1.0
    return float(t / s)

# Build per-posture CSVs
buckets = scan_files(ROOT_DIR)
for posture in sorted(buckets.keys()):
    g = buckets[posture]

    def load_group(paths):
        if not paths: return None, None
        mats = []; ch_ref = None
        for p in paths:
            m, ch = compute_tbr_matrix(p); mats.append(m)
            if ch_ref is None: ch_ref = ch
        return np.vstack(mats), ch_ref

    RN, ch_ref = load_group(g["RN"])
    RW, _ = load_group(g["RW"])
    SN, _ = load_group(g["SN"])
    SW, _ = load_group(g["SW"])

    if ch_ref is None:
        print(f"[WARN] No data for posture {posture}, skipping.")
        continue

    # per-channel Welch t-tests (raw, unscaled)
    t_stat_real = np.full(len(ch_ref), np.nan)
    p_val_real  = np.full(len(ch_ref), np.nan)
    t_stat_syn  = np.full(len(ch_ref), np.nan)
    p_val_syn   = np.full(len(ch_ref), np.nan)

    for ci in range(len(ch_ref)):
        if (RN is not None) and (RW is not None):
            try:
                ts, p = ttest_ind(RN[:, ci], RW[:, ci], equal_var=False, nan_policy="omit")
                t_stat_real[ci], p_val_real[ci] = ts, p
            except Exception:
                pass
        if (SN is not None) and (SW is not None):
            try:
                ts, p = ttest_ind(SN[:, ci], SW[:, ci], equal_var=False, nan_policy="omit")
                t_stat_syn[ci], p_val_syn[ci] = ts, p
            except Exception:
                pass

    # Compute median-based display scales (also write to CSV for traceability)
    SN_scale = 1.0 if (RN is None or SN is None) else median_scale(RN.ravel(), SN.ravel())
    SW_scale = 1.0 if (RW is None or SW is None) else median_scale(RW.ravel(), SW.ravel())

    # Reorder by region for a neat CSV
    ordered, t_stat_real = reorder_by_region(ch_ref, t_stat_real)
    _, p_val_real  = reorder_by_region(ch_ref, p_val_real)
    _, t_stat_syn  = reorder_by_region(ch_ref, t_stat_syn)
    _, p_val_syn   = reorder_by_region(ch_ref, p_val_syn)

    df = pd.DataFrame({
        "Channel": ordered,
        "t_stat_real (RN vs RW)": t_stat_real,
        "p_val_real (RN vs RW)": p_val_real,
        "t_stat_synth (SN vs SW)": t_stat_syn,
        "p_val_synth (SN vs SW)": p_val_syn,
    })

    # include global scales as columns for reference (same for all rows)
    df["SN_scale_display"] = SN_scale
    df["SW_scale_display"] = SW_scale

    out_csv = os.path.join(OUT_DIR, f"posture{posture}_ttest_results.csv")
    df.to_csv(out_csv, index=False)

print("Done. t-test CSVs written alongside scaled plots.")
