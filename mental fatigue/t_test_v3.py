# %%
# Four-group TBR plots with *pairwise median-based scaling* for synthetic groups only.
# - SN is scaled to match RN median (display only).
# - SW is scaled to match RW median (display only).
# - Stats (t-tests) are computed on original, unscaled values.
#
# Outputs: tbr_channel_bars_4groups_scaled_median/posture{k}_Real_Synth_4bars_scaled_median.png
#
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ttest_ind

ROOT_DIR = "all npz"
OUT_DIR  = "tbr_channel_bars_4groups_scaled_median_v2"
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

def sig_stars(p):
    if np.isnan(p): return ""
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""

def median_scale(target_vals, source_vals):
    """Return scale so that median(source) * scale ~= median(target)."""
    t = np.nanmedian(target_vals)
    s = np.nanmedian(source_vals)
    if s <= 0 or not np.isfinite(s) or not np.isfinite(t) or t <= 0:
        return 1.0
    return float(t / s)

def plot_4group_scaled_median(ch_names, RN, RW, SN, SW, posture, out_path):
    # Compute stats
    groups = {"RN": RN, "RW": RW, "SN": SN, "SW": SW}
    means = {}; stds = {}
    for k, arr in groups.items():
        if arr is None:
            means[k] = np.full(len(ch_names), np.nan)
            stds[k]  = np.full(len(ch_names), np.nan)
        else:
            means[k] = np.nanmean(arr, axis=0)
            stds[k]  = np.nanstd(arr, axis=0, ddof=1)

    # per-channel p-values for exo effect within each domain
    p_R = np.full(len(ch_names), np.nan)
    p_S = np.full(len(ch_names), np.nan)
    for ci in range(len(ch_names)):
        if (RN is not None) and (RW is not None):
            try: p_R[ci] = ttest_ind(RN[:, ci], RW[:, ci], equal_var=False, nan_policy="omit")[1]
            except Exception: p_R[ci] = np.nan
        if (SN is not None) and (SW is not None):
            try: p_S[ci] = ttest_ind(SN[:, ci], SW[:, ci], equal_var=False, nan_policy="omit")[1]
            except Exception: p_S[ci] = np.nan

    # Compute median-based scaling separately for SN and SW (display only)
    scale_SN = 1.0
    scale_SW = 1.0
    if (RN is not None) and (SN is not None):
        scale_SN = median_scale(RN.ravel(), SN.ravel())
    if (RW is not None) and (SW is not None):
        scale_SW = median_scale(RW.ravel(), SW.ravel())

    # Reorder by region
    ordered, means["RN"] = reorder_by_region(ch_names, means["RN"])
    _, means["RW"] = reorder_by_region(ch_names, means["RW"])
    _, means["SN"] = reorder_by_region(ch_names, means["SN"])
    _, means["SW"] = reorder_by_region(ch_names, means["SW"])
    _, stds["RN"] = reorder_by_region(ch_names, stds["RN"])
    _, stds["RW"] = reorder_by_region(ch_names, stds["RW"])
    _, stds["SN"] = reorder_by_region(ch_names, stds["SN"])
    _, stds["SW"] = reorder_by_region(ch_names, stds["SW"])
    _, p_R = reorder_by_region(ch_names, p_R)
    _, p_S = reorder_by_region(ch_names, p_S)

    # Apply pairwise scaling to synthetic *display only*
    disp_means_SN = means["SN"] * scale_SN
    disp_means_SW = means["SW"] * scale_SW
    disp_stds_SN  = stds["SN"] * scale_SN
    disp_stds_SW  = stds["SW"] * scale_SW

    x = np.arange(len(ordered))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5*width, means["RN"], width, yerr=stds["RN"], capsize=3, label="Real No Exo")
    ax.bar(x - 0.5*width, means["RW"], width, yerr=stds["RW"], capsize=3, label="Real With Exo")
    ax.bar(x + 0.5*width, disp_means_SN, width, yerr=disp_stds_SN, capsize=3,
           label=("Synthetic No Exo" if abs(scale_SN-1.0)<1e-3 else f"Synth No Exo"))
    ax.bar(x + 1.5*width, disp_means_SW, width, yerr=disp_stds_SW, capsize=3,
           label=("Synthetic With Exo" if abs(scale_SW-1.0)<1e-3 else f"Synth With Exo"))

    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylabel("TBR (Theta/Beta)")
    ax.set_title(f"Posture {posture}")

    # significance stars above pairs
    y_max_R = np.maximum(np.nan_to_num(means["RN"] + stds["RN"]),
                         np.nan_to_num(means["RW"] + stds["RW"]))
    y_max_S = np.maximum(np.nan_to_num(disp_means_SN + disp_stds_SN),
                         np.nan_to_num(disp_means_SW + disp_stds_SW))
    for i in range(len(x)):
        sR = sig_stars(p_R[i])
        if sR: ax.text(x[i] - width, y_max_R[i]*1.05, sR, ha="center", va="bottom", fontsize=10)
        sS = sig_stars(p_S[i])
        if sS: ax.text(x[i] + width, y_max_S[i]*1.05, sS, ha="center", va="bottom", fontsize=10)

    # annotation
    ann = []
    if abs(scale_SN-1.0) > 1e-3: ann.append(f"SN ×{round(scale_SN,2)}")
    if abs(scale_SW-1.0) > 1e-3: ann.append(f"SW ×{round(scale_SW,2)}")
    # if ann:
    #     ax.text(0.99, 0.02, "Scaled (display): " + ", ".join(ann) + "\nStats from raw values",
    #             transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# Build per-posture plots
def load_group(paths):
    if not paths: return None, None
    mats = []; ch_ref = None
    for p in paths:
        m, ch = compute_tbr_matrix(p); mats.append(m)
        if ch_ref is None: ch_ref = ch
    return np.vstack(mats), ch_ref

buckets = scan_files(ROOT_DIR)
for posture in sorted(buckets.keys()):
    g = buckets[posture]
    RN, ch_ref = load_group(g["RN"])
    RW, _ = load_group(g["RW"])
    SN, _ = load_group(g["SN"])
    SW, _ = load_group(g["SW"])
    if ch_ref is None:
        print(f"[WARN] No data for posture {posture}, skipping."); continue
    out_png = os.path.join(OUT_DIR, f"posture{posture}_Real_Synth_4bars_scaled_median.png")
    plot_4group_scaled_median(ch_ref, RN, RW, SN, SW, posture, out_png)

print(f"Done. Median-based scaled plots saved to: {OUT_DIR}")
