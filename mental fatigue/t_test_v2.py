# %%
# Update: scale synthetic bars for visibility (optional) without altering stats.
# If synthetic values are much larger than real, we scale only the DISPLAY of
# synthetic groups by a factor so that they're comparable on the same axis.
# We annotate the figure with the applied scale.
#
# Re-runs the "4 bars per channel" generator with visibility scaling.
#
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import ttest_ind

ROOT_DIR = "all npz"
OUT_DIR  = "tbr_channel_bars_4groups_scaled"
os.makedirs(OUT_DIR, exist_ok=True)

# Welch params
N_PER_SEG = None
N_OVERLAP = None
NFFT = None

# Bands
THETA = (4.0, 8.0)
BETA  = (13.0, 30.0)

# Channel order
REGION_ORDER = {
    "Frontal":   ["AF3","AF4","F3","F4","FC5","FC6","F7","F8"],
    "Temporal":  ["T7","T8"],
    "Parietal":  ["P7","P8"],
    "Occipital": ["O1","O2"],
}
ORDERED_CH = sum(REGION_ORDER.values(), [])

# Visibility scaling hyperparams
REAL_REF_QUANT = 0.95     # compare 95th percentile
SYN_REF_QUANT  = 0.95
ALLOWANCE      = 1.5      # allow synthetic to be up to 1.5x real before scaling
MIN_SCALE      = 0.05     # don't scale by less than 5% (avoid zero)
ROUND_TO       = 0.01     # show scale rounded

def load_npz_safe(path):
    npz = np.load(path, allow_pickle=True)
    if "X" not in npz: raise ValueError("No EEG array 'X' in file")
    X = npz["X"]
    if "fs" in npz:
        fs = npz["fs"]; fs = fs.item() if isinstance(fs, np.ndarray) else fs
        fs = float(fs)
    else:
        fs = 128.0
    if "ch_names" in npz:
        ch = npz["ch_names"]
        try:
            ch_names = ch.tolist() if hasattr(ch, "tolist") else list(ch)
            ch_names = [str(x) for x in ch_names]
        except Exception:
            C = X.shape[-1]; ch_names = [f"Ch{idx+1}" for idx in range(C)]
    else:
        C = X.shape[-1]; ch_names = [f"Ch{idx+1}" for idx in range(C)]
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
        if ch in ch_names: order.append(ch_names.index(ch))
    for i, ch in enumerate(ch_names):
        if i not in order: order.append(i)
    if arr_values.ndim == 1:
        return [ch_names[i] for i in order], arr_values[order]
    else:
        return [ch_names[i] for i in order], arr_values[order, ...]

def compute_visibility_scale(RN, RW, SN, SW):
    # Use percentiles over all values (epochs × channels) to get robust scale
    real_vals = []
    synth_vals = []
    for arr in [RN, RW]:
        if arr is not None: real_vals.append(arr.ravel())
    for arr in [SN, SW]:
        if arr is not None: synth_vals.append(arr.ravel())
    if not real_vals or not synth_vals: return 1.0  # nothing to scale
    real_vals = np.concatenate(real_vals)
    synth_vals = np.concatenate(synth_vals)
    real_ref  = np.nanpercentile(real_vals, REAL_REF_QUANT * 100.0)
    synth_ref = np.nanpercentile(synth_vals, SYN_REF_QUANT * 100.0)
    if synth_ref <= 0 or real_ref <= 0: return 1.0
    # Allow some difference; if beyond allowance, scale down synthetic
    if synth_ref > real_ref * ALLOWANCE:
        s = max(MIN_SCALE, (real_ref * ALLOWANCE) / synth_ref)
        return float(s)
    return 1.0

def plot_4group_scaled(ch_names, RN, RW, SN, SW, posture, out_path):
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

    # p-values for real and synthetic exo effects
    p_R = np.full(len(ch_names), np.nan)
    p_S = np.full(len(ch_names), np.nan)
    for ci in range(len(ch_names)):
        if (RN is not None) and (RW is not None):
            try: p_R[ci] = ttest_ind(RN[:, ci], RW[:, ci], equal_var=False, nan_policy="omit")[1]
            except Exception: p_R[ci] = np.nan
        if (SN is not None) and (SW is not None):
            try: p_S[ci] = ttest_ind(SN[:, ci], SW[:, ci], equal_var=False, nan_policy="omit")[1]
            except Exception: p_S[ci] = np.nan

    # Compute visibility scale for synthetic
    scale = compute_visibility_scale(RN, RW, SN, SW)

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

    # Apply scale to *display only* for synthetic
    disp_means_SN = means["SN"] * scale
    disp_means_SW = means["SW"] * scale
    disp_stds_SN  = stds["SN"] * scale
    disp_stds_SW  = stds["SW"] * scale

    def sig_stars(p):
        if np.isnan(p): return ""
        if p < 1e-3: return "***"
        if p < 1e-2: return "**"
        if p < 5e-2: return "*"
        return ""

    x = np.arange(len(ordered))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 1.5*width, means["RN"], width, yerr=stds["RN"], capsize=3, label="Real No Exo")
    ax.bar(x - 0.5*width, means["RW"], width, yerr=stds["RW"], capsize=3, label="Real With Exo")
    ax.bar(x + 0.5*width, disp_means_SN, width, yerr=disp_stds_SN, capsize=3,
           label=("Synthetic No Exo" if scale==1.0 else f"Synth No Exo (×{round(scale,2)})"))
    ax.bar(x + 1.5*width, disp_means_SW, width, yerr=disp_stds_SW, capsize=3,
           label=("Synthetic With Exo" if scale==1.0 else f"Synth With Exo (×{round(scale,2)})"))

    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylabel("TBR (Theta/Beta)")
    ax.set_title(f"Posture {posture} | Real & Synthetic | 4 groups per channel")

    # stars above pairs (real and synthetic)
    y_max_R = np.maximum(np.nan_to_num(means["RN"] + stds["RN"]),
                         np.nan_to_num(means["RW"] + stds["RW"]))
    y_max_S = np.maximum(np.nan_to_num(disp_means_SN + disp_stds_SN),
                         np.nan_to_num(disp_means_SW + disp_stds_SW))
    for i in range(len(x)):
        sR = sig_stars(p_R[i])
        if sR: ax.text(x[i] - width, y_max_R[i]*1.05, sR, ha="center", va="bottom", fontsize=10)
        sS = sig_stars(p_S[i])
        if sS: ax.text(x[i] + width, y_max_S[i]*1.05, sS, ha="center", va="bottom", fontsize=10)

    # annotation about scaling
    if scale != 1.0:
        ax.text(0.99, 0.02,
                f"Synthetic bars scaled by ×{round(scale/1.0,2)} for visibility.\n"
                f"Stats computed on original values.",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# Build per-posture scaled plots
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
    out_png = os.path.join(OUT_DIR, f"posture{posture}_Real_Synth_4bars_scaled.png")
    plot_4group_scaled(ch_ref, RN, RW, SN, SW, posture, out_png)

print(f"Done. Scaled plots saved to: {OUT_DIR}")
