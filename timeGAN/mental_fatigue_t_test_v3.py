# mental_fatigue_tbr_real_fake.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from pathlib import Path

# ===== USER PATHS =====
BASE_DIR = Path(r"C:/Users/jsultana2/All Code/EEG/timeGAN new v2/all npz")
REAL_DIR = BASE_DIR / "real"
FAKE_DIR = BASE_DIR / "synthetic"
OUT_DIR  = BASE_DIR / "plots_tbr"
OUT_DIR.mkdir(exist_ok=True)

CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1",
            "O2","P8","T8","FC6","F4","F8","AF4"]

# ===== TBR function =====
def compute_tbr(x, fs=128):
    """Compute Theta/Beta Ratio for each channel in (T,C) data"""
    T, C = x.shape
    w = np.hanning(T)[:,None]
    Xw = np.fft.rfft((x-x.mean(0))*w,axis=0)
    psd = (np.abs(Xw)**2)/np.sum(w**2)
    freqs = np.fft.rfftfreq(T, d=1/fs)
    th = psd[(freqs>=4)&(freqs<8)].mean(0)+1e-8
    be = psd[(freqs>=13)&(freqs<30)].mean(0)+1e-8
    return th/be

# ===== Loader =====
def load_npz(fname):
    if not fname.exists():
        return None
    d = np.load(fname, allow_pickle=True)
    X = d["X"]  # (N,T,C)
    if "scale_min" in d and "scale_range" in d:
        smin, srng = d["scale_min"], d["scale_range"]
        X = X * srng.reshape(1,1,-1) + smin.reshape(1,1,-1)  # inverse scaling
    return X

# ===== Main =====
for posture in range(1,10):
    results = {}
    for cond in ["no_exo","with_exo"]:
        # real
        fr = REAL_DIR / f"posture{posture}_{cond}.npz"
        Xr = load_npz(fr)
        if Xr is not None:
            results[f"real_{cond}"] = np.array([compute_tbr(x) for x in Xr])
        # fake
        ff = FAKE_DIR / f"posture{posture}_{cond}.npz"
        Xf = load_npz(ff)
        if Xf is not None:
            results[f"fake_{cond}"] = np.array([compute_tbr(x) for x in Xf])

    if not results:
        print(f"[SKIP] posture {posture}: no data found")
        continue

    # compute mean±sd for available groups
    means, sds = {}, {}
    for k, arr in results.items():
        means[k] = np.nanmean(arr, axis=0)
        sds[k]   = np.nanstd(arr, axis=0, ddof=1)

    # paired t-tests (real vs fake, same cond)
    pvals = {}
    for cond in ["no_exo","with_exo"]:
        rk, fk = f"real_{cond}", f"fake_{cond}"
        if rk in results and fk in results:
            R, F = results[rk], results[fk]
            n = min(len(R), len(F))
            if n>1:
                stat,p = ttest_rel(R[:n],F[:n],axis=0,nan_policy="omit")
                pvals[cond] = p
            else:
                pvals[cond] = np.ones(len(CHANNELS))
        else:
            pvals[cond] = np.ones(len(CHANNELS))

    # plotting
    x = np.arange(len(CHANNELS))*3.0
    w = 0.6
    fig, ax = plt.subplots(figsize=(16,6))

    colors = {"real_no_exo":"#1f77b4","real_with_exo":"#ff7f0e",
              "fake_no_exo":"#2ca02c","fake_with_exo":"#d62728"}

    shift = {"real_no_exo":-1.5,"real_with_exo":-0.5,
             "fake_no_exo":+0.5,"fake_with_exo":+1.5}

    for key in ["real_no_exo","real_with_exo","fake_no_exo","fake_with_exo"]:
        if key in means:
            ax.bar(x+shift[key], means[key], yerr=sds[key],
                   width=w, capsize=3, label=key.replace("_"," ").title(),
                   color=colors[key])

    # significance stars
    for cond in ["no_exo","with_exo"]:
        if cond in pvals:
            for ci,p in enumerate(pvals[cond]):
                if p<0.05:
                    y = max([means.get(f"real_{cond}",np.zeros(len(CHANNELS)))[ci],
                             means.get(f"fake_{cond}",np.zeros(len(CHANNELS)))[ci]])
                    ax.text(x[ci]+(0 if cond=="no_exo" else 0.5),
                            y*1.1,"*",ha="center",fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS, rotation=0)
    ax.set_ylabel("TBR (Theta/Beta Ratio, mean ± SD)")
    ax.set_title(f"Posture {posture}: TBR — Real vs Fake, No-Exo vs With-Exo")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR/f"posture{posture}_tbr_real_vs_fake.png"
    plt.savefig(out,dpi=200)
    plt.close()
    print(f"[SAVED] {out}")
