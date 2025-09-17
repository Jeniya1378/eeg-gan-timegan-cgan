#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mental_fatigue_real_vs_fake.py

End-to-end:
- Parse real trial CSVs across participants -> per-trial channelwise TBR.
- Average per participant per posture×condition -> group mean±SD.
- Load TimeGAN synthetic .npz -> channelwise TBR per sequence.
- Downsample synthetic to match real sample counts per posture×condition.
- Plot per posture: 4 bars per channel (Real No/With, Fake No/With) + paired t-tests.

If your pipeline helpers exist (apply_filters, epoch_data, compute_fatigue, CHANNELS_14, FS, etc.)
they will be used; otherwise safe fallbacks are applied.

Author: you + me :)
"""

import os, re, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# ------------------------------
# Try to import your EEG helpers; else provide fallbacks
# ------------------------------
CHANNELS_14_DEFAULT = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
try:
    # Edit this import to your actual helpers if needed:
    # from eeg_helpers import CHANNELS_14, FS, EPOCH_LEN_S, FATIGUE_METRIC, apply_filters, epoch_data, compute_fatigue
    raise ImportError  # remove this line if you have a real helpers module to import
except Exception:
    CHANNELS_14 = CHANNELS_14_DEFAULT
    FS = 128               # Hz
    EPOCH_LEN_S = 6.0
    FATIGUE_METRIC_DEFAULT = "TBR"  # Theta/Beta ratio

    def apply_filters(X, fs):
        # keep as-is; your real pipeline may bandpass etc.
        return X

    def epoch_data(X, fs, epoch_len_s):
        # treat entire trial as one epoch by default
        return [X]

    def compute_fatigue(epoch, fs, metric):
        # Fallback: simple FFT-PSD TBR per channel
        x = epoch
        T, C = x.shape
        if T < 4:  # too short
            return np.full(C, np.nan)
        w = np.hanning(T)[:, None]
        Xw = np.fft.rfft((x - x.mean(axis=0)) * w, axis=0)
        psd = (np.abs(Xw)**2) / np.sum(w**2)
        freqs = np.fft.rfftfreq(T, d=1.0/max(1, fs))
        def band_power(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            if not np.any(m):
                return np.ones(C)*1e-8
            return psd[m, :].mean(axis=0) + 1e-8
        theta = band_power(4.0, 8.0)
        beta  = band_power(13.0, 30.0)
        return theta / beta

# ------------------------------
# Robust filename parsing
# ------------------------------
COND_RE  = re.compile(r"(no[ _-]*exo|with[ _-]*exo)", re.IGNORECASE)
POST_RE  = re.compile(r"posture[ _-]?(\d+)", re.IGNORECASE)
TRIAL_RE = re.compile(r"trial[ _-]?(\d+)", re.IGNORECASE)
REST_RE  = re.compile(r"rest", re.IGNORECASE)

def parse_cond(text: str):
    m = COND_RE.search(text)
    if not m: return None
    s = m.group(1).lower().replace("_"," ").replace("-"," ").strip()
    if "no exo" in s:   return "no_exo"
    if "with exo" in s: return "with_exo"
    return None

def parse_posture(text: str):
    m = POST_RE.search(text)
    return int(m.group(1)) if m else None

def parse_trial(text: str):
    m = TRIAL_RE.search(text)
    return int(m.group(1)) if m else None

# ------------------------------
# Real data: per-trial fatigue vector (14,)
# ------------------------------
def fatigue_vector_from_trial_csv(fpath: str):
    df = pd.read_csv(fpath)

    # map 'eeg.af3'/'EEG.AF3'/'AF3' -> 'AF3'
    rename = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("eeg."):
            ch = cl.split("eeg.",1)[1].upper()
            rename[c] = ch
        elif c.upper() in CHANNELS_14:
            rename[c] = c.upper()
    if rename:
        df = df.rename(columns=rename)

    ch_cols = [c for c in CHANNELS_14 if c in df.columns]
    if not ch_cols:
        return None

    X = df[ch_cols].to_numpy(dtype=float)
    Xf = apply_filters(X, FS)
    epochs = epoch_data(Xf, FS, EPOCH_LEN_S)
    if not epochs:
        return None

    vals = []
    for ep in epochs:
        v = compute_fatigue(ep, FS, FATIGUE_METRIC_DEFAULT)
        vals.append(v)
    ch_mean = np.nanmean(np.vstack(vals), axis=0)  # align to ch_cols

    out = np.full(len(CHANNELS_14), np.nan)
    for i, ch in enumerate(CHANNELS_14):
        if ch in ch_cols:
            out[i] = ch_mean[ch_cols.index(ch)]
    return out

# ------------------------------
# Collect & aggregate REAL across participants
# ------------------------------
def collect_real(data_root: Path):
    """
    Returns:
      per_part[pid][posture]['no_exo'|'with_exo'] -> list of (14,) arrays
    """
    per_part = {}
    parts = sorted([p for p in glob.glob(str(data_root / "protocol_outputs_*")) if os.path.isdir(p)])
    if not parts:
        print("[WARN] No participant folders found under", data_root)
    for pdir in parts:
        pid = os.path.basename(pdir)
        per_part.setdefault(pid, {i: {"no_exo": [], "with_exo": []} for i in range(1,10)})

        csvs = glob.glob(os.path.join(pdir, "**", "*.csv"), recursive=True)
        for f in csvs:
            base = os.path.basename(f)
            if REST_RE.search(base):  # skip rest files
                continue
            # parse from the full path (some info might be in folders)
            text = f.lower()
            posture = parse_posture(text)
            cond    = parse_cond(text)
            tix     = parse_trial(text)  # optional
            if posture is None or cond is None:
                continue
            vec = fatigue_vector_from_trial_csv(f)
            if vec is None:
                continue
            per_part[pid][posture][cond].append(vec)
    return per_part

def aggregate_real(per_part):
    """
    Aggregate across participants.

    Returns:
      means[p][cond], sds[p][cond], pvals[p], counts[p] -> dict with counts['no_exo'], counts['with_exo'], counts['paired']
    """
    means = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    sds   = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    pvals = {p: np.full(len(CHANNELS_14), np.nan) for p in range(1,10)}
    counts = {p: {"no_exo":0, "with_exo":0, "paired":0} for p in range(1,10)}

    for p in range(1,10):
        per_part_no, per_part_we = {}, {}
        for pid, pdata in per_part.items():
            tr_no = pdata[p]["no_exo"]; tr_we = pdata[p]["with_exo"]
            if tr_no:
                per_part_no[pid]  = np.nanmean(np.vstack(tr_no), axis=0)
            if tr_we:
                per_part_we[pid]  = np.nanmean(np.vstack(tr_we), axis=0)

        counts[p]["no_exo"]  = len(per_part_no)
        counts[p]["with_exo"] = len(per_part_we)
        common = sorted(set(per_part_no.keys()) & set(per_part_we.keys()))
        counts[p]["paired"]  = len(common)

        if per_part_no:
            A = np.vstack(list(per_part_no.values()))
            means[p]["no_exo"] = np.nanmean(A, axis=0)
            sds[p]["no_exo"]   = np.nanstd(A, axis=0, ddof=0)
        if per_part_we:
            B = np.vstack(list(per_part_we.values()))
            means[p]["with_exo"] = np.nanmean(B, axis=0)
            sds[p]["with_exo"]   = np.nanstd(B, axis=0, ddof=0)

        if common:
            A_pair = np.vstack([per_part_no[pid] for pid in common])
            B_pair = np.vstack([per_part_we[pid] for pid in common])
            for ci in range(len(CHANNELS_14)):
                a = A_pair[:, ci]; b = B_pair[:, ci]
                m = ~np.isnan(a) & ~np.isnan(b)
                if m.sum() >= 2:
                    _, pv = ttest_rel(a[m], b[m], nan_policy="omit")
                    pvals[p][ci] = pv

    return means, sds, pvals, counts

# ------------------------------
# Synthetic: load and aggregate
# ------------------------------
def find_synth_npz(run_dir: Path):
    for name in ["synthetic_long.npz", "synthetic.npz"]:
        fp = run_dir / name
        if fp.exists(): return fp
    cands = sorted(run_dir.glob("*.npz"))
    return cands[0] if cands else None

def fatigue_from_synth_npz(npz_path: Path):
    if not npz_path or not npz_path.exists():
        return None
    X = np.load(npz_path)["X"].astype(np.float32)  # (N,T,C)
    N, T, C = X.shape
    out = np.full((N, len(CHANNELS_14)), np.nan)
    for i in range(N):
        seq = X[i]
        seq_f = apply_filters(seq, FS)
        epochs = epoch_data(seq_f, FS, EPOCH_LEN_S)
        if not epochs:
            continue
        vals = []
        for ep in epochs:
            vals.append(compute_fatigue(ep, FS, FATIGUE_METRIC_DEFAULT))
        out[i, :] = np.nanmean(np.vstack(vals), axis=0)
    return out  # (N,14) possibly with NaNs

def aggregate_fake(runs_dir: Path, real_counts):
    """
    Downsample synthetic counts to match *real participant counts* per posture×condition.
    Returns means[p][cond], sds[p][cond], pvals[p].
    """
    means = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    sds   = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    pvals = {p: np.full(len(CHANNELS_14), np.nan) for p in range(1,10)}

    rng = np.random.RandomState(0)

    for p in range(1,10):
        npz_no = find_synth_npz(runs_dir / f"posture{p}_no_exo")
        npz_we = find_synth_npz(runs_dir / f"posture{p}_with_exo")
        A = fatigue_from_synth_npz(npz_no) if npz_no else None
        B = fatigue_from_synth_npz(npz_we) if npz_we else None
        if A is None or B is None:
            continue

        # match to real participant counts per condition
        n_real_no  = max(0, real_counts[p]["no_exo"])
        n_real_we  = max(0, real_counts[p]["with_exo"])
        if n_real_no == 0 and n_real_we == 0:
            continue

        # sample without replacement to match counts (or as many as possible)
        def take(M, k):
            M = M[~np.isnan(M).all(axis=1)]
            if len(M) == 0: return M
            if k <= 0: return np.empty((0, M.shape[1]))
            if len(M) <= k: return M
            idx = rng.choice(len(M), size=k, replace=False)
            return M[idx]

        A_use = take(A, n_real_no)
        B_use = take(B, n_real_we)

        # Means & SDs for each condition (independent)
        if len(A_use):
            means[p]["no_exo"] = np.nanmean(A_use, axis=0)
            sds[p]["no_exo"]   = np.nanstd(A_use, axis=0, ddof=0)
        if len(B_use):
            means[p]["with_exo"] = np.nanmean(B_use, axis=0)
            sds[p]["with_exo"]   = np.nanstd(B_use, axis=0, ddof=0)

        # Paired t-test (No vs With) on intersection (min of the two sample sizes)
        n_pair = min(len(A_use), len(B_use))
        if n_pair >= 2:
            A_pair = A_use[:n_pair]
            B_pair = B_use[:n_pair]
            for ci in range(len(CHANNELS_14)):
                a = A_pair[:, ci]; b = B_pair[:, ci]
                m = ~np.isnan(a) & ~np.isnan(b)
                if m.sum() >= 2:
                    _, pv = ttest_rel(a[m], b[m], nan_policy="omit")
                    pvals[p][ci] = pv

    return means, sds, pvals

# ------------------------------
# Plot
# ------------------------------
def plot_posture(p, real_means, real_sds, real_pvals,
                 fake_means, fake_sds, fake_pvals, out_dir: Path,
                 metric_label="TBR"):
    chs = CHANNELS_14
    x = np.arange(len(chs))
    bw = 0.2

    rm_no  = real_means[p]["no_exo"];  rs_no = real_sds[p]["no_exo"]
    rm_we  = real_means[p]["with_exo"];rs_we = real_sds[p]["with_exo"]
    fm_no  = fake_means[p]["no_exo"];  fs_no = fake_sds[p]["no_exo"]
    fm_we  = fake_means[p]["with_exo"];fs_we = fake_sds[p]["with_exo"]

    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    ax.bar(x - 1.5*bw, rm_no, bw, yerr=rs_no, capsize=3, label="Real No Exo")
    ax.bar(x - 0.5*bw, rm_we, bw, yerr=rs_we, capsize=3, label="Real With Exo")
    ax.bar(x + 0.5*bw, fm_no, bw, yerr=fs_no, capsize=3, label="Fake No Exo")
    ax.bar(x + 1.5*bw, fm_we, bw, yerr=fs_we, capsize=3, label="Fake With Exo")

    ax.set_xticks(x); ax.set_xticklabels(chs, rotation=0)
    ax.set_ylabel(f"{metric_label} (mean ± SD)")
    ax.set_title(f"Posture {p}: Mental Fatigue by Channel — Real vs Fake, No-Exo vs With-Exo")
    ax.legend(ncols=2, frameon=False, loc="upper right")

    # annotate significance (Real, then Fake)
    ymax = 0.0
    stacks = [rm_no, rm_we, fm_no, fm_we]
    errs   = [rs_no, rs_we, fs_no, fs_we]
    for ci in range(len(chs)):
        vals = []
        for k in range(4):
            v = stacks[k][ci]; e = errs[k][ci]
            if not np.isnan(v):
                vals.append(v + (0 if np.isnan(e) else e))
        if vals:
            gmax = max(vals); ymax = max(ymax, gmax)
        # Real
        pv_r = real_pvals[p][ci]
        if pv_r==pv_r and pv_r < 0.05:
            y = (gmax if vals else 0.1) * 1.06 + 0.02
            ax.plot([ci-1.5*bw, ci-0.5*bw], [y, y], color="black", lw=1)
            ax.text(ci - bw, y + 0.005*max(1,y), "*", ha="center", va="bottom", fontsize=12)
        # Fake
        pv_f = fake_pvals[p][ci]
        if pv_f==pv_f and pv_f < 0.05:
            y2 = (gmax if vals else 0.1) * 1.16 + 0.02
            ax.plot([ci+0.5*bw, ci+1.5*bw], [y2, y2], color="black", lw=1)
            ax.text(ci + bw, y2 + 0.005*max(1,y2), "*", ha="center", va="bottom", fontsize=12)

    if ymax == 0.0: ymax = 1.0
    ax.set_ylim(0, ymax*1.35)
    fig.tight_layout()
    out = out_dir / f"posture{p}_fatigue_real_vs_fake.png"
    fig.savefig(out, dpi=220); plt.close(fig)
    print("[SAVED]", out)

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default="./6s_window", help="Folder containing protocol_outputs_*")
    parser.add_argument("--runs_dir",  type=str, default="./timegan_runs", help="Folder with postureX_cond synthetic")
    parser.add_argument("--out_dir",   type=str, default="./fatigue_plots", help="Output folder")
    parser.add_argument("--metric",    type=str, default="TBR", help="Label for y-axis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("== Collecting REAL trials ...")
    per_part = collect_real(Path(args.data_root))
    real_means, real_sds, real_pvals, real_counts = aggregate_real(per_part)

    # Quick diagnostics so you see why bars may be missing
    print("\n[Real participant counts per posture]")
    for p in range(1,10):
        c = real_counts[p]
        print(f"  posture {p}: no_exo={c['no_exo']}, with_exo={c['with_exo']}, paired={c['paired']}")

    print("\n== Loading SYNTHETIC and matching counts ...")
    fake_means, fake_sds, fake_pvals = aggregate_fake(Path(args.runs_dir), real_counts)

    print("\n== Plotting ...")
    for p in range(1,10):
        plot_posture(p, real_means, real_sds, real_pvals,
                     fake_means, fake_sds, fake_pvals, out_dir,
                     metric_label=args.metric)
    print("\nDone.")

if __name__ == "__main__":
    np.random.seed(0)
    main()
