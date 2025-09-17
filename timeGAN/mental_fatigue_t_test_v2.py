#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mental_fatigue_real_vs_fake_full.py

- Real EEG: compute per-trial channel-wise TBR, average per participant per posture×condition,
  then compute group mean±SD and paired t-tests (No vs With over participants with both).
- Synthetic (TimeGAN): load per posture×condition, inverse-scale to real using preprocessed
  scalers (scale_min/scale_range) and ch_names, clamp to real range, remap to 14 channels,
  compute TBR, downsample to match real counts, paired t-tests (No vs With).

Outputs: one PNG per posture in --out_dir with 4 bars per channel (Real/Fake × No/With).
"""

import os, re, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# ------------------------------
# Globals / fallbacks (safe defaults)
# ------------------------------
CHANNELS_14 = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
FS = 128.0
EPOCH_LEN_S = 6.0
Y_LABEL = "TBR"  # Theta/Beta Ratio

# Optional: plug your filter/epoch functions here
def apply_filters(X, fs):  # no-op fallback
    return X

def epoch_data(X, fs, epoch_len_s):
    return [X]  # treat whole trial as one epoch

def compute_tbr(epoch, fs):
    """Theta/Beta via FFT-PSD, per channel."""
    x = epoch
    T, C = x.shape
    if T < 8:
        return np.full(C, np.nan)
    w = np.hanning(T)[:, None]
    Xw = np.fft.rfft((x - x.mean(axis=0)) * w, axis=0)
    psd = (np.abs(Xw)**2) / np.sum(w**2)
    freqs = np.fft.rfftfreq(T, d=1.0/max(1.0, fs))
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
REST_RE  = re.compile(r"rest", re.IGNORECASE)

def parse_cond(text: str):
    m = COND_RE.search(text)
    if not m: return None
    s = m.group(1).lower().replace("_"," ").replace("-"," ").strip()
    if "no exo" in s: return "no_exo"
    if "with exo" in s: return "with_exo"
    return None

def parse_posture(text: str):
    m = POST_RE.search(text)
    return int(m.group(1)) if m else None

# ------------------------------
# Channel header canonicalization
# ------------------------------
CH_PAT = re.compile(r"\b(AF3|F7|F3|FC5|T7|P7|O1|O2|P8|T8|FC6|F4|F8|AF4)\b", re.IGNORECASE)
def canonical_channel(colname: str):
    m = CH_PAT.search(str(colname))
    return m.group(1).upper() if m else None

# ------------------------------
# REAL: per-trial fatigue vector (14,)
# ------------------------------
def fatigue_vector_from_trial_csv(fpath: str):
    df = pd.read_csv(fpath)

    # map raw headers -> canonical channels (choose first mapped column per channel)
    canon_map = {}
    for c in df.columns:
        ch = canonical_channel(c)
        if ch and ch not in canon_map.values():
            canon_map[c] = ch
    if not canon_map:
        print(f"[SKIP no EEG cols] {os.path.basename(fpath)} | headers[:6]={list(df.columns)[:6]}")
        return None

    sub = df[list(canon_map.keys())].copy()
    sub.columns = list(canon_map.values())

    ch_cols = [c for c in CHANNELS_14 if c in sub.columns]
    if not ch_cols:
        print(f"[SKIP order] {os.path.basename(fpath)} -> mapped={sorted(set(sub.columns))}")
        return None

    X = sub[ch_cols].to_numpy(dtype=float)
    Xf = apply_filters(X, FS)
    epochs = epoch_data(Xf, FS, EPOCH_LEN_S)
    if not epochs: return None

    vals = [compute_tbr(ep, FS) for ep in epochs]
    ch_mean = np.nanmean(np.vstack(vals), axis=0)  # aligned to ch_cols

    out = np.full(len(CHANNELS_14), np.nan)
    for i, ch in enumerate(CHANNELS_14):
        if ch in ch_cols:
            out[i] = ch_mean[ch_cols.index(ch)]
    return out

def collect_real(data_root: Path):
    per_part = {}
    parts = sorted([p for p in glob.glob(str(data_root / "protocol_outputs_*")) if os.path.isdir(p)])
    for pdir in parts:
        pid = os.path.basename(pdir)
        per_part.setdefault(pid, {i: {"no_exo": [], "with_exo": []} for i in range(1,10)})
        for f in glob.glob(os.path.join(pdir, "**", "*.csv"), recursive=True):
            if REST_RE.search(os.path.basename(f)): continue
            text = f.lower()
            posture = parse_posture(text)
            cond    = parse_cond(text)
            if posture is None or cond is None: continue
            vec = fatigue_vector_from_trial_csv(f)
            if vec is None: continue
            per_part[pid][posture][cond].append(vec)
    return per_part

def aggregate_real(per_part):
    means = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    sds   = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    pvals = {p: np.full(len(CHANNELS_14), np.nan) for p in range(1,10)}
    counts = {p: {"no_exo":0,"with_exo":0,"paired":0} for p in range(1,10)}

    for p in range(1,10):
        per_no, per_we = {}, {}
        for pid, pdata in per_part.items():
            tr_no = pdata[p]["no_exo"]; tr_we = pdata[p]["with_exo"]
            if tr_no: per_no[pid] = np.nanmean(np.vstack(tr_no), axis=0)
            if tr_we: per_we[pid] = np.nanmean(np.vstack(tr_we), axis=0)

        counts[p]["no_exo"]   = len(per_no)
        counts[p]["with_exo"] = len(per_we)
        common = sorted(set(per_no.keys()) & set(per_we.keys()))
        counts[p]["paired"]   = len(common)

        if per_no:
            A = np.vstack(list(per_no.values()))
            means[p]["no_exo"] = np.nanmean(A, axis=0)
            sds[p]["no_exo"]   = np.nanstd(A, axis=0, ddof=0)
        if per_we:
            B = np.vstack(list(per_we.values()))
            means[p]["with_exo"] = np.nanmean(B, axis=0)
            sds[p]["with_exo"]   = np.nanstd(B, axis=0, ddof=0)

        if common:
            A_pair = np.vstack([per_no[pid] for pid in common])
            B_pair = np.vstack([per_we[pid] for pid in common])
            for ci in range(len(CHANNELS_14)):
                a = A_pair[:,ci]; b = B_pair[:,ci]
                m = ~np.isnan(a) & ~np.isnan(b)
                if m.sum() >= 2:
                    _, pv = ttest_rel(a[m], b[m], nan_policy="omit")
                    pvals[p][ci] = pv
    return means, sds, pvals, counts

# ------------------------------
# Preprocessed scaler utilities (from your pipeline)
# ------------------------------
def load_preproc_scaler_and_ch(preproc_dir: Path, p: int, cond: str):
    """
    Read scaler+channel order from: preprocessed/posture{p}_{cond}.npz
    Keys expected: scale_min, scale_range, ch_names (from your preprocessor). :contentReference[oaicite:1]{index=1}
    """
    fp = preproc_dir / f"posture{p}_{cond}.npz"
    if not fp.exists():
        print(f"[WARN] missing scaler {fp}")
        return None, None, None
    z = np.load(fp, allow_pickle=True)
    smin = z.get("scale_min", None)
    srng = z.get("scale_range", None)
    ch_names = list(z.get("ch_names", []))
    if smin is None or srng is None or not ch_names:
        print(f"[WARN] scaler keys not found in {fp.name}")
        return None, None, None
    smin = np.asarray(smin).reshape(-1)
    srng = np.asarray(srng).reshape(-1)
    return smin, srng, ch_names

def remap_to_14_channels(X_real_scale: np.ndarray, ch_names: list) -> np.ndarray:
    """
    Map (N,T,Csrc) ordered by ch_names to (N,T,14) per CHANNELS_14; missing channels -> NaN.
    """
    N, T, Csrc = X_real_scale.shape
    out = np.full((N, T, len(CHANNELS_14)), np.nan, dtype=np.float32)
    idx = {nm.upper(): i for i, nm in enumerate(ch_names)}
    for j, ch in enumerate(CHANNELS_14):
        if ch.upper() in idx:
            out[:, :, j] = X_real_scale[:, :, idx[ch.upper()]]
    return out

# ------------------------------
# SYNTHETIC: load, inverse-scale+clamp, remap, fatigue
# ------------------------------
def find_synth_npz(run_dir: Path):
    for n in ["synthetic_long.npz","synthetic.npz"]:
        p = run_dir / n
        if p.exists(): return p
    cands = sorted(run_dir.glob("*.npz"))
    return cands[0] if cands else None

def synth_fatigue_rescaled(npz_synth: Path, preproc_dir: Path, p: int, cond: str,
                           clamp: bool = True) -> np.ndarray:
    """
    Load synthetic (N,T,C) in training channel order; inverse-scale with
    scale_min/scale_range; clamp to [min, min+range]; remap to 14 channels;
    return (N,14) TBR per sequence.
    """
    if not npz_synth or not npz_synth.exists():
        return None

    smin, srng, ch_names = load_preproc_scaler_and_ch(preproc_dir, p, cond)
    Xs = np.load(npz_synth)["X"].astype(np.float32)  # (N,T,C)
    N, T, C = Xs.shape

    if smin is not None and srng is not None and len(smin) == C and len(srng) == C:
        smin = smin.reshape(1,1,-1); srng = srng.reshape(1,1,-1)
        Xr = Xs * srng + smin
        if clamp:
            Xr = np.minimum(np.maximum(Xr, smin), smin + srng)
    else:
        Xr = Xs  # scaler missing (should not happen with your preprocessed files)

    # Remap to canonical 14-channel order
    Xr14 = remap_to_14_channels(Xr, ch_names if ch_names else [f"ch{i}" for i in range(C)])

    # Compute TBR per sequence
    out = np.full((N, len(CHANNELS_14)), np.nan, dtype=np.float32)
    for i in range(N):
        seq = Xr14[i]  # (T,14)
        good = ~np.isnan(seq).all(axis=0)
        if not np.any(good):
            continue
        seq[:, ~good] = 0.0
        seq_f = apply_filters(seq, FS)
        epochs = epoch_data(seq_f, FS, EPOCH_LEN_S)
        if not epochs:
            continue
        vals = [compute_tbr(ep, FS) for ep in epochs]
        out[i, :] = np.nanmean(np.vstack(vals), axis=0)
        out[i, ~good] = np.nan
    return out

def aggregate_fake(runs_dir: Path, preproc_dir: Path, real_counts):
    means = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    sds   = {p: {"no_exo": np.full(len(CHANNELS_14), np.nan),
                 "with_exo": np.full(len(CHANNELS_14), np.nan)} for p in range(1,10)}
    pvals = {p: np.full(len(CHANNELS_14), np.nan) for p in range(1,10)}
    rng = np.random.RandomState(0)

    for p in range(1,10):
        npz_no = find_synth_npz(runs_dir / f"posture{p}_no_exo")
        npz_we = find_synth_npz(runs_dir / f"posture{p}_with_exo")

        A = synth_fatigue_rescaled(npz_no, preproc_dir, p, "no_exo", clamp=True) if npz_no else None
        B = synth_fatigue_rescaled(npz_we, preproc_dir, p, "with_exo", clamp=True) if npz_we else None
        if A is None and B is None:
            continue

        n_real_no = real_counts[p]["no_exo"]
        n_real_we = real_counts[p]["with_exo"]

        def take(M, k):
            if M is None or k <= 0:
                return np.empty((0, len(CHANNELS_14)))
            M = M[~np.isnan(M).all(axis=1)]
            if len(M) <= k: return M
            return M[rng.choice(len(M), size=k, replace=False)]

        A_use = take(A, n_real_no)
        B_use = take(B, n_real_we)

        if len(A_use):
            means[p]["no_exo"] = np.nanmean(A_use, axis=0)
            sds[p]["no_exo"]   = np.nanstd(A_use,  axis=0, ddof=0)
        if len(B_use):
            means[p]["with_exo"] = np.nanmean(B_use, axis=0)
            sds[p]["with_exo"]   = np.nanstd(B_use,  axis=0, ddof=0)

        # paired t-test on overlapping counts
        n_pair = min(len(A_use), len(B_use))
        if n_pair >= 2:
            for ci in range(len(CHANNELS_14)):
                a = A_use[:n_pair, ci]; b = B_use[:n_pair, ci]
                m = ~np.isnan(a) & ~np.isnan(b)
                if m.sum() >= 2:
                    _, pv = ttest_rel(a[m], b[m], nan_policy="omit")
                    pvals[p][ci] = pv
    return means, sds, pvals

# ------------------------------
# Plotting
# ------------------------------
def plot_posture(p, real_means, real_sds, real_pvals,
                 fake_means, fake_sds, fake_pvals, out_dir: Path,
                 metric_label="TBR"):
    chs = CHANNELS_14
    x = np.arange(len(chs)); bw = 0.2

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

    ymax = 0.0
    stacks = [rm_no, rm_we, fm_no, fm_we]; errs = [rs_no, rs_we, fs_no, fs_we]
    for ci in range(len(chs)):
        vals = []
        for k in range(4):
            v, e = stacks[k][ci], errs[k][ci]
            if not np.isnan(v): vals.append(v + (0 if np.isnan(e) else e))
        if vals:
            gmax = max(vals); ymax = max(ymax, gmax)
        # real sig
        pv_r = real_pvals[p][ci]
        if pv_r==pv_r and pv_r < 0.05:
            y = (gmax if vals else 0.1) * 1.06 + 0.02
            ax.plot([ci-1.5*bw, ci-0.5*bw], [y,y], color="black", lw=1)
            ax.text(ci - bw, y + 0.005*max(1,y), "*", ha="center", va="bottom", fontsize=12)
        # fake sig
        pv_f = fake_pvals[p][ci]
        if pv_f==pv_f and pv_f < 0.05:
            y2 = (gmax if vals else 0.1) * 1.16 + 0.02
            ax.plot([ci+0.5*bw, ci+1.5*bw], [y2,y2], color="black", lw=1)
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
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_root",   type=str, default="./6s_window", help="Folder with protocol_outputs_*")
    ap.add_argument("--preproc_dir", type=str, default="./preprocessed", help="Folder with posture{p}_{cond}.npz")
    ap.add_argument("--runs_dir",    type=str, default="./timegan_runs", help="Folder with synthetic npz")
    ap.add_argument("--out_dir",     type=str, default="./fatigue_plots", help="Output folder")
    ap.add_argument("--metric_label",type=str, default=Y_LABEL, help="Y-axis label")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("== Collecting REAL trials ...")
    per_part = collect_real(Path(args.data_root))
    real_means, real_sds, real_pvals, real_counts = aggregate_real(per_part)

    print("\n[Real participant counts per posture]")
    for p in range(1,10):
        c = real_counts[p]
        print(f"  posture {p}: no_exo={c['no_exo']}, with_exo={c['with_exo']}, paired={c['paired']}")

    print("\n== Loading SYNTHETIC, inverse-scaling + clamping, matching counts ...")
    fake_means, fake_sds, fake_pvals = aggregate_fake(Path(args.runs_dir), Path(args.preproc_dir), real_counts)

    print("\n== Plotting ...")
    for p in range(1,10):
        plot_posture(p, real_means, real_sds, real_pvals,
                     fake_means, fake_sds, fake_pvals, out_dir,
                     metric_label=args.metric_label)
    print("\nDone.")

if __name__ == "__main__":
    np.random.seed(0)
    main()
