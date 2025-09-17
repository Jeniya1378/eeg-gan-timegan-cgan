#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Emotiv EPOC+ EEG CSVs into 6 s epochs (fixed length) per posture×condition.

Key fix: enforce a single epoch length for all files (default 6*128=768 samples)
to avoid shape mismatches when concatenating across files.

Run (defaults work out of the box):
  python preprocess_eeg.py
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, iirnotch

EPOC_CHS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", type=str, default="./6s_window",
                    help="Folder containing protocol_outputs_{1..4}")
    ap.add_argument("--out", type=str, default="./preprocessed", help="Output folder")
    ap.add_argument("--fs", type=float, default=128.0, help="Fallback sampling rate for filtering (Hz)")
    ap.add_argument("--epoch_sec", type=float, default=6.0, help="Epoch length (seconds)")
    ap.add_argument("--overlap", type=float, default=0.0, help="Epoch overlap fraction (0.0 = none)")
    ap.add_argument("--low_cut", type=float, default=1.0, help="Bandpass low cutoff (Hz)")
    ap.add_argument("--high_cut", type=float, default=45.0, help="Bandpass high cutoff (Hz)")
    ap.add_argument("--notch_q", type=float, default=30.0, help="Notch Q factor")
    ap.add_argument("--min_channels", type=int, default=10, help="Min EPOC+ channels required to accept a file")
    ap.add_argument("--dry_run", action="store_true", help="Scan only; do not save")
    # IMPORTANT: fixed epoch length
    ap.add_argument("--epoch_fs_target", type=float, default=128.0,
                    help="Target Hz used to compute epoch length (fixed for all files)")
    return ap.parse_args()

# ---------- Helpers ----------
def find_time_col(cols: List[str]) -> Optional[str]:
    lc = {c.lower(): c for c in cols}
    for key in ("timestamp","time","time (s)","time_s","unix_time","datetime","ms","seconds"):
        if key in lc: return lc[key]
    for key in ("counter","sample","samples","frame"):
        if key in lc: return lc[key]
    return None

def estimate_fs(series: pd.Series, fallback_fs: float = 128.0) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if len(s) < 6: return fallback_fs
    diffs = np.diff(s)
    diffs = diffs[(diffs > 0) & (diffs < np.nanpercentile(diffs, 99))]
    if len(diffs) == 0: return fallback_fs
    med = float(np.median(diffs))
    if med > 1.0 or (0.001 <= med <= 0.2):
        fs = 1.0 / med
    else:
        fs = 1000.0 / med
    if not np.isfinite(fs) or fs < 10: return fallback_fs
    return fs

def match_epoc_columns(df: pd.DataFrame) -> List[str]:
    present = []
    lowered = {c.lower(): c for c in df.columns}
    def cands(ch: str) -> List[str]:
        base = ch.lower()
        return [base, f"eeg.{base}", f"{base} (uv)", f"eeg.{base} (uv)", f"{base}_uv", f"eeg_{base}"]
    for ch in EPOC_CHS:
        found = None
        if ch in df.columns:
            found = ch
        else:
            for cand in cands(ch):
                if cand in lowered:
                    found = lowered[cand]; break
        if found is not None:
            present.append(found)
    return present

def detect_line_freq(x: np.ndarray, fs: float) -> float:
    N = min(len(x), int(fs*20))
    if N < int(fs*4): return 60.0
    f, P = welch(x[:N], fs=fs, nperseg=int(fs*4), noverlap=int(fs*2))
    def bp(lo,hi):
        m = (f>=lo)&(f<=hi); 
        return float(np.trapz(P[m], f[m])) if np.any(m) else 0.0
    return 50.0 if bp(49,51) > bp(59,61) else 60.0

def design_filters(fs: float, low_cut: float, high_cut: float, notch_hz: float, notch_q: float):
    nyq = 0.5*fs
    lo = max(0.001, low_cut/nyq); hi = min(0.999, high_cut/nyq)
    b_bp, a_bp = butter(4, [lo, hi], btype="band")
    w0 = notch_hz/nyq
    b_n, a_n = iirnotch(w0, notch_q)
    return (b_bp, a_bp), (b_n, a_n)

def epoch_array_fixed(arr: np.ndarray, samples_per_epoch: int, overlap: float) -> np.ndarray:
    """
    Segment *by a fixed sample count* (same across all files).
    """
    win = int(samples_per_epoch)
    step = win if overlap <= 0 else int(win * (1 - overlap))
    if step <= 0: step = win
    Ns = arr.shape[0]
    starts = np.arange(0, Ns - win + 1, step)
    if len(starts) == 0:
        return np.empty((0, win, arr.shape[1]), dtype=np.float32)
    out = np.stack([arr[s:s+win, :] for s in starts], axis=0)
    return out.astype(np.float32)

def parse_meta_from_name(name: str) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    lower = name.lower()
    m_post = re.search(r"posture[-_\s]*([0-9]+)", lower)
    posture = int(m_post.group(1)) if m_post else None
    cond = "with_exo" if ("with exo" in lower or "withexo" in lower) else ("no_exo" if ("no exo" in lower or "noexo" in lower) else None)
    m_trial = re.search(r"trial[-_\s]*([0-9]+)", lower) or re.search(r"t([0-9]+)(?![0-9])", lower)
    trial = int(m_trial.group(1)) if m_trial else None
    return posture, cond, trial

# ---------- Main ----------
def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed epoch length in samples for EVERY file (prevents 768 vs 766 mismatches)
    epoch_len_samples = int(round(args.epoch_sec * round(args.epoch_fs_target)))
    print(f"Using fixed epoch length: {epoch_len_samples} samples "
          f"(~{args.epoch_sec}s @ {round(args.epoch_fs_target)} Hz)")

    # Discover files
    files = []
    for pdir in sorted(root.glob("protocol_outputs_*")):
        files += list(pdir.rglob("*.csv"))
    files = [fp for fp in files if fp.name.lower().startswith("posture-") and "trial" in fp.name.lower()]
    if not files:
        raise SystemExit(f"No Posture-* trial CSVs found under '{root}'.")

    buckets: Dict[Tuple[int,str], Dict[str, list]] = {}
    index_rows = []

    for fp in files:
        try:
            df = pd.read_csv(fp, engine="python")
        except Exception as e:
            print(f"[SKIP] {fp.name}: read error: {e}"); continue

        cols = match_epoc_columns(df)
        if len(cols) < args.min_channels:
            print(f"[SKIP] {fp.name}: only {len(cols)}/{len(EPOC_CHS)} EPOC+ channels present (min {args.min_channels}).")
            continue

        time_col = find_time_col(df.columns)
        fs_est = estimate_fs(df[time_col], args.fs) if time_col else args.fs

        posture, cond, trial = parse_meta_from_name(fp.name)
        if posture is None or cond is None or trial is None:
            print(f"[SKIP] {fp.name}: could not parse posture/condition/trial."); continue

        m_part = re.search(r"protocol_outputs_([0-9]+)", str(fp.parent))
        participant = int(m_part.group(1)) if m_part else -1

        X = df[cols].apply(pd.to_numeric, errors="coerce").values
        if X.shape[0] < epoch_len_samples:
            print(f"[SKIP] {fp.name}: not enough samples ({X.shape[0]}) for one fixed epoch ({epoch_len_samples}).")
            continue

        # Filters (use fs_est for correct frequency response)
        notch_hz = detect_line_freq(X[:min(len(X), int(fs_est*20)), 0], fs_est)
        (b_bp, a_bp), (b_n, a_n) = design_filters(fs_est, args.low_cut, args.high_cut, notch_hz, args.notch_q)
        try:
            Xn = filtfilt(b_n, a_n, X, axis=0)
        except Exception:
            Xn = X
        Xf = filtfilt(b_bp, a_bp, Xn, axis=0)

        # Epoch with FIXED sample count
        epochs = epoch_array_fixed(Xf, epoch_len_samples, args.overlap)
        if epochs.shape[0] == 0:
            print(f"[SKIP] {fp.name}: epoching produced 0 windows."); continue

        key = (posture, cond)
        buckets.setdefault(key, {"X": [], "participant": [], "trial": [], "fs": [], "ch_names": []})
        buckets[key]["X"].append(epochs.astype(np.float32))
        buckets[key]["participant"].append(np.full((epochs.shape[0],), participant, dtype=np.int32))
        buckets[key]["trial"].append(np.full((epochs.shape[0],), trial, dtype=np.int32))
        buckets[key]["fs"].append(fs_est)
        buckets[key]["ch_names"] = [c for c in cols]

        print(f"[OK] {fp.name}: fs≈{fs_est:.1f}Hz | epochs={epochs.shape[0]} | notch={notch_hz:.0f}Hz | ch={len(cols)}")

    if args.dry_run:
        print("Dry run complete. No files were saved."); return

    # Save per (posture, condition)
    for (posture, cond), pack in sorted(buckets.items()):
        X = np.concatenate(pack["X"], axis=0)                     # (N, T_fixed, C)
        participant = np.concatenate(pack["participant"], axis=0)
        trial = np.concatenate(pack["trial"], axis=0)
        fs_bucket = float(np.median(np.array(pack["fs"], dtype=np.float32)))
        ch_names = pack["ch_names"]

        # Per-bucket min-max scaling (channel-wise)
        flat = X.reshape(-1, X.shape[-1])
        mn = np.nanmin(flat, axis=0); mx = np.nanmax(flat, axis=0)
        rng = mx - mn; rng[rng == 0] = 1e-6
        Xs = (X - mn) / rng

        out_fp = Path(args.out) / f"posture{posture}_{cond}.npz"
        np.savez_compressed(out_fp,
            X=Xs.astype(np.float32),
            participant=participant.astype(np.int32),
            trial=trial.astype(np.int32),
            posture=np.int32(posture),
            condition=str(cond),
            fs=np.float32(fs_bucket),
            ch_names=np.array(ch_names, dtype=object),
            scale_min=mn.astype(np.float32),
            scale_range=rng.astype(np.float32),
            epoch_len_samples=np.int32(epoch_len_samples)
        )
        print(f"Saved {out_fp}  shape={Xs.shape}  fs≈{fs_bucket:.2f}Hz")

        index_rows.append({
            "posture": posture, "condition": cond,
            "n_epochs": int(Xs.shape[0]), "seq_len": int(Xs.shape[1]),
            "n_channels": int(Xs.shape[2]), "fs_hz": round(fs_bucket, 3),
            "file": str(out_fp)
        })

    if index_rows:
        pd.DataFrame(index_rows).sort_values(["posture","condition"]).to_csv(Path(args.out)/"prep_index.csv", index=False)
        print(f"\nWrote index CSV: {Path(args.out)/'prep_index.csv'}")
    else:
        print("\nNo buckets produced. Check filename patterns and channel headers.")

if __name__ == "__main__":
    main()
