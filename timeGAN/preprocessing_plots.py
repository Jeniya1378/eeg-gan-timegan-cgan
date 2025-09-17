#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG Preprocessing + Labeling Visualization (no CLI)

Steps (each saves signal + spectrogram/PSD):
1. Raw
2. Notch (auto 50/60 Hz)
3. Bandpass (1–45 Hz)
4. Resampling (to 128 Hz)
5. Artifact attenuation (Hampel; light-touch)
6. Epoching (6 s)
7. Features (band powers)

Labeling visuals:
A) Labels timeline with Baseline/Task/Recovery (uses a 'marker'/'label' column if available; else illustrative split)
B) Label tracks (Posture / Condition / State) as stacked bars across time
C) Epoch grid (per 6 s) showing labels per epoch

Run: python eeg_pipeline_with_labels.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import butter, filtfilt, iirnotch, spectrogram, welch, resample

# ========= Defaults (edit these two if needed) =========
CSV_FILE = Path("6s_window\protocol_outputs_1\Posture-1-con-1-lifting in place-fatigue-no exo-08 Aug_EPOCPLUS_451532_20250808_155341_trial1.csv")     # <-- change to your CSV path
OUT_DIR  = Path("./figs_steps")

# Processing constants
FS_FALLBACK = 128.0
TARGET_FS   = 128.0
EPOCH_SEC   = 6.0
LOW_CUT, HIGH_CUT = 1.0, 45.0
EPOC_CHS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]

# ---------- Helpers ----------
def find_time_col(cols):
    lc = {c.lower(): c for c in cols}
    for key in ("timestamp","time","time (s)","time_s","unix_time","datetime","ms","seconds"):
        if key in lc: return lc[key]
    for key in ("counter","sample","samples","frame"):
        if key in lc: return lc[key]
    return None

def estimate_fs(series, fallback=128.0):
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if len(s) < 6: return fallback
    diffs = np.diff(s); diffs = diffs[(diffs > 0) & (diffs < np.nanpercentile(diffs, 99))]
    if len(diffs) == 0: return fallback
    med = np.median(diffs)
    fs = (1.0/med if med > 1.0 or (0.001 <= med <= 0.2) else 1000.0/med)
    return fs if np.isfinite(fs) and fs >= 10 else fallback

def match_epoc_columns(df):
    present = []
    lowered = {c.lower(): c for c in df.columns}
    def cands(ch):
        base = ch.lower()
        return [base, f"eeg.{base}", f"{base} (uv)", f"eeg.{base} (uv)", f"{base}_uv", f"eeg_{base}"]
    for ch in EPOC_CHS:
        if ch in df.columns: present.append(ch); continue
        for cand in cands(ch):
            if cand in lowered: present.append(lowered[cand]); break
    return present

def detect_line_freq(x, fs):
    f, P = welch(x[:min(len(x), int(fs*20))], fs=fs, nperseg=int(fs*4), noverlap=int(fs*2))
    def band(lo,hi):
        m = (f>=lo) & (f<=hi)
        return np.trapz(P[m], f[m]) if m.any() else 0.0
    return 50.0 if band(49,51) > band(59,61) else 60.0

def hampel(x, k=11, t0=5.0):
    y=x.copy(); n=len(x)
    for i in range(n):
        lo=max(i-k,0); hi=min(i+k+1,n)
        med=np.median(x[lo:hi]); mad=np.median(np.abs(x[lo:hi]-med)) or 1e-6
        if abs(x[i]-med) > t0*1.4826*mad: y[i]=med
    return y

# ----- plotting primitives -----
def plot_signal(x, fs, title, out_fp):
    t = np.arange(len(x))/fs
    plt.figure(figsize=(10,3)); plt.plot(t, x)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_fp, dpi=200); plt.close()

def plot_spectrogram(x, fs, title, out_fp):
    f,tt,Sxx=spectrogram(x,fs=fs,nperseg=int(fs*2),noverlap=int(fs*1))
    plt.figure(figsize=(10,4))
    plt.pcolormesh(tt,f,10*np.log10(Sxx+1e-12),shading="gouraud")
    plt.ylim(0,60); plt.xlabel("Time (s)"); plt.ylabel("Hz"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_fp,dpi=200); plt.close()

def plot_psd(x, fs, title, out_fp):
    f,P=welch(x,fs=fs,nperseg=int(fs*2),noverlap=int(fs*1))
    plt.figure(figsize=(8,4)); plt.semilogy(f,P)
    plt.xlim(0,60); plt.xlabel("Hz"); plt.ylabel("PSD"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_fp,dpi=200); plt.close()

def plot_bandpowers(x, fs, title, out_fp):
    f,P=welch(x,fs=fs,nperseg=int(fs*2),noverlap=int(fs*1))
    bands={"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
    bp=[np.trapz(P[(f>=lo)&(f<hi)],f[(f>=lo)&(f<hi)]) for lo,hi in bands.values()]
    plt.figure(figsize=(6,4)); plt.bar(list(bands.keys()),bp)
    plt.ylabel("Band Power"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_fp,dpi=200); plt.close()

# ----- labeling helpers -----
def parse_labels_from_filename(path: Path):
    name = path.name.lower()
    # posture number
    m_post = re.search(r"posture[-_\s]*([0-9]+)", name)
    posture = f"Posture {m_post.group(1)}" if m_post else "Posture ?"
    # condition
    cond = "with exo" if ("with exo" in name or "withexo" in name) else ("no exo" if ("no exo" in name or "noexo" in name) else "condition ?")
    # mental state keyword if present
    state = "fatigue" if "fatigue" in name else ("stress" if "stress" in name else ("cognitive load" if "cognitive load" in name or "cognitiveload" in name else "state ?"))
    # trial
    m_trial = re.search(r"trial[-_\s]*([0-9]+)", name)
    trial = f"trial {m_trial.group(1)}" if m_trial else "trial ?"
    # task (best-effort: take chunk between first two dashes after posture-*)
    task = None
    try:
        raw = path.stem.replace("_", " ")
        bits = raw.split("-")
        # crude search for a meaningful phrase inside the filename
        for b in bits:
            if any(k in b.lower() for k in ["lifting", "overhead", "squat", "kneel", "reach", "twist", "walk", "standing"]):
                task = b.strip()
                break
    except Exception:
        task = None
    if not task: task = "task ?"
    return {"posture": posture, "condition": cond, "state": state, "trial": trial, "task": task}

def find_marker_column(df):
    for c in df.columns:
        if str(c).lower() in ["marker","markers","event","events","label","labels","phase"]:
            return c
    return None

def make_segments_from_markers(marker_series, fs):
    # Expect categorical strings; derive contiguous segments with same label
    labels = marker_series.astype(str).fillna("").values
    segments = []
    if len(labels)==0: return segments
    start = 0; current = labels[0]
    for i in range(1,len(labels)):
        if labels[i] != current:
            segments.append((start/fs, i/fs, current))
            start = i; current = labels[i]
    segments.append((start/fs, len(labels)/fs, current))
    return segments

def make_default_btr_segments(total_sec):
    # Illustrative split: 30% baseline, 50% task, 20% recovery
    b = 0.30*total_sec; t = 0.50*total_sec; r = total_sec - (b+t)
    return [(0, b, "baseline"), (b, b+t, "task"), (b+t, total_sec, "recovery")]

def draw_timeline_segments(ax, segments, colors, y=0.1, h=0.8):
    for (t0, t1, lab) in segments:
        ax.add_patch(Rectangle((t0, y), t1-t0, h, color=colors.get(lab, "#cccccc"), alpha=0.35, lw=0))
    ax.set_ylim(0,1); ax.set_xlim(0, max(s[1] for s in segments))
    ax.set_yticks([])

# ---------- main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(CSV_FILE,engine="python")
    ch_cols=match_epoc_columns(df); 
    if not ch_cols: raise SystemExit("No EPOC+ EEG columns found.")
    ch_idx=0; ch_name=ch_cols[ch_idx]
    time_col=find_time_col(df.columns)
    fs=estimate_fs(df[time_col],FS_FALLBACK) if time_col else FS_FALLBACK
    sig_raw=df[ch_cols].apply(pd.to_numeric,errors="coerce").values[:,ch_idx]

    # Labels from filename
    meta = parse_labels_from_filename(CSV_FILE)

    # 1) RAW
    plot_signal(sig_raw,fs,f"Raw – {ch_name}",OUT_DIR/"1_raw_signal.png")
    plot_spectrogram(sig_raw,fs,"Raw Spectrogram",OUT_DIR/"1_raw_spec.png")

    # 2) NOTCH
    notch_hz = detect_line_freq(sig_raw, fs)
    b_n,a_n = iirnotch(notch_hz/(fs/2), 30.0)
    sig_notch = filtfilt(b_n,a_n,sig_raw)
    plot_signal(sig_notch,fs,f"After Notch ({int(notch_hz)} Hz) – {ch_name}",OUT_DIR/"2_notch_signal.png")
    plot_spectrogram(sig_notch,fs,"Notched Spectrogram",OUT_DIR/"2_notch_spec.png")
    plot_psd(sig_notch,fs,"PSD After Notch",OUT_DIR/"2_notch_psd.png")

    # 3) BANDPASS 1–45
    b_bp,a_bp = butter(4,[LOW_CUT/(fs/2), HIGH_CUT/(fs/2)], btype="band")
    sig_bp = filtfilt(b_bp,a_bp,sig_notch)
    plot_signal(sig_bp,fs,"After Bandpass (1–45 Hz)",OUT_DIR/"3_bandpass_signal.png")
    plot_spectrogram(sig_bp,fs,"Bandpass Spectrogram",OUT_DIR/"3_bandpass_spec.png")
    plot_psd(sig_bp,fs,"PSD After Bandpass",OUT_DIR/"3_bandpass_psd.png")

    # 4) RESAMPLE -> 128 Hz
    N = int(round(len(sig_bp)*TARGET_FS/fs))
    sig_rs = resample(sig_bp, N)
    plot_signal(sig_rs,TARGET_FS,"After Resampling (128 Hz)",OUT_DIR/"4_resampled_signal.png")
    plot_spectrogram(sig_rs,TARGET_FS,"Resampled Spectrogram",OUT_DIR/"4_resampled_spec.png")
    plot_psd(sig_rs,TARGET_FS,"PSD After Resampling",OUT_DIR/"4_resampled_psd.png")

    # 5) ARTIFACT (Hampel; conservative)
    sig_art = hampel(sig_rs, k=11, t0=5.0)
    plot_signal(sig_art,TARGET_FS,"After Artifact (Hampel)",OUT_DIR/"5_artifact_signal.png")
    plot_spectrogram(sig_art,TARGET_FS,"Artifact-attenuated Spectrogram",OUT_DIR/"5_artifact_spec.png")
    plot_psd(sig_art,TARGET_FS,"PSD After Artifact",OUT_DIR/"5_artifact_psd.png")

    # 6) EPOCHING (first 6 s only for figures)
    ep_len = int(EPOCH_SEC * TARGET_FS)
    if len(sig_art) < ep_len: raise SystemExit("Not enough samples for a 6 s epoch.")
    sig_ep = sig_art[:ep_len]
    plot_signal(sig_ep,TARGET_FS,"Epoch (6 s)",OUT_DIR/"6_epoch_signal.png")
    plot_spectrogram(sig_ep,TARGET_FS,"Epoch Spectrogram",OUT_DIR/"6_epoch_spec.png")
    plot_psd(sig_ep,TARGET_FS,"PSD (Epoch)",OUT_DIR/"6_epoch_psd.png")

    # 7) FEATURES
    plot_bandpowers(sig_ep,TARGET_FS,"Band Powers (Epoch)",OUT_DIR/"7_features_bandpower.png")
    plot_spectrogram(sig_ep,TARGET_FS,"Features Stage Spectrogram",OUT_DIR/"7_features_spec.png")

    # ===== Labeling visuals =====
    # Build B/T/R segments from markers if available, else illustrative split
    marker_col = find_marker_column(df)
    if marker_col:
        segments = make_segments_from_markers(df[marker_col], fs)
        if not segments:
            total_sec = len(sig_raw)/fs
            segments = make_default_btr_segments(total_sec)
    else:
        total_sec = len(sig_raw)/fs
        segments = make_default_btr_segments(total_sec)

    colors = {
        "baseline": "#4caf50", "task": "#2196f3", "recovery": "#ff9800",
        "with exo": "#6a1b9a", "no exo": "#00897b"
    }

    # A) Timeline: raw signal with background B/T/R bands
    t = np.arange(len(sig_raw))/fs
    fig, ax = plt.subplots(figsize=(12,3))
    draw_timeline_segments(ax, segments, colors)
    ax.plot(t, (sig_raw-np.median(sig_raw))/ (np.std(sig_raw)+1e-6), lw=0.8, color="k")
    ax.set_xlabel("Time (s)"); ax.set_title("Labels Timeline (Baseline/Task/Recovery)")
    # simple legend
    handles = [Rectangle((0,0),1,1,color=colors[k],alpha=0.35) for k in ["baseline","task","recovery"]]
    ax.legend(handles, ["Baseline","Task","Recovery"], loc="upper right", frameon=False)
    plt.tight_layout(); plt.savefig(OUT_DIR/"8_labels_timeline.png", dpi=200); plt.close()

    # B) Label tracks: posture / condition / state over full duration
    fig, ax = plt.subplots(figsize=(12,2.8))
    ax.set_xlim(0, segments[-1][1]); ax.set_ylim(0,3)
    ax.set_yticks([0.5,1.5,2.5]); ax.set_yticklabels(["Posture","Condition","State"])
    # posture & condition constant across file (from filename)
    ax.add_patch(Rectangle((0,0.1), segments[-1][1], 0.8, color="#9e9e9e", alpha=0.35))
    ax.text(0.2,0.5, meta["posture"], va="center", ha="left")
    cond_color = colors.get(meta["condition"], "#cccccc")
    ax.add_patch(Rectangle((0,1.1), segments[-1][1], 0.8, color=cond_color, alpha=0.35))
    ax.text(0.2,1.5, meta["condition"], va="center", ha="left")
    # state follows B/T/R segments
    for (t0,t1,lab) in segments:
        ax.add_patch(Rectangle((t0,2.1), t1-t0, 0.8, color=colors.get(lab,"#cccccc"), alpha=0.35))
        ax.text((t0+t1)/2,2.5, lab, va="center", ha="center", fontsize=9)
    ax.set_xlabel("Time (s)")
    plt.tight_layout(); plt.savefig(OUT_DIR/"8_labels_tracks.png", dpi=200); plt.close()

    # C) Epoch grid across the WHOLE resampled series (not just first 6 s)
    #    (use B/T/R to assign a state to each 6 s epoch)
    total_sec_rs = len(sig_rs)/TARGET_FS
    n_ep = int(np.floor(total_sec_rs / EPOCH_SEC))
    if n_ep < 1: n_ep = 1
    # helper to map time->state from segments
    def state_at(tsec):
        for (a,b,s) in segments:
            if a <= tsec < b: return s
        return segments[-1][2]
    states = [state_at(i*EPOCH_SEC) for i in range(n_ep)]
    # map to colors
    state_cols = [colors.get(s,"#cccccc") for s in states]
    fig, ax = plt.subplots(figsize=(max(6, n_ep*0.6), 1.8))
    for i, sc in enumerate(state_cols):
        ax.add_patch(Rectangle((i, 0), 1, 1, color=sc, alpha=0.8, lw=0))
        ax.text(i+0.5, 0.5, f"E{i+1}", ha="center", va="center", fontsize=8, color="k")
    ax.set_xlim(0, n_ep); ax.set_ylim(0,1); ax.set_yticks([])
    ax.set_xticks(np.arange(n_ep)+0.5); ax.set_xticklabels([f"{s}" for s in states], rotation=45, ha="right", fontsize=8)
    ax.set_title("Epoch Grid (6 s each) – State per Epoch")
    plt.tight_layout(); plt.savefig(OUT_DIR/"8_labels_epoch_grid.png", dpi=200); plt.close()

    # Small label card (metadata) – optional but handy
    fig, ax = plt.subplots(figsize=(4.2,2.8))
    ax.axis("off")
    txt = (f"{meta['posture']}\n"
           f"Task: {meta['task']}\n"
           f"Condition: {meta['condition']}\n"
           f"State (from name): {meta['state']}\n"
           f"{meta['trial']}")
    ax.text(0.02, 0.95, "LABEL CARD", fontsize=11, weight="bold", va="top")
    ax.text(0.02, 0.85, txt, fontsize=10, va="top")
    plt.tight_layout(); plt.savefig(OUT_DIR/"8_labels_card.png", dpi=200); plt.close()

    print("All figures (including labeling) saved to", OUT_DIR)

if __name__=="__main__":
    main()
