#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation for TimeGAN EEG augmentation
- Per posture (1..9) + Global metrics
- Discriminative: Accuracy, AUC
- Predictive (TSTR / TRTS): RMSE, R^2
- Statistical: PSD diff, ACF diff, inter-channel corr ("coherence") diff
- PCA & t-SNE plots (global): color by posture, marker by domain (Real/Synth)

Assumes folder layout:
  preprocessed/
    posture1_with_exo.npz
    posture1_no_exo.npz
    ...
  timegan_runs/
    posture1_with_exo/synthetic.npz
    posture1_no_exo/synthetic.npz
    ...

Usage:
  python evaluation.py --real_dir ./preprocessed --synth_dir ./timegan_runs --out ./eval_out --fs 128
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(max(1, torch.get_num_threads()))

# -------------------------- Models --------------------------

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, hn = self.rnn(x)
        return torch.sigmoid(self.out(hn[-1]))

class RNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, num_layers=1, output_dim=None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, hn = self.rnn(x)
        return self.out(hn[-1])

# -------------------------- Helpers --------------------------

def autocorr_seq(x, maxlag):
    if np.std(x) < 1e-8:  # nearly constant
        return 0.0
    vals = []
    for lag in range(1, maxlag+1):
        if lag >= len(x): break
        vals.append(np.corrcoef(x[:-lag], x[lag:])[0,1])
    return float(np.mean(vals)) if vals else 0.0

def discriminative_score(real, fake, epochs=20, lr=1e-3, hidden=24, seed=0):
    # balance classes
    n = min(len(real), len(fake))
    idx_r = np.random.RandomState(seed).permutation(len(real))[:n]
    idx_f = np.random.RandomState(seed+1).permutation(len(fake))[:n]
    X = np.concatenate([real[idx_r], fake[idx_f]], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)], axis=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = RNNClassifier(X.shape[-1], hidden)
    opt = optim.Adam(clf.parameters(), lr=lr)
    lossf = nn.BCELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        p = clf(Xt)
        loss = lossf(p, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        p = clf(torch.tensor(Xte, dtype=torch.float32)).numpy().flatten()
    yhat = (p >= 0.5).astype(int)
    acc = accuracy_score(yte, yhat)
    try:
        auc = roc_auc_score(yte, p)
    except ValueError:
        auc = np.nan
    return acc, auc

def predictive_score(X_train, y_train, X_test, y_test, epochs=50, lr=1e-3, hidden=24):
    model = RNNPredictor(X_train.shape[-1], hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = lossf(pred, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        yhat = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    rmse = np.sqrt(mean_squared_error(y_test, yhat))
    r2   = r2_score(y_test, yhat, multioutput='uniform_average')
    return rmse, r2

def statistical_similarity(real, fake, fs=128.0):
    # PSD
    fr, psd_r = sig.welch(real, fs=fs, axis=1, nperseg=256)
    ff, psd_f = sig.welch(fake, fs=fs, axis=1, nperseg=256)
    psd_diff = float(np.mean(np.abs(psd_r.mean(axis=0) - psd_f.mean(axis=0))))
    # ACF (mean across channels)
    maxlag = int(0.75*fs)
    acf_r = []; acf_f = []
    for ch in range(real.shape[-1]):
        acf_r.append(np.mean([autocorr_seq(seq[:,ch], maxlag) for seq in real]))
        acf_f.append(np.mean([autocorr_seq(seq[:,ch], maxlag) for seq in fake]))
    acf_diff = float(np.mean(np.abs(np.array(acf_r) - np.array(acf_f))))
    # Inter-channel correlation as coherence proxy
    r_flat = real.reshape(-1, real.shape[-1])
    f_flat = fake.reshape(-1, fake.shape[-1])
    corr_r = np.corrcoef(r_flat, rowvar=False)
    corr_f = np.corrcoef(f_flat, rowvar=False)
    coh_diff = float(np.mean(np.abs(corr_r - corr_f)))
    return psd_diff, acf_diff, coh_diff

def load_posture_pairs(real_dir: Path, synth_dir: Path):
    """
    Returns dict: posture -> (real_array, fake_array), concatenating conditions that exist.
    real_array/fake_array shapes: (N, T, C)
    """
    pairs = {}
    for p in range(1, 10):
        real_list = []
        fake_list = []
        for cond in ["with_exo", "no_exo"]:
            rfp = real_dir / f"posture{p}_{cond}.npz"
            sfp = synth_dir / f"posture{p}_{cond}" / "synthetic.npz"
            if rfp.exists() and sfp.exists():
                r = np.load(rfp)["X"].astype(np.float32)
                f = np.load(sfp)["X"].astype(np.float32)
                # balance within condition to avoid strong skew
                m = min(len(r), len(f))
                if m > 0:
                    real_list.append(r[:m])
                    fake_list.append(f[:m])
        if real_list and fake_list:
            real = np.concatenate(real_list, axis=0)
            fake = np.concatenate(fake_list, axis=0)
            pairs[p] = (real, fake)
    return pairs

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--real_dir", type=str, default="./preprocessed")
    ap.add_argument("--synth_dir", type=str, default="./timegan_runs")
    ap.add_argument("--out", type=str, default="./eval_out")
    ap.add_argument("--fs", type=float, default=128.0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    real_dir = Path(args.real_dir); synth_dir = Path(args.synth_dir)

    # Load per-posture real/synth (conditions concatenated)
    pairs = load_posture_pairs(real_dir, synth_dir)
    if not pairs:
        raise SystemExit("No matching posture pairs found. Make sure synthetic.npz exists for each trained model.")

    rows = []
    all_real = []
    all_fake = []
    all_labels = []
    all_domain = []

    for posture in sorted(pairs.keys()):
        real, fake = pairs[posture]
        # Discriminative
        acc, auc = discriminative_score(real, fake)
        # Predictive (TSTR/TRTS): predict last step from earlier steps
        Xr_in, yr = real[:,:-1,:], real[:,-1,:]
        Xf_in, yf = fake[:,:-1,:], fake[:,-1,:]
        rmse_tstr, r2_tstr = predictive_score(Xf_in, yf, Xr_in, yr)
        rmse_trts, r2_trts = predictive_score(Xr_in, yr, Xf_in, yf)
        # Statistical
        psd_diff, acf_diff, coh_diff = statistical_similarity(real, fake, fs=args.fs)

        rows.append({
            "posture": posture,
            "disc_acc": acc, "disc_auc": auc,
            "rmse_tstr": rmse_tstr, "r2_tstr": r2_tstr,
            "rmse_trts": rmse_trts, "r2_trts": r2_trts,
            "psd_diff": psd_diff, "acf_diff": acf_diff, "coh_diff": coh_diff,
            "n_real": len(real), "n_fake": len(fake),
            "seq_len": real.shape[1], "n_ch": real.shape[2]
        })

        all_real.append(real); all_fake.append(fake)
        all_labels += [posture]*len(real) + [posture]*len(fake)
        all_domain += [1]*len(real) + [0]*len(fake)

    # Save per-posture CSV
    df = pd.DataFrame(rows).sort_values("posture")
    df.to_csv(out / "metrics_per_posture.csv", index=False)
    print(f"Wrote {out/'metrics_per_posture.csv'}")

    # Global metrics
    R = np.concatenate(all_real, axis=0); F = np.concatenate(all_fake, axis=0)
    acc_g, auc_g = discriminative_score(R, F)
    Xr_in, yr = R[:,:-1,:], R[:,-1,:]
    Xf_in, yf = F[:,:-1,:], F[:,-1,:]
    rmse_tstr_g, r2_tstr_g = predictive_score(Xf_in, yf, Xr_in, yr)
    rmse_trts_g, r2_trts_g = predictive_score(Xr_in, yr, Xf_in, yf)
    psd_diff_g, acf_diff_g, coh_diff_g = statistical_similarity(R, F, fs=args.fs)

    g = {
        "disc_acc": acc_g, "disc_auc": auc_g,
        "rmse_tstr": rmse_tstr_g, "r2_tstr": r2_tstr_g,
        "rmse_trts": rmse_trts_g, "r2_trts": r2_trts_g,
        "psd_diff": psd_diff_g, "acf_diff": acf_diff_g, "coh_diff": coh_diff_g,
        "n_real": len(R), "n_fake": len(F), "seq_len": R.shape[1], "n_ch": R.shape[2]
    }
    pd.DataFrame([g]).to_csv(out / "metrics_global.csv", index=False)
    print(f"Wrote {out/'metrics_global.csv'}")

    # Visualizations
    X_all = np.concatenate([R, F], axis=0).reshape(len(R)+len(F), -1)
    labels = np.array(all_labels)
    domain = np.array(all_domain)

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pc = pca.fit_transform(X_all)
    plt.figure(figsize=(7,5))
    for pid in np.unique(labels):
        m_real = (labels==pid)&(domain==1)
        m_fake = (labels==pid)&(domain==0)
        plt.scatter(pc[m_real,0], pc[m_real,1], s=10, alpha=0.65, label=f"P{pid} real")
        plt.scatter(pc[m_fake,0], pc[m_fake,1], s=10, alpha=0.65, marker='x', label=f"P{pid} gen")
    plt.title("PCA: posture clusters (real vs generated)")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small')
    plt.tight_layout(); plt.savefig(out/"pca_global.png", dpi=160); plt.close()

    # t-SNE (guard against OOM/slow)
    try:
        ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
        ts2 = ts.fit_transform(X_all)
        plt.figure(figsize=(7,5))
        for pid in np.unique(labels):
            m_real = (labels==pid)&(domain==1)
            m_fake = (labels==pid)&(domain==0)
            plt.scatter(ts2[m_real,0], ts2[m_real,1], s=10, alpha=0.65, label=f"P{pid} real")
            plt.scatter(ts2[m_fake,0], ts2[m_fake,1], s=10, alpha=0.65, marker='x', label=f"P{pid} gen")
        plt.title("t-SNE: posture clusters (real vs generated)")
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small')
        plt.tight_layout(); plt.savefig(out/"tsne_global.png", dpi=160); plt.close()
    except Exception as e:
        print(f"t-SNE skipped: {e}")

    print(f"Saved plots to {out}")

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
