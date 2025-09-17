#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_cgan_posture_defaults.py — Evaluation for 9 posture-specialist CGANs (no CLI).
Per posture, the model is conditioned on condition ∈ {no_exo=0, with_exo=1}.

Outputs under SAVE_ROOT/:
  global/
    metrics_discriminative.csv
    metrics_predictive.csv
    metrics_stats.csv
    pca_scatter.png
    tsne_scatter.png
    tsne_real_gen.png
  posture{p}/
    metrics_discriminative.csv
    metrics_predictive.csv
    metrics_stats.csv
"""

# =================== DEFAULTS (edit as needed) ===================
DATA_DIR          = "./preprocessed"
RUNS_ROOT         = "./cgan_runs_posture"
SAVE_ROOT         = "./cgan_eval_posture_v2"
POSTURES          = list(range(1, 10))         # [1..9]
NOISE_DIM         = 100
SAMPLES_PER_COND  = "match"                    # "match" or an int, e.g., 400
TARGET_CHANNEL    = ""                         # channel name (e.g., "AF4"); if "", uses last channel
TSNE_PERPLEXITY   = 30.0
TSNE_ITER         = 1000
SEED              = 123
# ================================================================

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

C, T = 14, 768
NUM_COND = 2  # 0=no_exo, 1=with_exo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _split_y_posture(y_posture, n_real):
    """
    y_posture is concatenated [y_real ; y_gen].
    Return (y_real, y_gen) with lengths matching X_real, X_gen.
    """
    y_real = y_posture[:n_real]
    y_gen  = y_posture[n_real:]
    return y_real, y_gen

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------- IO helpers -----------------
def load_real_posture(data_dir: str, posture: int):
    paths = {
        0: Path(data_dir)/f"posture{posture}_no_exo.npz",
        1: Path(data_dir)/f"posture{posture}_with_exo.npz",
    }
    real = {}
    meta_any = None
    for cond, fp in paths.items():
        if not fp.exists():
            raise SystemExit(f"Missing real data file: {fp}")
        z = np.load(fp, allow_pickle=True)
        X = z["X"].astype(np.float32).transpose(0,2,1)  # (N,C,T)
        real[cond] = X
        if meta_any is None:
            meta_any = dict(
                ch_names=list(z["ch_names"]),
                fs=float(z["fs"]),
                scale_min=z["scale_min"].astype(np.float32),
                scale_range=z["scale_range"].astype(np.float32),
            )
    return real, meta_any

# ----------------- Generator (match training names) -----------------
class CBN1d(nn.Module):
    def __init__(self, nf, ncls):
        super().__init__()
        self.bn = nn.BatchNorm1d(nf, affine=False)
        self.emb = nn.Embedding(ncls, nf*2)
        nn.init.ones_(self.emb.weight[:, :nf]); nn.init.zeros_(self.emb.weight[:, nf:])
    def forward(self, x, y):
        h = self.bn(x); g,b = self.emb(y).chunk(2, dim=1)
        return g.unsqueeze(-1)*h + b.unsqueeze(-1)

class UpsampleBlock(nn.Module):
    def __init__(self, ci, co, ncls):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(ci, co, 3, 1, 1)
        self.cbn = CBN1d(co, ncls)
    def forward(self, x, y):
        return F.relu(self.cbn(self.conv(self.up(x)), y), inplace=True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=2):
        super().__init__()
        self.nd, self.nc = noise_dim, num_classes
        self.init_ch, self.init_len = 512, 24
        self.proj = nn.Linear(noise_dim+num_classes, self.init_ch*self.init_len)
        self.up1 = UpsampleBlock(512,256,num_classes)
        self.up2 = UpsampleBlock(256,128,num_classes)
        self.up3 = UpsampleBlock(128, 64,num_classes)
        self.up4 = UpsampleBlock(64,  32,num_classes)
        self.up5 = UpsampleBlock(32,  16,num_classes)
        self.to_out = nn.Conv1d(16, C, 3, 1, 1)
        self.out_act = nn.Sigmoid()
    def forward(self, z, labels):
        oh = F.one_hot(labels, num_classes=self.nc).float()
        h = self.proj(torch.cat([z,oh],1)).view(-1, self.init_ch, self.init_len)
        h = self.up1(h,labels); h=self.up2(h,labels); h=self.up3(h,labels); h=self.up4(h,labels); h=self.up5(h,labels)
        return self.out_act(self.to_out(h))

def safe_load_generator(G: nn.Module, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    try:
        G.load_state_dict(sd, strict=True)
    except RuntimeError:
        # allow older names like u1..u5 or out.*
        new_sd={}
        for k,v in sd.items():
            nk = k.replace("u1.","up1.").replace("u2.","up2.").replace("u3.","up3.").replace("u4.","up4.").replace("u5.","up5.").replace("out.","to_out.")
            new_sd[nk]=v
        G.load_state_dict(new_sd, strict=False)

# ----------------- Synthesis -----------------
@torch.no_grad()
def synthesize_for_posture(runs_root, posture, noise_dim, n_per_cond, device):
    gpath = Path(runs_root)/f"posture{posture}"/f"CGAN_generator_posture{posture}_best.pth"
    if not gpath.exists():
        gpath = Path(runs_root)/f"posture{posture}"/f"CGAN_generator_posture{posture}_last.pth"
    G = Generator(noise_dim=noise_dim, num_classes=NUM_COND).to(device).eval()
    safe_load_generator(G, gpath, device)

    fakes = {}
    for cond in (0,1):
        z = torch.randn(n_per_cond, noise_dim, device=device)
        y = torch.full((n_per_cond,), cond, dtype=torch.long, device=device)
        x = G(z,y).cpu().numpy()   # (N,C,T) in [0,1]
        fakes[cond] = x
    return fakes

# ----------------- Features (NaN-safe) -----------------
def psd_features(X, n_bins=64, eps=1e-6):
    N,C,T = X.shape
    F = np.fft.rfft(X.astype(np.float32), axis=2)
    P = (F.real**2 + F.imag**2) / (T/2.0 + 1e-8)
    P = np.log(P + eps)
    Fbins = P.shape[2]
    if n_bins < Fbins:
        pool = Fbins // n_bins
        P = P[:, :, :pool*n_bins].reshape(N, C, n_bins, pool).mean(-1)
    else:
        pad = n_bins - Fbins
        P = np.pad(P, ((0,0),(0,0),(0,max(0,pad))), mode="edge")[:, :, :n_bins]
    feats = P.reshape(N, C*n_bins)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ----------------- Metrics -----------------
def discriminative_metrics(X_real, X_gen, y_posture, out_csv):
    Fr, Fg = psd_features(X_real), psd_features(X_gen)
    X = np.vstack([Fr,Fg])
    y = np.hstack([np.zeros(len(Fr), dtype=np.int64), np.ones(len(Fg), dtype=np.int64)])

    scaler = StandardScaler()
    Xs = np.nan_to_num(scaler.fit_transform(X), 0.0, 0.0, 0.0)

    # global
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, stratify=y, random_state=SEED)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]; pred = (prob>0.5).astype(int)
    acc = accuracy_score(yte, pred)
    try: auc = roc_auc_score(yte, prob)
    except ValueError: auc = float("nan")
    rows=[dict(level="global", posture=0, acc=acc, auc=auc)]

    # per posture
    posts = np.unique(y_posture)
    for p in posts:
        m = (y_posture==p)
        Xp, yp = Xs[m], y[m]
        if len(np.unique(yp)) < 2:  # avoid single-class split
            continue
        Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.3, stratify=yp, random_state=SEED)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:,1]; pred=(prob>0.5).astype(int)
        acc = accuracy_score(yte, pred)
        try: auc = roc_auc_score(yte, prob)
        except ValueError: auc = float("nan")
        rows.append(dict(level="posture", posture=int(p), acc=acc, auc=auc))

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def predictive_scores(X_real, X_gen, y_posture, out_csv, ch_names=None,
                      target_name="", default_target_idx=13):
    """
    FIXED: splits y_posture into y_real and y_gen so masks match array lengths.
    """
    # pick target channel
    if target_name and ch_names is not None and target_name in list(ch_names):
        target_idx = list(ch_names).index(target_name)
    else:
        target_idx = default_target_idx

    def make_xy(X):
        Xf = X[:, np.arange(C) != target_idx, :].transpose(0, 2, 1).reshape(len(X), -1)
        Y  = X[:, target_idx, :].reshape(len(X), -1)
        return Xf.astype(np.float32), Y.astype(np.float32)

    # split posture labels
    yr, yg = _split_y_posture(y_posture, n_real=len(X_real))

    rows = []

    # ---------- global ----------
    # TSTR: train on GEN, test on REAL
    sX, sY = StandardScaler(), StandardScaler()
    Xtr, Ytr = make_xy(X_gen); Xte, Yte = make_xy(X_real)
    Xtr = np.nan_to_num(sX.fit_transform(Xtr), 0, 0, 0); Ytr = np.nan_to_num(sY.fit_transform(Ytr), 0, 0, 0)
    Xte = np.nan_to_num(sX.transform(Xte), 0, 0, 0);     Yte = np.nan_to_num(sY.transform(Yte), 0, 0, 0)
    reg = Ridge(alpha=1.0).fit(Xtr, Ytr); Yhat = reg.predict(Xte)
    rows.append(dict(level="global", posture=0, split="TSTR",
                     rmse=float(np.sqrt(mean_squared_error(Yte, Yhat))),
                     r2=float(r2_score(Yte, Yhat))))

    # TRTS: train on REAL, test on GEN
    sX, sY = StandardScaler(), StandardScaler()
    Xtr, Ytr = make_xy(X_real); Xte, Yte = make_xy(X_gen)
    Xtr = np.nan_to_num(sX.fit_transform(Xtr), 0, 0, 0); Ytr = np.nan_to_num(sY.fit_transform(Ytr), 0, 0, 0)
    Xte = np.nan_to_num(sX.transform(Xte), 0, 0, 0);     Yte = np.nan_to_num(sY.transform(Yte), 0, 0, 0)
    reg = Ridge(alpha=1.0).fit(Xtr, Ytr); Yhat = reg.predict(Xte)
    rows.append(dict(level="global", posture=0, split="TRTS",
                     rmse=float(np.sqrt(mean_squared_error(Yte, Yhat))),
                     r2=float(r2_score(Yte, Yhat))))

    # ---------- per posture ----------
    posts = np.unique(np.concatenate([yr, yg]))
    for p in posts:
        mr = (yr == p)
        mg = (yg == p)
        if mr.sum() < 5 or mg.sum() < 5:
            continue

        # TSTR (train GEN posture p, test REAL posture p)
        sX, sY = StandardScaler(), StandardScaler()
        Xtr, Ytr = make_xy(X_gen[mg]); Xte, Yte = make_xy(X_real[mr])
        Xtr = np.nan_to_num(sX.fit_transform(Xtr), 0, 0, 0); Ytr = np.nan_to_num(sY.fit_transform(Ytr), 0, 0, 0)
        Xte = np.nan_to_num(sX.transform(Xte), 0, 0, 0);     Yte = np.nan_to_num(sY.transform(Yte), 0, 0, 0)
        reg = Ridge(alpha=1.0).fit(Xtr, Ytr); Yhat = reg.predict(Xte)
        rows.append(dict(level="posture", posture=int(p), split="TSTR",
                         rmse=float(np.sqrt(mean_squared_error(Yte, Yhat))),
                         r2=float(r2_score(Yte, Yhat))))

        # TRTS (train REAL posture p, test GEN posture p)
        sX, sY = StandardScaler(), StandardScaler()
        Xtr, Ytr = make_xy(X_real[mr]); Xte, Yte = make_xy(X_gen[mg])
        Xtr = np.nan_to_num(sX.fit_transform(Xtr), 0, 0, 0); Ytr = np.nan_to_num(sY.fit_transform(Ytr), 0, 0, 0)
        Xte = np.nan_to_num(sX.transform(Xte), 0, 0, 0);     Yte = np.nan_to_num(sY.transform(Yte), 0, 0, 0)
        reg = Ridge(alpha=1.0).fit(Xtr, Ytr); Yhat = reg.predict(Xte)
        rows.append(dict(level="posture", posture=int(p), split="TRTS",
                         rmse=float(np.sqrt(mean_squared_error(Yte, Yhat))),
                         r2=float(r2_score(Yte, Yhat))))

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def stats_similarity(X_real, X_gen, y_posture, out_csv):
    """
    FIXED: splits y_posture into y_real and y_gen for posture-wise comparisons.
    """
    def psd_avg(X):
        F = np.fft.rfft(X, axis=2); P = (F.real**2 + F.imag**2)
        return P.mean(axis=0)  # (C,F)

    def acf_avg(X, max_lag=128):
        Xc = X - X.mean(axis=2, keepdims=True)
        acfs=[]
        for ch in range(C):
            xi = Xc[:,ch,:]
            ac = np.array([np.mean(xi[:,:-k]*xi[:,k:]) for k in range(1,max_lag+1)], dtype=np.float32)
            acfs.append(ac)
        return np.stack(acfs,0)  # (C,L)

    def coh_avg(X):
        pairs=[(0,13),(6,7),(9,10),(1,12)]
        F = np.fft.rfft(X, axis=2); out=[]
        for i,j in pairs:
            A=F[:,i,:]; B=F[:,j,:]
            num=np.sqrt((A*B.conj()).real**2 + (A*B.conj()).imag**2)
            den=np.sqrt((A.real**2 + A.imag**2)*(B.real**2 + B.imag**2) + 1e-8)
            out.append((num/den).mean(axis=0))
        return np.stack(out,0)

    yr, yg = _split_y_posture(y_posture, n_real=len(X_real))

    rows=[]
    rows.append(dict(level="global", posture=0,
        psd_l1=float(np.mean(np.abs(psd_avg(X_real)-psd_avg(X_gen)))),
        acf_l1=float(np.mean(np.abs(acf_avg(X_real)-acf_avg(X_gen)))),
        coh_l1=float(np.mean(np.abs(coh_avg(X_real)-coh_avg(X_gen))))))

    posts = np.unique(np.concatenate([yr, yg]))
    for p in posts:
        mr = (yr == p); mg = (yg == p)
        if mr.sum() < 5 or mg.sum() < 5:
            continue
        rows.append(dict(level="posture", posture=int(p),
            psd_l1=float(np.mean(np.abs(psd_avg(X_real[mr])-psd_avg(X_gen[mg])))),
            acf_l1=float(np.mean(np.abs(acf_avg(X_real[mr])-acf_avg(X_gen[mg])))),
            coh_l1=float(np.mean(np.abs(coh_avg(X_real[mr])-coh_avg(X_gen[mg]))))))
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---- TSNE back-compat ----
def tsne_fit_transform(X, perpl, tsne_iter, seed):
    try:
        tsne = TSNE(n_components=2, perplexity=perpl, n_iter=tsne_iter,
                    learning_rate="auto", init="pca", random_state=seed, verbose=0)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perpl,
                    learning_rate=200.0, init="pca", random_state=seed)
    return tsne.fit_transform(X)

def scatter_plots(X_real, X_gen, y_posture, out_dir, perpl=30.0, tsne_iter=1000, seed=123):
    # features
    Fr, Fg = psd_features(X_real), psd_features(X_gen)
    X = np.vstack([Fr, Fg])

    # labels
    n_real = len(Fr)
    assert len(y_posture) == n_real + len(Fg), \
        f"y_posture len={len(y_posture)} but n_real+len(Fg)={n_real+len(Fg)}"
    y = y_posture  # already concatenated [real; gen], do NOT duplicate

    # source flags for real vs gen
    src = np.hstack([np.zeros(n_real), np.ones(len(Fg))])

    # PCA 2D
    pca = PCA(n_components=2, svd_solver="full", random_state=seed)
    Zp = pca.fit_transform(np.nan_to_num(X, 0, 0, 0))
    fig, ax = plt.subplots(figsize=(7, 6))
    m = ax.scatter(Zp[:, 0], Zp[:, 1], c=y, cmap="tab10", s=8, alpha=0.7, edgecolors="none")
    plt.colorbar(m, ax=ax, label="posture"); ax.set_title("PCA (color=posture)")
    plt.savefig(Path(out_dir) / "pca_scatter.png", dpi=150, bbox_inches="tight"); plt.close()

    # t-SNE on PCA50
    pca50 = PCA(n_components=min(50, X.shape[1]-1), svd_solver="full", random_state=seed)
    X50 = pca50.fit_transform(np.nan_to_num(X, 0, 0, 0))
    try:
        tsne = TSNE(n_components=2, perplexity=perpl, n_iter=tsne_iter,
                    learning_rate="auto", init="pca", random_state=seed, verbose=0)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perpl, learning_rate=200.0, init="pca", random_state=seed)
    Z = tsne.fit_transform(X50)

    fig, ax = plt.subplots(figsize=(7, 6))
    m = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8, alpha=0.7, edgecolors="none")
    plt.colorbar(m, ax=ax, label="posture"); ax.set_title("t-SNE (color=posture)")
    plt.savefig(Path(out_dir) / "tsne_scatter.png", dpi=150, bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[src == 0, 0], Z[src == 0, 1], c="C0", s=8, alpha=0.6, label="real")
    ax.scatter(Z[src == 1, 0], Z[src == 1, 1], c="C3", s=8, alpha=0.6, label="gen")
    ax.legend(); ax.set_title("t-SNE (real vs gen)")
    plt.savefig(Path(out_dir) / "tsne_real_gen.png", dpi=150, bbox_inches="tight"); plt.close()


# ----------------- main -----------------
def main():
    set_seed(SEED)
    save_root = Path(SAVE_ROOT); save_root.mkdir(parents=True, exist_ok=True)

    # global buffers
    Xr_glob, Xg_glob, yp_glob = [], [], []
    ch_names_ref = None

    for p in POSTURES:
        # real
        real_dict, meta = load_real_posture(DATA_DIR, p)
        if ch_names_ref is None: ch_names_ref = meta["ch_names"]

        # how many to synthesize per condition
        if isinstance(SAMPLES_PER_COND, str) and SAMPLES_PER_COND.lower()=="match":
            n0 = real_dict[0].shape[0]; n1 = real_dict[1].shape[0]
            n_synth = min(n0, n1)
        else:
            n_synth = int(SAMPLES_PER_COND)

        # synth
        fakes = synthesize_for_posture(RUNS_ROOT, p, NOISE_DIM, n_synth, DEVICE)

        # balance reals to min across conds and synths (fair posture-level eval)
        n = min(real_dict[0].shape[0], real_dict[1].shape[0], fakes[0].shape[0], fakes[1].shape[0])
        R = np.concatenate([real_dict[0][:n], real_dict[1][:n]], 0)
        G = np.concatenate([fakes[0][:n],     fakes[1][:n]],     0)
        yp = np.full((R.shape[0]+G.shape[0],), p, dtype=np.int64)

        # per-posture outputs
        out_dir_p = save_root/f"posture{p}"; out_dir_p.mkdir(parents=True, exist_ok=True)
        discriminative_metrics(R, G, yp, out_dir_p/"metrics_discriminative.csv")
        predictive_scores  (R, G, yp, out_dir_p/"metrics_predictive.csv", ch_names=ch_names_ref)
        stats_similarity   (R, G, yp, out_dir_p/"metrics_stats.csv")

        # collect for global
        Xr_glob.append(R); Xg_glob.append(G); yp_glob.append(np.full(R.shape[0]+G.shape[0], p, dtype=np.int64))

    # global eval
    Xr_all = np.concatenate(Xr_glob, 0)
    Xg_all = np.concatenate(Xg_glob, 0)
    yp_all = np.concatenate(yp_glob, 0)

    out_dir_g = save_root/"global"; out_dir_g.mkdir(parents=True, exist_ok=True)
    discriminative_metrics(Xr_all, Xg_all, yp_all, out_dir_g/"metrics_discriminative.csv")
    predictive_scores  (Xr_all, Xg_all, yp_all, out_dir_g/"metrics_predictive.csv", ch_names=ch_names_ref)
    stats_similarity   (Xr_all, Xg_all, yp_all, out_dir_g/"metrics_stats.csv")
    scatter_plots      (Xr_all, Xg_all, yp_all, out_dir_g)

    print(f"✅ Saved all evaluations under: {save_root}")

if __name__ == "__main__":
    main()
