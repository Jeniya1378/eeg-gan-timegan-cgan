#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Evaluation for CGAN EEG (14x768). Includes discriminative, predictive (TSTR/TRTS),
# statistical similarity (PSD/ACF/Coherence), PCA/t-SNE. Safe for older sklearn TSNE.

import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------- args ----------------
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="./preprocessed")
    p.add_argument("--runs-root", type=str, default="./cgan_runs")
    p.add_argument("--save-root", type=str, default="./cgan_eval")
    p.add_argument("--condition", type=str, default="both", choices=["both","with_exo","no_exo"])
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--samples-per-posture", type=int, default=400)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--tsne-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

# -------------- data / model --------------
NUM_CLASSES = 9
C, T = 14, 768
def set_seed(s): np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_condition_dataset(data_dir: str, condition: str):
    files = sorted(glob(str(Path(data_dir) / f"posture*_{condition}.npz")))
    if not files: raise SystemExit(f"No files posture*_{condition}.npz in {data_dir}")
    Xs, ys, meta = [], [], {}
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        X = z["X"].astype(np.float32).transpose(0,2,1)  # (N,C,T) in [0,1]
        posture = int(z["posture"])
        Xs.append(X); ys.append(np.full((X.shape[0],), posture, dtype=np.int64))
        meta[posture] = dict(
            scale_min=z["scale_min"].astype(np.float32),
            scale_range=z["scale_range"].astype(np.float32),
            ch_names=z["ch_names"], fs=float(z["fs"])
        )
    X = np.concatenate(Xs,0); y = np.concatenate(ys,0); p = np.random.permutation(len(y))
    return X[p], y[p], meta

class CBN1d(nn.Module):
    def __init__(self, nf, ncls):
        super().__init__()
        self.bn = nn.BatchNorm1d(nf, affine=False)
        self.emb = nn.Embedding(ncls, nf*2)
        nn.init.ones_(self.emb.weight[:, :nf]); nn.init.zeros_(self.emb.weight[:, nf:])
    def forward(self,x,y):
        h=self.bn(x); g,b=self.emb(y).chunk(2,dim=1)
        return g.unsqueeze(-1)*h + b.unsqueeze(-1)

class UpsampleBlock(nn.Module):
    def __init__(self, ci, co, ncls):
        super().__init__(); self.up=nn.Upsample(scale_factor=2,mode="nearest"); self.conv=nn.Conv1d(ci,co,3,1,1); self.cbn=CBN1d(co,ncls)
    def forward(self,x,y): return F.relu(self.cbn(self.conv(self.up(x)),y),True)

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=9):
        super().__init__()
        self.noise_dim=noise_dim; self.num_classes=num_classes; self.init_ch=512; self.init_len=24
        self.proj=nn.Linear(noise_dim+num_classes, self.init_ch*self.init_len)
        self.up1=UpsampleBlock(512,256,num_classes); self.up2=UpsampleBlock(256,128,num_classes)
        self.up3=UpsampleBlock(128,64,num_classes);  self.up4=UpsampleBlock(64,32,num_classes); self.up5=UpsampleBlock(32,16,num_classes)
        self.to_out=nn.Conv1d(16, C, 3,1,1); self.out_act=nn.Sigmoid()
    def forward(self,z,labels):
        oh=F.one_hot(labels,num_classes=self.num_classes).float()
        h=self.proj(torch.cat([z,oh],1)).view(-1,self.init_ch,self.init_len)
        for blk in [self.up1,self.up2,self.up3,self.up4,self.up5]: h=blk(h,labels)
        return self.out_act(self.to_out(h))

def safe_load_generator(G, path, device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    try:
        G.load_state_dict(sd, strict=True)
    except RuntimeError:
        new_sd={}; 
        for k,v in sd.items():
            nk=k.replace("u1.","up1.").replace("u2.","up2.").replace("u3.","up3.").replace("u4.","up4.").replace("u5.","up5.")
            nk=nk.replace("out.","to_out.")
            new_sd[nk]=v
        G.load_state_dict(new_sd, strict=False)

@torch.no_grad()
def synthesize(meta, runs_root, condition, n_per_posture, noise_dim, device):
    G=Generator(noise_dim=noise_dim,num_classes=NUM_CLASSES).to(device).eval()
    gpath=Path(runs_root)/condition/f"CGAN_generator_{condition}_best.pth"
    if not gpath.exists(): gpath=Path(runs_root)/condition/f"CGAN_generator_{condition}_last.pth"
    safe_load_generator(G, gpath, device)
    outs, labs = [], []
    for posture in range(1, NUM_CLASSES+1):
        z=torch.randn(n_per_posture, noise_dim, device=device)
        y=torch.full((n_per_posture,), posture-1, dtype=torch.long, device=device)
        outs.append(G(z,y).cpu().numpy()); labs.append(np.full(n_per_posture, posture, dtype=np.int64))
    return np.concatenate(outs,0), np.concatenate(labs,0)

# -------------- features --------------
def psd_features(X, n_bins=64, eps=1e-6):
    N,C,T=X.shape
    F=np.fft.rfft(X.astype(np.float32),axis=2)
    P=(F.real**2 + F.imag**2)/(T/2.0 + 1e-8)
    P=np.log(P + eps)
    Fbins=P.shape[2]
    if n_bins < Fbins:
        pool=Fbins//n_bins; P=P[:, :, :pool*n_bins].reshape(N,C,n_bins,pool).mean(-1)
    else:
        pad=n_bins-Fbins; P=np.pad(P,((0,0),(0,0),(0,max(0,pad))),mode="edge")[:, :, :n_bins]
    feats=P.reshape(N, C*n_bins)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# -------------- metrics --------------
def discriminative_metrics(Xr, Xg, yr, yg, out_csv):
    Fr,Fg=psd_features(Xr), psd_features(Xg)
    X=np.vstack([Fr,Fg]); y=np.hstack([np.zeros(len(Fr),dtype=np.int64), np.ones(len(Fg),dtype=np.int64)])
    y_post=np.hstack([yr,yg])
    scaler=StandardScaler(); Xs=np.nan_to_num(scaler.fit_transform(X),0,0,0)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.3,stratify=y,random_state=123)
    clf=LogisticRegression(max_iter=1000).fit(Xtr,ytr); prob=clf.predict_proba(Xte)[:,1]; pred=(prob>0.5).astype(int)
    acc=accuracy_score(yte,pred); 
    try: auc=roc_auc_score(yte,prob)
    except ValueError: auc=float("nan")
    rows=[dict(level="global", posture=0, acc=acc, auc=auc)]
    for p in range(1, NUM_CLASSES+1):
        m=(y_post==p); 
        if m.sum()<20: continue
        Xp, yp = Xs[m], y[m]
        Xtr,Xte,ytr,yte=train_test_split(Xp,yp,test_size=0.3,stratify=yp,random_state=123)
        clf=LogisticRegression(max_iter=1000).fit(Xtr,ytr); prob=clf.predict_proba(Xte)[:,1]; pred=(prob>0.5).astype(int)
        acc=accuracy_score(yte,pred)
        try: auc=roc_auc_score(yte,prob)
        except ValueError: auc=float("nan")
        rows.append(dict(level="posture", posture=p, acc=acc, auc=auc))
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def predictive_scores(Xr, Xg, yr, yg, out_csv, target_idx=13):
    def make_xy(X):
        Xf=X[:, np.arange(C)!=target_idx, :].transpose(0,2,1).reshape(len(X), -1)
        Y =X[:, target_idx, :].reshape(len(X), -1)
        return Xf.astype(np.float32), Y.astype(np.float32)
    rows=[]
    # TSTR
    sX,sY=StandardScaler(),StandardScaler()
    Xtr,Ytr=make_xy(Xg); Xte,Yte=make_xy(Xr)
    Xtr=np.nan_to_num(sX.fit_transform(Xtr),0,0,0); Ytr=np.nan_to_num(sY.fit_transform(Ytr),0,0,0)
    Xte=np.nan_to_num(sX.transform(Xte),0,0,0);     Yte=np.nan_to_num(sY.transform(Yte),0,0,0)
    reg=Ridge(alpha=1.0).fit(Xtr,Ytr); Yhat=reg.predict(Xte)
    rows.append(dict(level="global", posture=0, split="TSTR",
                     rmse=float(np.sqrt(mean_squared_error(Yte,Yhat))), r2=float(r2_score(Yte,Yhat))))
    # TRTS
    sX,sY=StandardScaler(),StandardScaler()
    Xtr,Ytr=make_xy(Xr); Xte,Yte=make_xy(Xg)
    Xtr=np.nan_to_num(sX.fit_transform(Xtr),0,0,0); Ytr=np.nan_to_num(sY.fit_transform(Ytr),0,0,0)
    Xte=np.nan_to_num(sX.transform(Xte),0,0,0);     Yte=np.nan_to_num(sY.transform(Yte),0,0,0)
    reg=Ridge(alpha=1.0).fit(Xtr,Ytr); Yhat=reg.predict(Xte)
    rows.append(dict(level="global", posture=0, split="TRTS",
                     rmse=float(np.sqrt(mean_squared_error(Yte,Yhat))), r2=float(r2_score(Yte,Yhat))))
    # per posture
    for p in range(1, NUM_CLASSES+1):
        mr,mg=(yr==p),(yg==p)
        if mr.sum()<10 or mg.sum()<10: continue
        sX,sY=StandardScaler(),StandardScaler()
        Xtr,Ytr=make_xy(Xg[mg]); Xte,Yte=make_xy(Xr[mr])
        Xtr=np.nan_to_num(sX.fit_transform(Xtr),0,0,0); Ytr=np.nan_to_num(sY.fit_transform(Ytr),0,0,0)
        Xte=np.nan_to_num(sX.transform(Xte),0,0,0);     Yte=np.nan_to_num(sY.transform(Yte),0,0,0)
        reg=Ridge(alpha=1.0).fit(Xtr,Ytr); Yhat=reg.predict(Xte)
        rows.append(dict(level="posture", posture=p, split="TSTR",
                         rmse=float(np.sqrt(mean_squared_error(Yte,Yhat))), r2=float(r2_score(Yte,Yhat))))
        sX,sY=StandardScaler(),StandardScaler()
        Xtr,Ytr=make_xy(Xr[mr]); Xte,Yte=make_xy(Xg[mg])
        Xtr=np.nan_to_num(sX.fit_transform(Xtr),0,0,0); Ytr=np.nan_to_num(sY.fit_transform(Ytr),0,0,0)
        Xte=np.nan_to_num(sX.transform(Xte),0,0,0);     Yte=np.nan_to_num(sY.transform(Yte),0,0,0)
        reg=Ridge(alpha=1.0).fit(Xtr,Ytr); Yhat=reg.predict(Xte)
        rows.append(dict(level="posture", posture=p, split="TRTS",
                         rmse=float(np.sqrt(mean_squared_error(Yte,Yhat))), r2=float(r2_score(Yte,Yhat))))
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# >>> This was missing in your run <<<
def stats_similarity(Xr, Xg, yr, yg, out_csv):
    def psd_avg(X):
        F=np.fft.rfft(X,axis=2); P=(F.real**2 + F.imag**2); return P.mean(axis=0)   # (C,F)
    def acf_avg(X, max_lag=128):
        Xc=X - X.mean(axis=2,keepdims=True); ac=[]
        for ch in range(C):
            xi=Xc[:,ch,:]; ac.append(np.array([np.mean(xi[:,:-k]*xi[:,k:]) for k in range(1,max_lag+1)],dtype=np.float32))
        return np.stack(ac,0)                                                       # (C,L)
    def coh_avg(X):
        pairs=[(0,13),(6,7),(9,10),(1,12)]; out=[]; F=np.fft.rfft(X,axis=2)
        for i,j in pairs:
            A=F[:,i,:]; B=F[:,j,:]
            num=np.sqrt((A*B.conj()).real**2 + (A*B.conj()).imag**2)
            den=np.sqrt((A.real**2 + A.imag**2)*(B.real**2 + B.imag**2) + 1e-8)
            out.append((num/den).mean(axis=0))
        return np.stack(out,0)                                                      # (P,F)
    rows=[]
    rows.append(dict(level="global", posture=0,
        psd_l1=float(np.mean(np.abs(psd_avg(Xr)-psd_avg(Xg)))),
        acf_l1=float(np.mean(np.abs(acf_avg(Xr)-acf_avg(Xg)))),
        coh_l1=float(np.mean(np.abs(coh_avg(Xr)-coh_avg(Xg))))
    ))
    for p in range(1, NUM_CLASSES+1):
        mr,mg=(yr==p),(yg==p)
        if mr.sum()<10 or mg.sum()<10: continue
        rows.append(dict(level="posture", posture=p,
            psd_l1=float(np.mean(np.abs(psd_avg(Xr[mr])-psd_avg(Xg[mg])))),
            acf_l1=float(np.mean(np.abs(acf_avg(Xr[mr])-acf_avg(Xg[mg])))),
            coh_l1=float(np.mean(np.abs(coh_avg(Xr[mr])-coh_avg(Xg[mg]))))))
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# ---- TSNE with old/new sklearn support ----
def tsne_fit_transform(X, perpl, tsne_iter, seed):
    try:
        tsne=TSNE(n_components=2, perplexity=perpl, n_iter=tsne_iter, learning_rate="auto", init="pca", random_state=seed, verbose=0)
    except TypeError:
        tsne=TSNE(n_components=2, perplexity=perpl, learning_rate=200.0, init="pca", random_state=seed)
    return tsne.fit_transform(X)

def scatter_plots(Xr, Xg, yr, yg, out_dir, perpl=30.0, tsne_iter=1000, seed=123):
    Fr,Fg=psd_features(Xr), psd_features(Xg); X=np.vstack([Fr,Fg]); y=np.hstack([yr,yg])
    src=np.hstack([np.zeros(len(Fr)), np.ones(len(Fg))])
    pca=PCA(n_components=2, svd_solver="full", random_state=seed); Zp=pca.fit_transform(np.nan_to_num(X,0,0,0))
    fig,ax=plt.subplots(figsize=(7,6)); m=ax.scatter(Zp[:,0],Zp[:,1],c=y,cmap="tab10",s=10,alpha=0.7,edgecolors="none")
    plt.colorbar(m,ax=ax,label="posture"); ax.set_title("PCA (color=posture)"); plt.savefig(Path(out_dir)/"pca_scatter.png",dpi=150,bbox_inches="tight"); plt.close()
    pca50=PCA(n_components=min(50,X.shape[1]-1),svd_solver="full",random_state=seed); X50=pca50.fit_transform(np.nan_to_num(X,0,0,0))
    Z=tsne_fit_transform(X50, perpl, tsne_iter, seed)
    fig,ax=plt.subplots(figsize=(7,6)); m=ax.scatter(Z[:,0],Z[:,1],c=y,cmap="tab10",s=8,alpha=0.7,edgecolors="none")
    plt.colorbar(m,ax=ax,label="posture"); ax.set_title("t-SNE (color=posture)"); plt.savefig(Path(out_dir)/"tsne_scatter.png",dpi=150,bbox_inches="tight"); plt.close()
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(Z[src==0,0],Z[src==0,1],c="C0",s=8,alpha=0.6,label="real"); ax.scatter(Z[src==1,0],Z[src==1,1],c="C3",s=8,alpha=0.6,label="gen")
    ax.legend(); ax.set_title("t-SNE (real vs gen)"); plt.savefig(Path(out_dir)/"tsne_real_gen.png",dpi=150,bbox_inches="tight"); plt.close()

# -------------- main --------------
def main():
    args=get_args(); set_seed(args.seed)
    conditions=["with_exo","no_exo"] if args.condition=="both" else [args.condition]
    for condition in conditions:
        Xr,yr,meta=load_condition_dataset(args.data_dir, condition)
        npp=args.samples_per_posture; keep=[]
        for p in range(1, NUM_CLASSES+1):
            idx=np.where(yr==p)[0]; 
            if len(idx): 
                np.random.shuffle(idx); keep.append(idx[:min(npp,len(idx))])
        if keep: keep=np.concatenate(keep); Xr=Xr[keep]; yr=yr[keep]
        Xg,yg=synthesize(meta, args.runs_root, condition, n_per_posture=npp, noise_dim=args.noise_dim, device=args.device)
        out_dir=Path(args.save_root)/condition; out_dir.mkdir(parents=True, exist_ok=True)
        discriminative_metrics(Xr,Xg,yr,yg,out_dir/"metrics_discriminative.csv")
        predictive_scores  (Xr,Xg,yr,yg,out_dir/"metrics_predictive.csv")
        stats_similarity   (Xr,Xg,yr,yg,out_dir/"metrics_stats.csv")
        scatter_plots      (Xr,Xg,yr,yg,out_dir,args.tsne_perplexity,args.tsne_iter,args.seed)
        print(f"[{condition}] ✅ Saved results to {out_dir}")

if __name__ == "__main__":
    main()
