#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate long synthetic EEG from trained TimeGAN checkpoints.

It scans:  runs_dir/posture{1..9}_{with_exo|no_exo}/
Finds:     ckpt_best.pt (fallback: ckpt_latest.pt)
Loads:     model state + meta (z_dim, h_dim), and real NPZ to infer x_dim & fs
Outputs:   synthetic_long.npz (or synthetic_len{T}.npz) per run

Examples:
  python generate_long_synth.py --runs_dir ./timegan_runs --real_dir ./preprocessed --gen_seconds 60
  python generate_long_synth.py --runs_dir ./timegan_runs --real_dir ./preprocessed --gen_len 8192 --n 256
  python generate_long_synth.py --gen_seconds 30 --denorm

Notes:
- Training saves ckpt_best.pt/ckpt_latest.pt + a default synthetic.npz after training. (training code)  # cites below
- Preprocessed NPZ stores fs, scale_min, scale_range for optional denorm. (preprocessing code)
"""

import argparse
from pathlib import Path
import re
import numpy as np
import torch

# Import your TimeGAN module
from timegan_model import TimeGAN  # architecture supports variable T at inference (GRU-based)

def device_autoselect():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample_noise(batch_size: int, seq_len: int, z_dim: int, device):
    # Same distribution used in training (uniform [0,1]) per train_timegan.sample_noise
    return torch.rand(batch_size, seq_len, z_dim, device=device)

def infer_posture_cond(run_dir_name: str):
    m = re.match(r"posture(\d+)_(with_exo|no_exo)$", run_dir_name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs_dir", type=str, default="./timegan_runs",
                    help="Root containing postureX_with_exo/postureX_no_exo folders")
    ap.add_argument("--real_dir", type=str, default="./preprocessed",
                    help="Where postureX_with_exo.npz etc. live (to get x_dim and fs)")
    ap.add_argument("--out_suffix", type=str, default="synthetic_long.npz",
                    help="Output filename written inside each run folder")
    ap.add_argument("--gen_seconds", type=float, default=None,
                    help="Length to generate in seconds (overrides gen_len).")
    ap.add_argument("--gen_len", type=int, default=None,
                    help="Length to generate in samples. If not set, uses training T.")
    ap.add_argument("--n", type=int, default=None,
                    help="Number of sequences to generate. Default: match real N.")
    ap.add_argument("--prefer_latest", action="store_true",
                    help="Use ckpt_latest.pt instead of ckpt_best.pt if both exist.")
    ap.add_argument("--denorm", action="store_true",
                    help="If set, invert scaling using scale_min/scale_range from real NPZ.")
    args = ap.parse_args()

    runs_root = Path(args.runs_dir)
    real_root = Path(args.real_dir)
    run_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir() and re.match(r"posture\d+_(with_exo|no_exo)$", p.name)]
    if not run_dirs:
        raise SystemExit(f"No run folders found under {runs_root}")

    device = device_autoselect()
    print(f"Using device: {device}")

    for rd in run_dirs:
        posture, cond = infer_posture_cond(rd.name)
        if posture is None:
            continue

        # Find checkpoint
        ckpt_best = rd / "ckpt_best.pt"
        ckpt_last = rd / "ckpt_latest.pt"
        ckpt = ckpt_last if args.prefer_latest and ckpt_last.exists() else (ckpt_best if ckpt_best.exists() else ckpt_last)
        if not ckpt or not ckpt.exists():
            print(f"[SKIP] {rd.name}: no checkpoint found.")
            continue

        # Load matching real NPZ to get x_dim (channels) and fs
        real_npz = real_root / f"posture{posture}_{cond}.npz"
        if not real_npz.exists():
            print(f"[SKIP] {rd.name}: real file missing: {real_npz}")
            continue
        real = np.load(real_npz)
        Xr = real["X"].astype(np.float32)  # (N, T, C)
        N_real, T_train, C = Xr.shape
        fs = float(real["fs"]) if "fs" in real.files else 128.0  # default if absent

        # Build model & load weights
        state = torch.load(ckpt, map_location="cpu")
        meta = state.get("meta", {})
        z_dim = int(meta.get("z_dim"))
        h_dim = int(meta.get("h_dim"))
        model = TimeGAN(x_dim=C, z_dim=z_dim, hidden_dim=h_dim, num_layers=1, dropout=0.2).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        # Decide output length (samples)
        if args.gen_seconds is not None:
            T_out = int(round(args.gen_seconds * fs))
        elif args.gen_len is not None:
            T_out = int(args.gen_len)
        else:
            T_out = int(T_train)  # default: match training sequence length

        # Decide number of sequences
        N_out = int(args.n) if args.n is not None else int(N_real)

        print(f"[{rd.name}] N_out={N_out}  T_out={T_out}  C={C}  z_dim={z_dim}  fs≈{fs:.2f}")

        with torch.no_grad():
            Z = sample_noise(N_out, T_out, z_dim, device)
            H0 = model.gen_latent(Z)
            H  = model.refine_latent(H0)
            Xh = model.decode(H).cpu().numpy().astype(np.float32)  # still in [0,1]-like scaled space

        # Optional denormalization back to original units
        if args.denorm and "scale_min" in real.files and "scale_range" in real.files:
            mn = real["scale_min"].astype(np.float32)
            rg = real["scale_range"].astype(np.float32)
            Xh = Xh * rg[None, None, :] + mn[None, None, :]

        out_fp = rd / (args.out_suffix if "{" not in args.out_suffix else args.out_suffix.format(T=T_out))
        np.savez_compressed(out_fp, X=Xh)
        print(f"[OK] wrote {out_fp}")

if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
