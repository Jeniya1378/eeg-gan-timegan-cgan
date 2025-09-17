#!/usr/bin/env python3
"""
main.py — reads config (JSON or YAML) and launches TimeGAN training
for every NPZ under data_dir (one model per posture×condition).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import train_timegan as tt

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as e:
            raise SystemExit("YAML config requested but PyYAML not installed. "
                             "Install with `pip install pyyaml` or use JSON.") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="timegan_config.json",
                    help="Path to config file (JSON or YAML)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    data_dir = Path(cfg.get("data_dir", "./preprocessed"))
    out_root = Path(cfg.get("out_dir", "./timegan_runs")); out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("posture*_*.npz"))
    if not files:
        raise SystemExit(f"No NPZs found in {data_dir}. Did you run preprocessing?")

    device = tt.device_autoselect()
    print(f"Using device: {device}")
    print(f"Found {len(files)} datasets → training {len(files)} models.")

    betas = (float(cfg.get("beta1", 0.5)), float(cfg.get("beta2", 0.9)))

    # loop over datasets
    for fp in files:
        run_dir = out_root / fp.stem
        print(f"\n=== Training {fp.name} → {run_dir} ===")
        tt.train_single_npz(
            npz_path=fp,
            out_dir=run_dir,
            batch_size=int(cfg.get("batch_size", 64)),
            ae_epochs=int(cfg.get("ae_epochs", 120)),
            sup_epochs=int(cfg.get("sup_epochs", 150)),
            gan_steps=int(cfg.get("gan_steps", 8000)),
            lr_g=float(cfg.get("lr_g", 1e-3)),
            lr_d=float(cfg.get("lr_d", 2e-4)),
            betas=betas,
            alpha_sup=float(cfg.get("alpha_sup", 5.0)),
            beta_rec=float(cfg.get("beta_rec", 0.2)),
            label_smooth=float(cfg.get("label_smooth", 0.2)),
            inst_noise_start=float(cfg.get("inst_noise_start", 0.3)),
            inst_noise_end=float(cfg.get("inst_noise_end", 0.1)),
            grad_clip=float(cfg.get("grad_clip", 0.5)),
            layers=int(cfg.get("layers", 1)),
            dropout=float(cfg.get("dropout", 0.2)),
            seed=int(cfg.get("seed", 42)),
            r1_gamma=float(cfg.get("r1_gamma", 1.0)),
            d_min_acc=float(cfg.get("d_min_acc", 0.45)),
            d_max_acc=float(cfg.get("d_max_acc", 0.60)),
            gamma_cov=float(cfg.get("gamma_cov", 0.05)),
            gamma_acf=float(cfg.get("gamma_acf", 0.05)),
            acf_max_lag=int(cfg.get("acf_max_lag", 64)),
            device=device
        )

    print("\nAll models trained. Checkpoints, logs, and synthetic data are under:", out_root)

if __name__ == "__main__":
    main()
