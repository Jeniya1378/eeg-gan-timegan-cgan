# EEG GANs: TimeGAN & CGAN for EEG Synthesis

This repo hosts two generative models for EEG sequence synthesis:
- **TimeGAN** (GRU-based embedder/recovery/generator/supervisor/discriminator)
- **CGAN** (conditional transformer-based generator/discriminator)

**Use case:** augment small, posture-conditioned EEG datasets (e.g., 14-ch Emotiv EPOC+) to study mental fatigue / workload in construction tasks.

## Repo Structure
- `timegan/` – model & training for TimeGAN
- `cgan/` – model & training for CGAN
- `scripts/` – preprocessing, metrics, plotting
- `eval/` – TSTR/TRTS, PSD/ACF/coherence, t-SNE
- `configs/` – JSON/YAML configs (seq_len=768, C=14, etc.)
- `data/` – **empty**; place your data locally; see “Data Policy”
- `figures/` – generated plots
- `docs/` – notes, diagrams

## Quickstart
```bash
# create env
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# train TimeGAN on a prepared NPZ
python timegan/train_timegan.py --config configs/timegan_config.json

# train CGAN on the same preprocessed bucket
python cgan/train_cgan.py --config configs/cgan_config.json
