# train_timegan.py
# Trains 1 TimeGAN per (posture, condition) NPZ in --data_dir.
# Now includes: spectral norm (in model), R1 penalty for D, D throttling,
# covariance + ACF losses for G, stronger supervisor weighting, and tuned defaults.

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from timegan_model import TimeGAN

# ---------------------- Utilities ----------------------

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def device_autoselect():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, batch_size: int) -> DataLoader:
    tens = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(tens)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False,
                      pin_memory=True, num_workers=0)


def smooth_labels(size: int, smooth: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    real = (1.0 - smooth) + smooth * torch.rand(size, 1, device=device)
    fake = smooth * torch.rand(size, 1, device=device)
    return real, fake


def add_instance_noise(h: torch.Tensor, std: float) -> torch.Tensor:
    return h if std <= 0 else h + std * torch.randn_like(h)


def adaptive_dims(x_dim: int, seq_len: int) -> Tuple[int, int]:
    z = max(16, min(64, x_dim * 2))         # with 14 ch -> 28
    h = max(32, min(128, x_dim * 4))        # with 14 ch -> 56
    if seq_len > 800:
        z = min(64, z + 8); h = min(128, h + 16)
    return z, h


def save_ckpt(path: Path, model: TimeGAN, optG, optD, step: int, meta: Dict):
    state = {"step": step, "model": model.state_dict(),
             "optG": optG.state_dict(), "optD": optD.state_dict(), "meta": meta}
    torch.save(state, path)


def sample_noise(batch_size: int, seq_len: int, z_dim: int, device):
    return torch.rand(batch_size, seq_len, z_dim, device=device)


# ---------------------- Losses ----------------------

bce = nn.BCELoss()

def recon_loss(x, x_tilde, eps=1e-8):
    mse = torch.mean((x - x_tilde) ** 2)
    return 10.0 * torch.sqrt(mse + eps)

def sup_loss(h_real):
    return torch.mean((h_real[:, 1:, :] - h_real[:, :-1, :]) ** 2)

def sup_loss_fake(h_fake):
    return torch.mean((h_fake[:, 1:, :] - h_fake[:, :-1, :]) ** 2)

@torch.no_grad()
def batch_cov(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, C) -> covariance matrix (C, C), no grad needed for real.
    """
    B, T, C = x.shape
    X = x.reshape(B * T, C)
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.t() @ X) / (X.size(0) - 1)
    return cov

def batch_cov_with_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Same as batch_cov but differentiable (for generated x).
    """
    B, T, C = x.shape
    X = x.reshape(B * T, C)
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.transpose(0, 1) @ X) / (X.size(0) - 1)
    return cov

def acf_loss_torch(x_gen: torch.Tensor, x_real: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Average L1 difference between per-channel autocorrelations across lags.
    x_*: (B, T, C)
    """
    B, T, C = x_gen.shape
    max_lag = max(1, min(max_lag, T - 1))
    def acf_all(x):
        acfs = []
        # center and scale per channel across (B,T)
        xm = x.mean(dim=(0,1), keepdim=True)
        xs = x.std(dim=(0,1), keepdim=True) + 1e-8
        xz = (x - xm) / xs
        for lag in range(1, max_lag+1):
            a = xz[:, :-lag, :]  # (B, T-lag, C)
            b = xz[:, lag:,  :]  # (B, T-lag, C)
            # correlation at this lag (mean over B,T)
            corr = (a * b).mean(dim=(0,1))  # (C,)
            acfs.append(corr)
        return torch.stack(acfs, dim=0)  # (L, C)
    acf_g = acf_all(x_gen)
    with torch.no_grad():
        acf_r = acf_all(x_real)
    return torch.mean(torch.abs(acf_g - acf_r))


# ---------------------- Training phases ----------------------

def phase_autoencoder(model: TimeGAN, loader: DataLoader, device, optER: optim.Optimizer,
                      clip: float, epochs: int, log):
    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for (x_batch,) in loader:
            x = x_batch.to(device, non_blocking=True)
            x_tilde = model.reconstruct(x)
            loss = recon_loss(x, x_tilde)
            optER.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(list(model.embedder.parameters()) + list(model.recovery.parameters()), clip)
            optER.step()
            epoch_loss += loss.item() * x.size(0); n += x.size(0)
        log(f"[AE] epoch {ep}/{epochs}  recon={epoch_loss / n:.5f}")


def phase_supervisor(model: TimeGAN, loader: DataLoader, device, optS: optim.Optimizer,
                     clip: float, epochs: int, log):
    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for (x_batch,) in loader:
            x = x_batch.to(device, non_blocking=True)
            with torch.no_grad():
                h = model.encode(x)
            h_in, h_tgt = h[:, :-1, :], h[:, 1:, :]
            h_pred = model.supervisor(h_in)
            loss = torch.mean((h_pred - h_tgt) ** 2)
            optS.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.supervisor.parameters(), clip)
            optS.step()
            epoch_loss += loss.item() * x.size(0); n += x.size(0)
        log(f"[SUP] epoch {ep}/{epochs}  sup={epoch_loss / n:.5f}")


def disc_step(model: TimeGAN, x, device, optD, label_smooth, inst_noise_std, clip,
              schedulerD=None, r1_gamma: float = 1.0,
              target_acc: float = 0.55, band: float = 0.10):
    """
    Discriminator update with R1. We always update D, but scale its loss when D is already strong.
    target_acc ~ 0.55; band ~ 0.10 -> soft throttle between ~0.45 and ~0.65.
    """
    model.discriminator.train()

    with torch.no_grad():
        h_real = model.encode(x)

    # Fake latent from G+S
    z = sample_noise(x.size(0), x.size(1), model.embedder.rnn.rnn.hidden_size, device)
    e_hat = model.gen_latent(z)
    h_fake = model.refine_latent(e_hat)

    # Instance noise
    h_real_n = add_instance_noise(h_real, inst_noise_std).requires_grad_(True)
    h_fake_n = add_instance_noise(h_fake.detach(), inst_noise_std)

    # Labels with smoothing
    y_real, y_fake = smooth_labels(x.size(0), label_smooth, device)

    # Forward (cuDNN off on real branch to allow R1 double-backward)
    with torch.backends.cudnn.flags(enabled=False):
        d_real = model.disc(h_real_n)
    d_fake = model.disc(h_fake_n)

    # BCE on real+fake
    loss = 0.5 * (bce(d_real, y_real) + bce(d_fake, y_fake))

    # R1 gradient penalty on real
    if r1_gamma > 0.0:
        grad_real = torch.autograd.grad(d_real.sum(), h_real_n, create_graph=True, retain_graph=True)[0]
        r1 = grad_real.reshape(grad_real.size(0), -1).pow(2).sum(1).mean()
        loss = loss + 0.5 * r1_gamma * r1

    # Measure current accuracy (balanced)
    with torch.no_grad():
        acc_real = (d_real > 0.5).float().mean().item()
        acc_fake = (d_fake < 0.5).float().mean().item()
        acc = 0.5 * (acc_real + acc_fake)

    # ---- Soft throttle: scale D's loss if it's already too strong ----
    if band > 0:
        # scale ∈ [0.2, 1.0]; near target -> 1, far above target -> down to 0.2
        over = max(0.0, acc - target_acc)
        scale = max(0.2, 1.0 - over / band)
        loss = loss * scale
    # -----------------------------------------------------------------

    optD.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.discriminator.parameters(), clip)
    optD.step()
    if schedulerD is not None:
        schedulerD.step()

    return loss.item(), acc


def gen_step(model: TimeGAN, x, device, optG,
             alpha_sup, beta_rec, inst_noise_std, clip,
             schedulerG=None, gamma_cov: float = 0.0,
             gamma_acf: float = 0.0, acf_max_lag: int = 32):
    model.generator.train(); model.supervisor.train(); model.embedder.train(); model.recovery.train()

    # 1) adversarial loss
    z = sample_noise(x.size(0), x.size(1), model.embedder.rnn.rnn.hidden_size, device)
    e_hat = model.gen_latent(z)
    h_hat = model.refine_latent(e_hat)

    # Instance noise for D input
    d_fake = model.disc(add_instance_noise(h_hat, inst_noise_std))
    g_adv = bce(d_fake, torch.ones_like(d_fake))

    # 2) supervised loss on fake latent dynamics
    g_sup = sup_loss_fake(h_hat)

    # 3) reconstruction loss on real -> keeps E/R aligned
    x_tilde = model.reconstruct(x)
    g_rec = recon_loss(x, x_tilde)

    # 4) decode generated to X_hat for stat losses
    x_hat = model.decode(h_hat)

    # Covariance loss (match channel covariance)
    cov_term = torch.tensor(0.0, device=device)
    if gamma_cov > 0:
        cov_r = batch_cov(x.detach())
        cov_g = batch_cov_with_grad(x_hat)
        cov_term = torch.norm(cov_g - cov_r, p='fro') / (cov_r.numel() ** 0.5)

    # ACF loss (match autocorrelation structure)
    acf_term = torch.tensor(0.0, device=device)
    if gamma_acf > 0:
        acf_term = acf_loss_torch(x_hat, x.detach(), acf_max_lag)

    g_total = g_adv + alpha_sup * g_sup + beta_rec * g_rec + gamma_cov * cov_term + gamma_acf * acf_term

    optG.zero_grad(set_to_none=True); g_total.backward()
    nn.utils.clip_grad_norm_(list(model.generator.parameters()) +
                             list(model.supervisor.parameters()) +
                             list(model.embedder.parameters()) +
                             list(model.recovery.parameters()), clip)
    optG.step()
    if schedulerG is not None:
        schedulerG.step()

    return g_total.item(), g_adv.item(), g_sup.item(), g_rec.item(), cov_term.item(), acf_term.item()


# ---------------------- Full training ----------------------

def train_single_npz(npz_path: Path, out_dir: Path,
                     batch_size=64,
                     ae_epochs=120,
                     sup_epochs=150,
                     gan_steps=8000,
                     lr_g=1e-3, lr_d=2e-4,
                     betas=(0.5, 0.9),
                     alpha_sup=5.0,
                     beta_rec=0.2,
                     label_smooth=0.2,
                     inst_noise_start=0.3,
                     inst_noise_end=0.1,
                     grad_clip=0.5,
                     layers=1,
                     dropout=0.2,
                     seed=42,
                     r1_gamma=1.0,
                     d_min_acc=0.45,
                     d_max_acc=0.60,
                     gamma_cov=0.05,
                     gamma_acf=0.05,
                     acf_max_lag=64,
                     device=None):

    set_seeds(seed)
    device = device or device_autoselect()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    X = data["X"].astype(np.float32)  # (N,T,C)
    N, T, C = X.shape
    z_dim, h_dim = adaptive_dims(C, T)

    # Logs
    log_file = out_dir / "train_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["step", "phase", "loss_D", "acc_D", "loss_G", "loss_adv", "loss_sup",
             "loss_rec", "loss_cov", "loss_acf"]
        )

    def LOG(msg): print(msg, flush=True)

    LOG(f"==> {npz_path.name} | N={N} T={T} C={C}  z_dim={z_dim} h_dim={h_dim}  device={device}")

    # Data & Model
    loader = make_loader(X, batch_size)
    model = TimeGAN(x_dim=C, z_dim=z_dim, hidden_dim=h_dim, num_layers=layers, dropout=dropout).to(device)

    # Phase 1: Autoencoder
    optER = optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()),
                       lr=lr_g, betas=betas)
    phase_autoencoder(model, loader, device, optER, grad_clip, ae_epochs, LOG)

    # Phase 2: Supervisor
    optS = optim.Adam(model.supervisor.parameters(), lr=lr_g, betas=betas)
    phase_supervisor(model, loader, device, optS, grad_clip, sup_epochs, LOG)

    # Phase 3: Adversarial
    optD = optim.Adam(model.discriminator.parameters(), lr=lr_d, betas=betas)
    optG = optim.Adam(list(model.generator.parameters()) +
                      list(model.supervisor.parameters()) +
                      list(model.embedder.parameters()) +
                      list(model.recovery.parameters()),
                      lr=lr_g, betas=betas)

    # LR schedulers (mild decay)
    schedulerG = optim.lr_scheduler.MultiStepLR(optG, milestones=[gan_steps // 2, int(gan_steps * 0.75)], gamma=0.5)
    schedulerD = optim.lr_scheduler.MultiStepLR(optD, milestones=[gan_steps // 2, int(gan_steps * 0.75)], gamma=0.5)

    loader_iter = iter(loader)
    inst_noise = inst_noise_start
    noise_decay = (inst_noise_start - inst_noise_end) / max(1, gan_steps)

    best_ckpt_loss = math.inf
    ckpt_path = out_dir / "ckpt_latest.pt"
    best_path = out_dir / "ckpt_best.pt"

    def log_row(step, phase, d, acc, g, adv, supv, rec, covv, acfv):
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([step, phase, d, acc, g, adv, supv, rec, covv, acfv])

    last_d_acc = 0.5

    for step in range(1, gan_steps + 1):
        try:
            (x_batch,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            (x_batch,) = next(loader_iter)
        x = x_batch.to(device, non_blocking=True)

        # --- D step with throttling ---
        d_loss, d_acc = float("nan"), last_d_acc
        # if last_d_acc <= d_max_acc:
            # Always update D (soft throttle inside disc_step)
        target = 0.5 * (d_min_acc + d_max_acc)
        band   = max(0.0, d_max_acc - d_min_acc)
        d_loss, d_acc = disc_step(
            model, x, device, optD, label_smooth, inst_noise, grad_clip,
            schedulerD, r1_gamma, target_acc=target, band=band
)
        last_d_acc = d_acc

        # # if D too weak, give it an extra nudge
        # if not np.isnan(d_loss) and d_acc < d_min_acc:
        #     d_loss, d_acc = disc_step(model, x, device, optD, label_smooth,
        #                               inst_noise, grad_clip, schedulerD, r1_gamma)
        # last_d_acc = d_acc

        # --- G step ---
        g_total, g_adv, g_supv, g_rec, g_cov, g_acf = gen_step(
            model, x, device, optG, alpha_sup, beta_rec, inst_noise, grad_clip,
            schedulerG, gamma_cov, gamma_acf, acf_max_lag
        )

        if step % 100 == 0:
            LOG(f"[GAN] step {step}/{gan_steps}  D:loss={d_loss:.4f} acc≈{d_acc:.2f}  "
                f"G:total={g_total:.4f} (adv={g_adv:.4f}, sup={g_supv:.4f}, "
                f"rec={g_rec:.4f}, cov={g_cov:.4f}, acf={g_acf:.4f})")
        log_row(step, "GAN", d_loss, d_acc, g_total, g_adv, g_supv, g_rec, g_cov, g_acf)

        # schedule instance noise
        inst_noise = max(inst_noise_end, inst_noise - noise_decay)

        # Save latest and best
        if step % 500 == 0 or step == gan_steps:
            save_ckpt(ckpt_path, model, optG, optD, step,
                      {"npz": npz_path.name, "z_dim": z_dim, "h_dim": h_dim})
        if g_total < best_ckpt_loss:
            best_ckpt_loss = g_total
            save_ckpt(best_path, model, optG, optD, step,
                      {"npz": npz_path.name, "z_dim": z_dim, "h_dim": h_dim, "best": True})

    # Generate same number as real
    model.eval()
    with torch.no_grad():
        Z = sample_noise(N, T, z_dim, device)
        X_hat = model.decode(model.refine_latent(model.gen_latent(Z))).cpu().numpy().astype(np.float32)
    np.savez_compressed(out_dir / "synthetic.npz", X=X_hat)
    print(f"Saved synthetic: {out_dir/'synthetic.npz'}")
    return True


# ---------------------- Entry point ----------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_dir", type=str, default="./preprocessed",
                    help="Folder with postureX_with_exo.npz / postureX_no_exo.npz")
    ap.add_argument("--out_dir", type=str, default="./timegan_runs", help="Output root folder")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--ae_epochs", type=int, default=120)
    ap.add_argument("--sup_epochs", type=int, default=150)
    ap.add_argument("--gan_steps", type=int, default=8000)
    ap.add_argument("--lr_g", type=float, default=1e-3)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.9)
    ap.add_argument("--alpha_sup", type=float, default=5.0)
    ap.add_argument("--beta_rec", type=float, default=0.2)
    ap.add_argument("--label_smooth", type=float, default=0.2)
    ap.add_argument("--inst_noise_start", type=float, default=0.3)
    ap.add_argument("--inst_noise_end", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # New stabilization + regularizers
    ap.add_argument("--r1_gamma", type=float, default=1.0)
    ap.add_argument("--d_min_acc", type=float, default=0.45)
    ap.add_argument("--d_max_acc", type=float, default=0.60)
    ap.add_argument("--gamma_cov", type=float, default=0.05)
    ap.add_argument("--gamma_acf", type=float, default=0.05)
    ap.add_argument("--acf_max_lag", type=int, default=64)
    args = ap.parse_args()

    device = device_autoselect()
    print(f"Using device: {device}")

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.data_dir).glob("posture*_*.npz"))
    if not files:
        raise SystemExit(f"No NPZs found in {args.data_dir}. Run preprocessing first.")

    for fp in files:
        out_dir = out_root / fp.stem
        train_single_npz(
            fp, out_dir,
            batch_size=args.batch_size,
            ae_epochs=args.ae_epochs,
            sup_epochs=args.sup_epochs,
            gan_steps=args.gan_steps,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            betas=(args.beta1, args.beta2),
            alpha_sup=args.alpha_sup,
            beta_rec=args.beta_rec,
            label_smooth=args.label_smooth,
            inst_noise_start=args.inst_noise_start,
            inst_noise_end=args.inst_noise_end,
            grad_clip=args.grad_clip,
            layers=args.layers,
            dropout=args.dropout,
            seed=args.seed,
            r1_gamma=args.r1_gamma,
            d_min_acc=args.d_min_acc,
            d_max_acc=args.d_max_acc,
            gamma_cov=args.gamma_cov,
            gamma_acf=args.gamma_acf,
            acf_max_lag=args.acf_max_lag,
            device=device
        )


if __name__ == "__main__":
    main()
