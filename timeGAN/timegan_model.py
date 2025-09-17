# timegan_model.py
# PyTorch TimeGAN components for multivariate EEG (14 channels)
# Uses GRU stacks (can switch to LSTM by flipping GRU->LSTM)

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.utils as U


def init_weights_(m):
    if isinstance(m, (nn.Linear,)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, (nn.GRU, nn.LSTM)):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


class GRUStack(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )

    def forward(self, x):
        y, _ = self.rnn(x)   # (B, T, H)
        return y


class Embedder(nn.Module):
    # X -> H
    def __init__(self, x_dim: int, z_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = GRUStack(x_dim, z_dim, num_layers, dropout)

    def forward(self, x):
        return self.rnn(x)


class Recovery(nn.Module):
    # H -> X~
    def __init__(self, z_dim: int, x_dim: int, hidden_dim: int = None, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        h = hidden_dim or z_dim
        self.rnn = GRUStack(z_dim, h, num_layers, dropout)
        self.out = nn.Linear(h, x_dim)

    def forward(self, h):
        y = self.rnn(h)
        return self.out(y)


class Generator(nn.Module):
    # Z -> E_hat (latent)
    def __init__(self, z_dim: int, hidden_dim: int = None, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        h = hidden_dim or z_dim
        self.rnn = GRUStack(z_dim, h, num_layers, dropout)
        self.proj = nn.Linear(h, z_dim) if h != z_dim else nn.Identity()

    def forward(self, z):
        y = self.rnn(z)
        return self.proj(y)


class Supervisor(nn.Module):
    # E_hat -> H_hat (temporal refinement / next-step dynamics)
    def __init__(self, z_dim: int, hidden_dim: int = None, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        h = hidden_dim or z_dim
        self.rnn = GRUStack(z_dim, h, num_layers, dropout)
        self.proj = nn.Linear(h, z_dim) if h != z_dim else nn.Identity()

    def forward(self, e_hat):
        y = self.rnn(e_hat)
        return self.proj(y)


class Discriminator(nn.Module):
    # H or H_hat -> prob(real)
    def __init__(self, z_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = GRUStack(z_dim, hidden_dim, num_layers, dropout)
        # Spectral normalization stabilizes D
        self.fc = U.spectral_norm(nn.Linear(hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        y = self.rnn(h)              # (B, T, H)
        last = y[:, -1, :]           # take last step
        return self.sigmoid(self.fc(last))


class TimeGAN(nn.Module):
    """Bundle of submodules + convenience calls."""
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedder = Embedder(x_dim, z_dim, num_layers, dropout)
        self.recovery = Recovery(z_dim, x_dim, hidden_dim, num_layers, dropout)
        self.generator = Generator(z_dim, hidden_dim, num_layers, dropout)
        self.supervisor = Supervisor(z_dim, hidden_dim, num_layers, dropout)
        self.discriminator = Discriminator(z_dim, hidden_dim, num_layers, dropout)
        self.apply(init_weights_)

    # Convenience passes
    def encode(self, x):        return self.embedder(x)                 # H
    def reconstruct(self, x):   return self.recovery(self.embedder(x))  # X~
    def gen_latent(self, z):    return self.generator(z)                # E_hat
    def refine_latent(self, e): return self.supervisor(e)               # H_hat
    def decode(self, h):        return self.recovery(h)                 # X_hat
    def disc(self, h):          return self.discriminator(h)            # prob
