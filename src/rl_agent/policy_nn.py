# src/rl_agent/policy_nn.py
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import Normal


def _flatten_obs(x) -> torch.Tensor:
    """
    x can be a numpy array (window, n_feat) or 1D.
    Returns a 1D float32 torch tensor with NaNs replaced by 0.
    """
    import numpy as np
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = x
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return torch.from_numpy(arr)


@dataclass
class PolicyConfig:
    input_dim: int             # window * n_features (flattened)
    hidden: int = 64
    init_log_std: float = -1.0 # e.g., std ≈ 0.37
    device: str = "cpu"


class PolicyNetwork(nn.Module):
    """
    Minimal stochastic policy:
      - MLP -> Normal(mu, std) over pre-tanh actions
      - Tanh squashing to keep final action in [-1, 1]
    """
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(cfg.init_log_std, dtype=torch.float32))
        self.tanh = nn.Tanh()
        self.to(cfg.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (input_dim,) or (batch, input_dim) -> pre-tanh mean"""
        return self.net(x)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """
        Returns: action(float), log_prob(tensor), mu_action(float)
        """
        x = _flatten_obs(obs).to(self.cfg.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        mu = self.forward(x)                   # (1,1)
        std = self.log_std.exp().clamp(1e-4, 5.0)
        if deterministic:
            z = mu
        else:
            dist = Normal(mu, std)
            z = dist.rsample()                 # reparameterized sample

        a = self.tanh(z)                       # squashed to [-1, 1]

        # log_prob with tanh correction (see SAC)
        # log π(a) = log N(z; mu, std) - log(1 - tanh(z)^2)
        if deterministic:
            logp = torch.zeros_like(a)
        else:
            logp = dist.log_prob(z) - torch.log1p(-a.pow(2) + 1e-6)
            logp = logp.sum(dim=-1, keepdim=True)

        a_float = float(a.squeeze().cpu().item())
        mu_float = float(self.tanh(mu).squeeze().cpu().item())
        return a_float, logp.squeeze(0), mu_float
