from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch

from rl_agent.train_ac_gae import PolicyValueNet, make_obs_fixer
from rl_agent.experiment import build_envs
from rl_agent.metrics import (
    nav_curve,
    calc_drawdown,
    rollout_stats,
    summary_from_nav,
    nav_from_bps,
    max_drawdown,
    sortino_bps,
)

__all__ = [
    "load_policy",
    "nav_curve",
    "calc_drawdown",
    "rollout_stats",
    "summary_from_nav",
    "nav_from_bps",
    "max_drawdown",
    "sortino_bps",
]


def load_policy(
    run_dir: str | Path,
    panel: pd.DataFrame,
    device: str = "cpu",
    reward_scale: float = 1e4,
) -> Tuple[PolicyValueNet, dict, dict]:
    """
    Restore a trained actor-critic policy together with its config and env splits.

    Args:
        run_dir: Directory containing `best.pt` and `config.json`.
        panel:   Pre-built panel DataFrame (same schema used for training).
        device:  Torch device to map the policy to (e.g. "cpu" or "cuda").
        reward_scale: Reward scaling factor to pass to `build_envs`.

    Returns:
        policy: Loaded PolicyValueNet ready for deterministic or stochastic rollout.
        cfg:    Run configuration dictionary parsed from config.json.
        envs:   Dict with `env_tr`, `env_va`, `env_te`, matching `build_envs`.
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg = json.loads(cfg_path.read_text())

    envs = build_envs(
        panel=panel,
        features=cfg["features"],
        train_end=cfg["train_end"],
        valid_end=cfg["valid_end"],
        window=cfg["window"],
        txn_cost_bps=cfg["txn_cost_bps"],
        pos_limit=cfg["pos_limit"],
        reward_scale=reward_scale,
        rebalance_every=int(cfg.get("rebalance_every", 1)),
        slippage_bps=float(cfg.get("slippage_bps", 0.0)),
    )

    env_tr = envs["env_tr"]
    input_dim = env_tr.reset().size

    policy = PolicyValueNet(input_dim=input_dim, hidden=cfg["hidden"]).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.obs_fix = make_obs_fixer(cfg["window"], len(cfg["features"]))
    policy.pos_limit = float(cfg["pos_limit"])

    return policy, cfg, envs
