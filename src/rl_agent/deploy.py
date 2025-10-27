from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_agent.train_ac_gae import PolicyValueNet, make_obs_fixer
from rl_agent.experiment import build_envs, deterministic_rewards

__all__ = ["load_policy"]


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
    )

    env_tr = envs["env_tr"]
    input_dim = env_tr.reset().size

    policy = PolicyValueNet(input_dim=input_dim, hidden=cfg["hidden"]).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.obs_fix = make_obs_fixer(cfg["window"], len(cfg["features"]))
    policy.pos_limit = float(cfg["pos_limit"])

    return policy, cfg, envs

## Additional utility functions for results analysis

def nav_curve(env, policy, label):
    r = deterministic_rewards(env, policy)
    nav = (1 + r / 1e4).cumprod()
    plt.plot(nav, label=label)
    return r, nav

def calc_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())

def rollout_stats(env, policy):
    ro = env.rollout(lambda obs: policy.act(obs, deterministic=True)[0])
    nav = (1 + ro["rewards"] / 1e4).cumprod()
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    turnover = np.abs(np.diff(ro["positions"])).sum()
    return dd, turnover

def summary_from_nav(nav, rewards=None):
    nav = np.asarray(nav, float)
    rewards = np.asarray(rewards, float) if rewards is not None else np.diff(nav) / nav[:-1]
    sharpe = rewards.mean() / rewards.std(ddof=1) * np.sqrt(252) if rewards.std(ddof=1) > 0 else 0.0
    dd = (nav / np.maximum.accumulate(nav) - 1.0).min()
    cagr = nav[-1] ** (252 / len(nav)) - 1.0  # assuming daily steps
    return dict(
        nav_final=float(nav[-1]),
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=float(dd),
        vol=float(rewards.std(ddof=1) * np.sqrt(252)),
    )
