# rewards.py
from __future__ import annotations
import numpy as np
from typing import Dict

def pnl_only(pnl: float, info: Dict) -> float:
    """Raw per-step P&L."""
    return float(pnl)

def log_utility(pnl: float, info: Dict) -> float:
    """
    Δ log(NAV) as reward. Requires 'nav' and 'pnl' in info.
    """
    nav = float(info["nav"])
    prev = max(1e-12, nav / (1.0 + float(info["pnl"])))
    return float(np.log(max(nav, 1e-12)) - np.log(prev))

def mean_variance(pnl: float, info: Dict, lam: float = 5.0) -> float:
    """Per-step mean-variance proxy."""
    return float(pnl - lam * (pnl ** 2))

def downside_focus(pnl: float, info: Dict, kappa: float = 5.0) -> float:
    """Penalize losses more than gains."""
    return float(pnl if pnl >= 0 else pnl * (1.0 + kappa))

def reward_bps(pnl: float, info: Dict, scale: float = 1e4) -> float:
    """Scale pnl into basis points (default: 1 bp = 1e-4)."""
    return pnl_only(pnl, info) * float(scale)

def first_full_date(df, cols):
    m = df[cols].notna().all(axis=1)
    return df.loc[m, "date"].min()

def period_sharpe(panel, rewards, start, end, window: int = 0):
    """
    Compute Sharpe over a date range, aligning rewards to the rows the env stepped through.

    Args:
        panel:   DataFrame with a 'date' column matching the env data.
        rewards: Reward array from an env rollout.
        start:   Period start date (inclusive).
        end:     Period end date (inclusive).
        window:  Observation window used by the env; rewards begin after this many rows.
    """
    n = len(rewards)
    dates = panel["date"].iloc[window : window + n]
    mask = (dates >= start) & (dates <= end)
    r = rewards[mask.to_numpy()]
    if r.size < 5 or np.isclose(r.std(ddof=1), 0.0):
        return 0.0
    return r.mean() / r.std(ddof=1) * np.sqrt(252)
