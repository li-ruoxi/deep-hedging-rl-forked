from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from rl_agent.experiment import deterministic_rewards

__all__ = [
    "nav_curve",
    "calc_drawdown",
    "rollout_stats",
    "summary_from_nav",
    "nav_from_bps",
    "max_drawdown",
    "sortino_bps",
    "combo_summary",
]


def nav_from_bps(bps: np.ndarray) -> np.ndarray:
    r = np.asarray(bps, dtype=float)
    return (1.0 + r / 1e4).cumprod()


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())


def sortino_bps(bps: np.ndarray) -> float:
    r = np.asarray(bps, dtype=float)
    dn = r[r < 0]
    if dn.size == 0:
        return float("inf")
    dsd = dn.std(ddof=1) if dn.size > 1 else dn.std()
    if dsd == 0:
        return float("inf")
    return float(r.mean() / dsd * np.sqrt(252))


def nav_curve(env, policy, label: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rewards = deterministic_rewards(env, policy)
    nav = nav_from_bps(rewards)
    if label is not None:
        plt.plot(nav, label=label)
    return rewards, nav


def calc_drawdown(nav: np.ndarray) -> float:
    return max_drawdown(nav)


def rollout_stats(env, policy) -> Tuple[float, float]:
    ro = env.rollout(lambda obs: policy.act(obs, deterministic=True)[0])
    nav = nav_from_bps(ro["rewards"])
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    turnover = np.abs(np.diff(ro["positions"])).sum()
    return dd, turnover


def summary_from_nav(nav: np.ndarray, rewards: np.ndarray | None = None) -> dict:
    nav = np.asarray(nav, float)
    rewards = np.asarray(rewards, float) if rewards is not None else np.diff(nav) / nav[:-1]
    std = rewards.std(ddof=1)
    sharpe = rewards.mean() / std * np.sqrt(252) if std > 0 else 0.0
    dd = max_drawdown(nav)
    cagr = nav[-1] ** (252 / len(nav)) - 1.0
    return {
        "nav_final": float(nav[-1]),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "vol": float(std * np.sqrt(252)),
    }


def combo_summary(env, policy_main, policy_baseline, blend_weight: float, split: str) -> dict:
    blend = float(blend_weight)
    blend = min(max(blend, 0.0), 1.0)
    main_r = deterministic_rewards(env, policy_main)
    base_r = deterministic_rewards(env, policy_baseline)
    combo_r = blend * main_r + (1.0 - blend) * base_r
    nav = nav_from_bps(combo_r)
    mean = combo_r.mean()
    std = combo_r.std(ddof=1) if combo_r.size > 1 else 0.0
    sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0.0
    sortino = sortino_bps(combo_r)
    dd = max_drawdown(nav)
    hit = float((combo_r > 0).mean())
    steps = int(combo_r.size)
    cagr = float(nav[-1] ** (252 / steps) - 1.0) if steps > 0 else 0.0
    calmar = float(cagr / abs(dd)) if dd < 0 else float("inf")
    return {
        "split": split,
        "mean_bps": float(mean),
        "std_bps": float(std),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd),
        "hit_rate": hit,
        "cagr": cagr,
        "calmar": calmar,
        "nav": nav,
        "rewards": combo_r,
    }
