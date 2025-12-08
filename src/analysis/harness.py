"""
Helper functions for robustness, cost stress, and feature ablation sweeps.

These helpers keep the notebooks light: they take paths/configs,
train/evaluate the policy with `train_ac_gae`, and return DataFrames
that notebooks can display or write to disk.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd

from rl_agent.train_ac_gae import train as train_gae
from rl_agent.experiment import build_envs, deterministic_rewards


@dataclass
class SplitSpec:
    tag: str
    train_end: str
    valid_end: str


def shift_dates(train_end: str, valid_end: str, days: int) -> SplitSpec:
    tr = pd.Timestamp(train_end) + pd.DateOffset(days=days)
    va = pd.Timestamp(valid_end) + pd.DateOffset(days=days)
    return SplitSpec(f"shift_{days}d", tr.date().isoformat(), va.date().isoformat())


def summarize_rewards(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    mean = float(r.mean())
    std = float(r.std(ddof=1) if r.size > 1 else 0.0)
    sharpe = float(mean / std * np.sqrt(252)) if std > 0 else 0.0
    nav = (1.0 + r / 1e4).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1.0).min())
    return {"mean_bps": mean, "std_bps": std, "sharpe": sharpe, "maxdd": dd, "steps": len(r)}


def run_robustness(panel_path: Path, config_path: Path, seeds: Iterable[int], shifts: Iterable[int], steps: int) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    base_cfg = json.loads(Path(config_path).read_text())

    rows: List[Dict] = []
    for shift in shifts:
        if shift == 0:
            split = SplitSpec("base", base_cfg["train_end"], base_cfg["valid_end"])
        else:
            split = shift_dates(base_cfg["train_end"], base_cfg["valid_end"], days=shift)

        for seed in seeds:
            envs = build_envs(
                panel,
                features=base_cfg["features"],
                train_end=split.train_end,
                valid_end=split.valid_end,
                window=int(base_cfg["window"]),
                txn_cost_bps=float(base_cfg["txn_cost_bps"]),
                pos_limit=float(base_cfg["pos_limit"]),
                rebalance_every=int(base_cfg.get("rebalance_every", 1)),
                slippage_bps=float(base_cfg.get("slippage_bps", 0)),
            )
            policy, _ = train_gae(
                panel=panel,
                state_cols=base_cfg["features"],
                train_end=split.train_end,
                valid_end=split.valid_end,
                window=int(base_cfg["window"]),
                txn_cost_bps=float(base_cfg["txn_cost_bps"]),
                pos_limit=float(base_cfg["pos_limit"]),
                rebalance_every=int(base_cfg.get("rebalance_every", 1)),
                slippage_bps=float(base_cfg.get("slippage_bps", 0)),
                hidden=256,
                steps=steps,
                seed=seed,
            )
            for split_name, env in [("train", envs["env_tr"]), ("valid", envs["env_va"]), ("test", envs["env_te"])]:
                m = summarize_rewards(deterministic_rewards(env, policy))
                m.update(dict(tag=split.tag, seed=seed, split=split_name))
                rows.append(m)

    return pd.DataFrame(rows)


def run_cost_stress(panel_path: Path, config_path: Path, configs: List[tuple[float, float]], steps: int, slippage_fn=None) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    base_cfg = json.loads(Path(config_path).read_text())
    rows: List[Dict] = []
    for txn_bps, slip_bps in configs:
        envs = build_envs(
            panel,
            features=base_cfg["features"],
            train_end=base_cfg["train_end"],
            valid_end=base_cfg["valid_end"],
            window=int(base_cfg["window"]),
            txn_cost_bps=txn_bps,
            pos_limit=float(base_cfg["pos_limit"]),
            rebalance_every=int(base_cfg.get("rebalance_every", 1)),
            slippage_bps=slip_bps,
            slippage_fn=slippage_fn,
        )
        policy, _ = train_gae(
            panel=panel,
            state_cols=base_cfg["features"],
            train_end=base_cfg["train_end"],
            valid_end=base_cfg["valid_end"],
            window=int(base_cfg["window"]),
            txn_cost_bps=txn_bps,
            pos_limit=float(base_cfg["pos_limit"]),
            rebalance_every=int(base_cfg.get("rebalance_every", 1)),
            slippage_bps=slip_bps,
            slippage_fn=slippage_fn,
            hidden=256,
            steps=steps,
            seed=12345,
        )
        for split_name, env in [("train", envs["env_tr"]), ("valid", envs["env_va"]), ("test", envs["env_te"])]:
            m = summarize_rewards(deterministic_rewards(env, policy))
            m.update(dict(split=split_name, txn_cost_bps=txn_bps, slippage_bps=slip_bps))
            rows.append(m)
    return pd.DataFrame(rows)


def run_feature_ablation(panel_path: Path, config_path: Path, feature_sets: Dict[str, List[str]], steps: int) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    base_cfg = json.loads(Path(config_path).read_text())
    rows: List[Dict] = []
    for tag, feats in feature_sets.items():
        feats = [f for f in feats if f in panel.columns]
        if not feats:
            continue
        envs = build_envs(
            panel,
            features=feats,
            train_end=base_cfg["train_end"],
            valid_end=base_cfg["valid_end"],
            window=int(base_cfg["window"]),
            txn_cost_bps=float(base_cfg["txn_cost_bps"]),
            pos_limit=float(base_cfg["pos_limit"]),
            rebalance_every=int(base_cfg.get("rebalance_every", 1)),
            slippage_bps=float(base_cfg.get("slippage_bps", 0)),
        )
        policy, _ = train_gae(
            panel=panel,
            state_cols=feats,
            train_end=base_cfg["train_end"],
            valid_end=base_cfg["valid_end"],
            window=int(base_cfg["window"]),
            txn_cost_bps=float(base_cfg["txn_cost_bps"]),
            pos_limit=float(base_cfg["pos_limit"]),
            rebalance_every=int(base_cfg.get("rebalance_every", 1)),
            slippage_bps=float(base_cfg.get("slippage_bps", 0)),
            hidden=256,
            steps=steps,
            seed=12345,
        )
        val_metrics = summarize_rewards(deterministic_rewards(envs["env_va"], policy))
        val_metrics.update(dict(tag=tag))
        rows.append(val_metrics)
    return pd.DataFrame(rows)

def stress_slippage(dpos, obs, ctx):
    # obs shape: (window, n_features); use last row for current features
    vix_idx = None
    # infer vix index if present
    for i, name in enumerate(getattr(ctx, "feature_names", [])):
        if name == 'vix':
            vix_idx = i
            break
    vix_level = float(obs[-1, vix_idx]) if vix_idx is not None else 20.0
    base = 0.0
    # spread widens with VIX (simple affine model)
    spread_bps = max(0.0, 0.5 + 0.1 * (vix_level - 20.0))
    # quadratic impact for large turnover
    impact_bps = 0.05 * (dpos ** 2)
    return (spread_bps + impact_bps) * 1e-4 * abs(dpos)