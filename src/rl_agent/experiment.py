# src/rl_agent/experiment.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, platform, sys, datetime as dt
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

from simulator.env import HedgingEnv
from simulator.rewards import reward_bps

# ----------------- split + checks -----------------
def _validate_features(panel: pd.DataFrame, features: List[str]) -> None:
    miss = [c for c in features if c not in panel.columns]
    if miss:
        raise ValueError(f"Missing features in panel: {miss}")
    bad = [c for c in features if not np.issubdtype(panel[c].dtype, np.number)]
    if bad:
        raise TypeError(f"Non-numeric feature columns: {bad}")

def make_splits(panel: pd.DataFrame, train_end: str, valid_end: str):
    p = panel.copy()
    if "date" not in p.columns:
        raise ValueError("panel must contain a 'date' column")
    p["date"] = pd.to_datetime(p["date"])
    TRAIN_END = pd.Timestamp(train_end)
    VALID_END = pd.Timestamp(valid_end)
    m_tr = p["date"] <= TRAIN_END
    m_va = (p["date"] > TRAIN_END) & (p["date"] <= VALID_END)
    m_te = p["date"] > VALID_END
    return p, m_tr, m_va, m_te, TRAIN_END, VALID_END

# ----------------- scaler -----------------
def make_scaler(panel: pd.DataFrame, cols: List[str], mask_train):
    mu = panel.loc[mask_train, cols].mean()
    sg = panel.loc[mask_train, cols].std(ddof=1).replace(0, np.nan).fillna(1.0)

    def zscale(obs: np.ndarray) -> np.ndarray:
        return (obs - mu.values) / sg.values

    return zscale, mu.astype(float).to_dict(), sg.astype(float).to_dict()

def save_scaler(path: Path, mu: Dict[str, float], sg: Dict[str, float]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mu": mu, "sg": sg}, indent=2))

def load_scaler(path: Path, cols: List[str]) -> Callable[[np.ndarray], np.ndarray]:
    js = json.loads(Path(path).read_text())
    mu = np.array([js["mu"][c] for c in cols], dtype=float)
    sg = np.array([js["sg"][c] for c in cols], dtype=float)
    def zscale(obs: np.ndarray) -> np.ndarray:
        return (obs - mu) / np.where(sg == 0.0, 1.0, sg)
    return zscale

# ----------------- reward -----------------
def reward_bps(pnl, info, scale: float = 1e4):
    from simulator.rewards import pnl_only
    return pnl_only(pnl, info) * scale

# ----------------- env factory -----------------
def make_env(panel: pd.DataFrame, mask, features: List[str], window: int,
             txn_cost_bps: float, scaler: Callable[[np.ndarray], np.ndarray],
             pos_limit: float, reward_scale: float = 1e4):
    return HedgingEnv(
        df=panel.loc[mask].reset_index(drop=True),
        features=features,
        reward_fn=lambda pnl, info: reward_bps(pnl, info, reward_scale),
        window=window,
        txn_cost_bps=txn_cost_bps,
        scaler=scaler,
        hold_on_nan=True,
        pos_limit=pos_limit,
    )

# ----------------- deterministic rollout -----------------
def deterministic_rewards(env, policy, device: str = "cpu") -> np.ndarray:
    import numpy as np
    obs = env.reset()
    rewards: list[float] = []
    while True:
        act = policy.act(obs, deterministic=True)
        action = act[0] if isinstance(act, tuple) else act
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        action = float(np.asarray(action).squeeze())
        obs, r, done, _ = env.step(action)
        rewards.append(float(r))
        if done:
            break
    return np.asarray(rewards, float)

# ----------------- metrics -----------------
def summarize_bps(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    sd = r.std(ddof=1)
    sharpe = (r.mean() / (sd if sd>0 else 1.0)) * np.sqrt(252)
    eq = (1 + r/1e4).cumprod()
    dd = float((eq/eq.cummax()-1).min())
    return {"mean": float(r.mean()), "std": float(sd), "sharpe": float(sharpe), "maxdd": dd}

@dataclass
class ArtifactLogger:
    outdir: Path
    def save(self, split_metrics: dict, curves: dict, config: dict):
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir/"results.json").write_text(json.dumps(split_metrics, indent=2))
        for k, v in curves.items():
            pd.Series(v, name="bps").to_csv(self.outdir/f"{k}_bps.csv", index=False)
        ver = dict(python=sys.version, system=platform.platform(),
                   timestamp=dt.datetime.utcnow().isoformat()+"Z")
        (self.outdir/"versions.json").write_text(json.dumps(ver, indent=2))
        (self.outdir/"config.json").write_text(json.dumps(config, indent=2))

# ----------------- one-call builder -----------------
def build_envs(panel: pd.DataFrame, features: List[str], train_end: str, valid_end: str,
               window: int, txn_cost_bps: float, pos_limit: float,
               reward_scale: float = 1e4,
               scaler_out: Path | None = None):
    _validate_features(panel, features)
    p, m_tr, m_va, m_te, TRAIN_END, VALID_END = make_splits(panel, train_end, valid_end)
    scaler, mu, sg = make_scaler(p, features, m_tr)
    if scaler_out is not None:
        save_scaler(Path(scaler_out), mu, sg)

    env_tr = make_env(p, m_tr, features, window, txn_cost_bps, scaler, pos_limit, reward_scale)
    env_va = make_env(p, m_va, features, window, txn_cost_bps, scaler, pos_limit, reward_scale)
    env_te = make_env(p, m_te, features, window, txn_cost_bps, scaler, pos_limit, reward_scale)
    input_dim = env_tr.reset().size
    config = dict(
        features=features, window=window, txn_cost_bps=txn_cost_bps, pos_limit=pos_limit,
        reward_scale=reward_scale, train_end=str(TRAIN_END.date()), valid_end=str(VALID_END.date())
    )
    return dict(
        env_tr=env_tr, env_va=env_va, env_te=env_te,
        input_dim=input_dim, config=config, scaler=scaler,
        TRAIN_END=TRAIN_END, VALID_END=VALID_END,
    )
