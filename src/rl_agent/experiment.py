# src/rl_agent/experiment.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, platform, sys, datetime as dt
from typing import Callable, Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from simulator.env import HedgingEnv
from simulator.rewards import reward_bps
from simulator.baselines import (
    momentum_policy,
    delta_hedge_policy,
    volatility_targeting,
    hold_policy,
    vix_band_policy,
    vix_vol_target,
)

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
    p = p.sort_values("date").reset_index(drop=True)
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

# ----------------- env factory -----------------
def make_env(panel: pd.DataFrame, mask, features: List[str], window: int,
             txn_cost_bps: float, scaler: Callable[[np.ndarray], np.ndarray],
             pos_limit: float, reward_scale: float = 1e4,
             rebalance_every: int = 1, slippage_bps: float = 0.0,
             slippage_fn: Callable[[float, np.ndarray, dict], float] | None = None):
    return HedgingEnv(
        df=panel.loc[mask].reset_index(drop=True),
        features=features,
        reward_fn=lambda pnl, info: reward_bps(pnl, info, reward_scale),
        window=window,
        txn_cost_bps=txn_cost_bps,
        scaler=scaler,
        hold_on_nan=True,
        pos_limit=pos_limit,
        rebalance_every=rebalance_every,
        slippage_bps=slippage_bps,
        slippage_fn=slippage_fn,
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

@dataclass
class BaselineSpec:
    """Container describing a deterministic baseline policy and its aligned envs."""
    policy: Callable[[np.ndarray], float]
    envs: Dict[str, HedgingEnv]
    features: List[str]
    description: Optional[str] = None

# ----------------- one-call builder -----------------
def build_envs(panel: pd.DataFrame, features: List[str], train_end: str, valid_end: str,
               window: int, txn_cost_bps: float, pos_limit: float,
               reward_scale: float = 1e4,
               scaler_out: Path | None = None,
               rebalance_every: int = 1,
               slippage_bps: float = 0.0,
               slippage_fn: Callable[[float, np.ndarray, dict], float] | None = None):
    _validate_features(panel, features)
    if "ret_fwd" not in panel.columns:
        raise ValueError("panel must contain a 'ret_fwd' column before building envs")
    p, m_tr, m_va, m_te, TRAIN_END, VALID_END = make_splits(panel, train_end, valid_end)
    scaler, mu, sg = make_scaler(p, features, m_tr)
    if scaler_out is not None:
        save_scaler(Path(scaler_out), mu, sg)

    env_tr = make_env(p, m_tr, features, window, txn_cost_bps, scaler, pos_limit, reward_scale,
                      rebalance_every=rebalance_every, slippage_bps=slippage_bps,
                      slippage_fn=slippage_fn)
    env_va = make_env(p, m_va, features, window, txn_cost_bps, scaler, pos_limit, reward_scale,
                      rebalance_every=rebalance_every, slippage_bps=slippage_bps,
                      slippage_fn=slippage_fn)
    env_te = make_env(p, m_te, features, window, txn_cost_bps, scaler, pos_limit, reward_scale,
                      rebalance_every=rebalance_every, slippage_bps=slippage_bps,
                      slippage_fn=slippage_fn)
    input_dim = env_tr.reset().size
    config = dict(
        features=features, window=window, txn_cost_bps=txn_cost_bps, pos_limit=pos_limit,
        reward_scale=reward_scale, train_end=str(TRAIN_END.date()), valid_end=str(VALID_END.date()),
        rebalance_every=rebalance_every, slippage_bps=slippage_bps,
    )
    return dict(
        env_tr=env_tr, env_va=env_va, env_te=env_te,
        input_dim=input_dim, config=config, scaler=scaler,
        TRAIN_END=TRAIN_END, VALID_END=VALID_END,
        panel=p,
        masks={"train": m_tr, "valid": m_va, "test": m_te},
    )

# ----------------- baseline helpers -----------------
def augment_panel_with_baseline_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add convenience columns that make heuristic baselines easier to express.
    Returns a *copy* of panel so upstream references remain untouched.
    """
    p = panel.copy()
    if "close_spy" in p.columns and "close_spy_ratio5" not in p.columns:
        roll = p["close_spy"].rolling(5, min_periods=1).mean()
        p["close_spy_ratio5"] = (p["close_spy"] / roll).clip(lower=1e-6)
    if "delta_atm_30d_spx" in p.columns and "delta_atm_30d_spx_ewm5" not in p.columns:
        p["delta_atm_30d_spx_ewm5"] = p["delta_atm_30d_spx"].ewm(
            span=5, adjust=False, min_periods=1
        ).mean()
    if {"hvol_30d", "rv_21d"}.issubset(p.columns) and "vol_blend_30d" not in p.columns:
        p["vol_blend_30d"] = 0.5 * p["hvol_30d"] + 0.5 * p["rv_21d"]
    return p

def build_aligned_env_bundle(
    env_bundle: Dict[str, Any],
    extra_features: Optional[List[str]] = None,
    panel_override: Optional[pd.DataFrame] = None,
    slippage_fn: Optional[Callable[[float, np.ndarray, dict], float]] = None,
) -> Tuple[Dict[str, HedgingEnv], List[str]]:
    """
    Rebuild envs with the same splits/costs used by the GAE policy but with
    additional features appended to the observation window.
    """
    extra = extra_features or []
    if not {"panel", "masks", "config"}.issubset(env_bundle):
        raise ValueError("env_bundle must include 'panel', 'masks', and 'config'")
    panel = panel_override if panel_override is not None else env_bundle["panel"]
    masks = env_bundle["masks"]
    cfg = env_bundle["config"]

    base_features = list(cfg.get("features", []))
    requested = list(dict.fromkeys(base_features + list(extra)))
    missing = [c for c in requested if c not in panel.columns]
    if missing:
        raise KeyError(f"Cannot build envs: missing columns {missing}")

    scaler, _, _ = make_scaler(panel, requested, masks["train"])
    envs = {}
    for split, mask in masks.items():
        envs[split] = make_env(
            panel,
            mask,
            requested,
            window=cfg.get("window"),
            txn_cost_bps=cfg.get("txn_cost_bps"),
            scaler=scaler,
            pos_limit=cfg.get("pos_limit"),
            reward_scale=cfg.get("reward_scale", 1e4),
            rebalance_every=cfg.get("rebalance_every", 1),
            slippage_bps=cfg.get("slippage_bps", 0.0),
            slippage_fn=slippage_fn,
        )
    return envs, requested

def adaptive_vol_target_policy(
    feature_idx: int,
    window: int = 5,
    min_target: float = 0.10,
    max_target: float = 0.35,
) -> Callable[[np.ndarray], float]:
    """
    Simple adaptive volatility targeting rule that scales exposure down when
    blended volatility exceeds the rolling target.
    """
    def policy(obs: np.ndarray) -> float:
        tail = obs[-window:, feature_idx]
        vol = float(np.nanmean(tail))
        if not np.isfinite(vol) or vol <= 0:
            return 0.0
        target = float(np.clip(0.15 + 0.5 * (vol - 0.20), min_target, max_target))
        hedge = 1.0 - target / max(vol, 1e-6)
        return float(np.clip(hedge, -1.0, 1.0))
    return policy

def make_tuned_baselines(env_bundle: Dict[str, Any]) -> Dict[str, BaselineSpec]:
    """
    Convenience factory that returns tuned heuristic baselines aligned with the
    exact panel/cost settings used by the GAE policy. Each entry exposes the
    envs dict (train/valid/test), the features fed to that policy, and the
    callable policy itself.
    """
    required = {"panel", "masks", "config"}
    missing = required - set(env_bundle.keys())
    if missing:
        raise ValueError(f"env_bundle missing keys: {missing}")

    augmented_panel = augment_panel_with_baseline_features(env_bundle["panel"])
    bundle = dict(env_bundle)
    bundle["panel"] = augmented_panel

    specs: Dict[str, BaselineSpec] = {}

    mom_envs, mom_features = build_aligned_env_bundle(bundle, ["close_spy_ratio5"])
    mom_idx = mom_features.index("close_spy_ratio5")
    specs["momentum"] = BaselineSpec(
        policy=momentum_policy(feature_idx=mom_idx, lookback=15, threshold=1e-3),
        envs=mom_envs,
        features=mom_features,
        description="Momentum, 15-day lookback with turnover gate on smoothed SPY",
    )

    delta_envs, delta_features = build_aligned_env_bundle(bundle, ["delta_atm_30d_spx_ewm5"])
    delta_idx = delta_features.index("delta_atm_30d_spx_ewm5")
    specs["delta_hedge"] = BaselineSpec(
        policy=delta_hedge_policy(lambda obs, idx=delta_idx: obs[-1, idx]),
        envs=delta_envs,
        features=delta_features,
        description="Vendor ATM delta hedge smoothed with 5-day EWMA",
    )

    vol_envs, vol_features = build_aligned_env_bundle(bundle, ["vol_blend_30d"])
    vol_idx = vol_features.index("vol_blend_30d")
    specs["vol_target"] = BaselineSpec(
        policy=adaptive_vol_target_policy(vol_idx),
        envs=vol_envs,
        features=vol_features,
        description="Adaptive volatility targeting on blended realized/implied vol",
    )

    # Long-SPY hold as a canonical baseline (matches env costs/limits)
    hold_envs, hold_features = build_aligned_env_bundle(bundle, bundle["config"]["features"])
    specs["long_spy"] = BaselineSpec(
        policy=hold_policy(bundle["config"].get("pos_limit", 1.0)),
        envs=hold_envs,
        features=hold_features,
        description="Hold long underlying at pos_limit",
    )

    # VIX-based overlays (band and vol-scaling)
    if "vix" in augmented_panel.columns:
        vix_envs, vix_features = build_aligned_env_bundle(bundle, ["vix"])
        vix_idx = vix_features.index("vix")
        vix_mean = float(augmented_panel.loc[bundle["masks"]["train"], "vix"].mean())
        vix_std = float(augmented_panel.loc[bundle["masks"]["train"], "vix"].std(ddof=1))
        specs["vix_band"] = BaselineSpec(
            policy=vix_band_policy(
                vix_idx=vix_idx,
                low=20.0,
                high=30.0,
                pos_low=0.0,
                pos_high=bundle["config"].get("pos_limit", 1.0),
                vix_mean=vix_mean,
                vix_std=vix_std,
            ),
            envs=vix_envs,
            features=vix_features,
            description="Simple VIX band overlay (reduce in calm, add in stress)",
        )
        specs["vix_vol_target"] = BaselineSpec(
            policy=vix_vol_target(
                vix_idx=vix_idx,
                vix_target=20.0,
                max_pos=bundle["config"].get("pos_limit", 1.0),
                vix_mean=vix_mean,
                vix_std=vix_std,
            ),
            envs=vix_envs,
            features=vix_features,
            description="Scale hedge with VIX level to target volatility",
        )

    # Placeholder put-protection comparator (zero hedge; cost modeled externally if desired)
    specs["put_protection"] = BaselineSpec(
        policy=hold_policy(0.0),
        envs=hold_envs,
        features=hold_features,
        description="Static zero overlay (proxy for paying premium; add cost externally)",
    )

    return specs
