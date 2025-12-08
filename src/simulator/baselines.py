# baselines.py
from __future__ import annotations
import numpy as np
from typing import Callable

def hold_policy(level: float = 0.0) -> Callable[[np.ndarray], float]:
    """Return a constant position (e.g., long SPY overlay)."""
    lvl = float(level)
    return lambda obs: float(np.clip(lvl, -1.0, 1.0))

def no_hedge_policy() -> Callable[[np.ndarray], float]:
    """Always zero position."""
    return lambda obs: 0.0

def momentum_policy(
    feature_idx: int = 0,
    k: float = 1.0,
    lookback: int = 1,
    threshold: float = 0.0,
) -> Callable[[np.ndarray], float]:
    """
    Uses sign of the recent log-return average of the chosen feature.
    Args:
        feature_idx: column inside the observation window to inspect.
        k: scale factor applied to the resulting signal.
        lookback: number of most recent returns to average (>=1).
        threshold: |signal| below this value yields no trade (units of log return).
    """
    def policy(obs: np.ndarray) -> float:
        x = obs[:, feature_idx]
        if len(x) < 2:
            return 0.0
        lk = max(1, int(lookback))
        x = np.asarray(x, float)
        if not np.isfinite(x).all():
            return 0.0
        if np.nanmin(x) > 0:
            r = np.diff(np.log(np.clip(x, 1e-12, None)))
        else:
            r = np.diff(x)
        tail = r[-lk:]
        signal = float(np.nanmean(tail)) if tail.size else 0.0
        if not np.isfinite(signal) or abs(signal) <= float(threshold):
            return 0.0
        sig = np.sign(signal)
        return float(np.clip(k * sig, -1.0, 1.0))
    return policy

def volatility_targeting(feature_idx: int = 0, ann_vol_target: float = 0.15) -> Callable[[np.ndarray], float]:
    """
    Hedge size increases when the selected feature's volatility rises.
    NOTE: This expects a *price-like* feature in obs[:, feature_idx].
    """
    def policy(obs: np.ndarray) -> float:
        x = obs[:, feature_idx]
        x = np.asarray(x, float)
        if not np.isfinite(x).all() or len(x) < 5:
            return 0.0
        # If user passed a strictly positive level series (e.g., VIX), guard logs:
        x = np.clip(x, 1e-12, None)
        r = np.diff(np.log(x))
        vol = np.nanstd(r, ddof=1) * np.sqrt(252.0)
        if not np.isfinite(vol) or vol <= 1e-6:
            return 0.0
        h = 1.0 - (ann_vol_target / vol)
        return float(np.clip(h, -1.0, 1.0))
    return policy


def delta_hedge_policy(delta_fn: Callable[[np.ndarray], float], scale: float = 1.0) -> Callable[[np.ndarray], float]:
    """
    Wrap a delta estimator (from obs window) into a policy that sets hedge = -scale * delta.
    """
    def policy(obs: np.ndarray) -> float:
        try:
            d = float(delta_fn(obs))
        except Exception:
            d = 0.0
        return float(np.clip(-scale * d, -1.0, 1.0))
    return policy


def delta_band_vix(
    vix_idx: int,
    lower_band: float = 0.2,
    upper_band: float = 0.4,
    pos_low: float = 0.0,
    pos_high: float = 1.0,
) -> Callable[[np.ndarray], float]:
    """
    Delta hedge with VIX-adjusted bands:
      - If VIX >= upper_band: hedge toward pos_high
      - If VIX <= lower_band: hedge toward pos_low
      - Else: hold mid-point between pos_low and pos_high
    Clip to [-1, 1]; scale by env.pos_limit outside.
    """
    def policy(obs: np.ndarray) -> float:
        try:
            vix = float(obs[-1, vix_idx]) if obs.ndim == 2 else float(obs[vix_idx])
        except Exception:
            vix = 0.0
        vix = np.nan_to_num(vix, nan=0.0, posinf=0.0, neginf=0.0)
        if vix >= upper_band:
            target = pos_high
        elif vix <= lower_band:
            target = pos_low
        else:
            target = 0.5 * (pos_low + pos_high)
        return float(np.clip(target, -1.0, 1.0))
    return policy


def vix_band_policy(
    vix_idx: int = 0,
    low: float = 20.0,
    high: float = 30.0,
    pos_low: float = 0.0,
    pos_high: float = 1.0,
    vix_mean: float | None = None,
    vix_std: float | None = None,
) -> Callable[[np.ndarray], float]:
    """
    Simple VIX-band overlay:
      - If VIX > high: increase hedge toward +pos_high
      - If VIX < low: reduce hedge toward +pos_low
      - Else: hold mid-point between pos_low and pos_high.
    Actions are clipped to [-1, 1] and should be scaled by env.pos_limit.
    """
    def policy(obs: np.ndarray) -> float:
        try:
            vix = float(obs[-1, vix_idx]) if obs.ndim == 2 else float(obs[vix_idx])
        except Exception:
            vix = 0.0
        vix = np.nan_to_num(vix, nan=0.0, posinf=0.0, neginf=0.0)
        if vix_mean is not None and vix_std not in (None, 0):
            vix = vix_mean + vix_std * vix
        if vix > high:
            target = pos_high
        elif vix < low:
            target = pos_low
        else:
            target = 0.5 * (pos_low + pos_high)
        return float(np.clip(target, -1.0, 1.0))
    return policy


def vix_vol_target(
    vix_idx: int = 0,
    vix_target: float = 20.0,
    max_pos: float = 1.0,
    vix_mean: float | None = None,
    vix_std: float | None = None,
) -> Callable[[np.ndarray], float]:
    """
    Scale hedge notional based on VIX level to target equity volatility.
    Hedge fraction = min(1, VIX / vix_target); clipped to [-max_pos, max_pos].
    """
    def policy(obs: np.ndarray) -> float:
        try:
            vix = float(obs[-1, vix_idx]) if obs.ndim == 2 else float(obs[vix_idx])
        except Exception:
            vix = 0.0
        vix = np.nan_to_num(vix, nan=0.0, posinf=0.0, neginf=0.0)
        if vix_mean is not None and vix_std not in (None, 0):
            vix = vix_mean + vix_std * vix
        frac = vix / vix_target if vix_target > 0 else 0.0
        return float(np.clip(frac, -max_pos, max_pos))
    return policy
