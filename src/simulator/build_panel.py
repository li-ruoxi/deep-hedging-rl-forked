# simulator/build_panel.py
from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
from .features import make_option_features
import numpy as np

def _guarded_ffill_on_staleness(panel: pd.DataFrame, feat_cols, max_ff_days: int) -> pd.DataFrame:
    if not feat_cols or not max_ff_days or max_ff_days <= 0:
        return panel

    df = panel.sort_values("date").reset_index(drop=True).copy()

    # Build an index-aligned datetime Series
    idx = pd.to_datetime(df["date"]).values
    idx_s = pd.Series(idx, index=np.arange(len(df)), dtype="datetime64[ns]")

    # Last time any IV feature was observed
    has_any = df[feat_cols].notna().any(axis=1).to_numpy()
    last_seen = pd.Series(pd.NaT, index=idx_s.index, dtype="datetime64[ns]")
    last_seen.loc[np.where(has_any)[0]] = idx[has_any]
    last_seen = last_seen.ffill()

    # Robust staleness in days (works across pandas versions)
    staleness = (idx_s - last_seen).dt.days.astype("float64")
    # Forward fill then blank out stale entries
    df[feat_cols] = df[feat_cols].ffill()
    df.loc[staleness > float(max_ff_days), feat_cols] = np.nan
    return df


def build_sim_panel(
    market_df: pd.DataFrame,
    spx_clean_dir: Path,
    include_spy: bool = False,
    spy_clean_dir: Path | None = None,
    act_at_open: bool = False,
    ffill_limit: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """Compose the simulation panel from market data and precomputed option features."""
    if "date" not in market_df.columns:
        raise ValueError("market_df must contain a 'date' column")

    from .features import make_option_features
    panel = market_df.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["date"]).reset_index(drop=True)

    fe_spx = make_option_features(spx_clean_dir, "spx")
    panel = panel.merge(fe_spx, on="date", how="left")

    if include_spy and spy_clean_dir is not None:
        fe_spy = make_option_features(spy_clean_dir, "spy")
        panel = panel.merge(fe_spy, on="date", how="left")
    if include_spy and spy_clean_dir is None:
        raise ValueError("spy_clean_dir must be provided when include_spy=True")

    # Feature columns to guard-ffill (IVs + derived IVs)
    feat_cols = [c for c in panel.columns if c.startswith(("iv_atm","iv_ts_slope","iv_skew"))]

    # Guarded forward-fill by staleness measured in calendar days
    if ffill_limit:
        panel = _guarded_ffill_on_staleness(panel, feat_cols, max_ff_days=ffill_limit)

    panel = panel.sort_values("date").reset_index(drop=True)

    # --- Forward return definition is explicit by action time ---
    if act_at_open:
        if "open_spy" in panel.columns:
            price_col = "open_spy"
            panel["ret_fwd"] = panel[price_col].shift(-1) / panel[price_col] - 1.0
        else:
            fallback_cols = ["open_gspc", "close_spy", "close_gspc"]
            price_col = next((c for c in fallback_cols if c in panel.columns), None)
            if price_col is None:
                raise ValueError(
                    "Cannot compute forward returns: expected 'open_spy' "
                    "or a fallback like 'open_gspc'/'close_spy'/'close_gspc'."
                )
            warnings.warn(
                f"'open_spy' not found; using '{price_col}' to compute forward returns.",
                RuntimeWarning,
            )
            if price_col.startswith("open"):
                panel["ret_fwd"] = panel[price_col].shift(-1) / panel[price_col] - 1.0
            else:
                panel["ret_fwd"] = panel[price_col].pct_change().shift(-1)
    else:
        close_candidates = ["close_spy", "close_gspc"]
        price_col = next((c for c in close_candidates if c in panel.columns), None)
        if price_col is None:
            raise ValueError(
                "Cannot compute forward returns: provide a close price column "
                "(e.g., 'close_spy' or 'close_gspc')."
            )
        panel["ret_fwd"] = panel[price_col].pct_change().shift(-1)

    panel = panel.dropna(subset=["ret_fwd"]).reset_index(drop=True)

    # Default state features (keep only the ones that exist)
    state_cols = [
        "iv_atm_30d_spx","iv_ts_slope_spx","iv_skew_30d_spx",
        "vix","rate_10y","rv_21d","hvol_30d","hvol_91d"
    ]
    state_cols = [c for c in state_cols if c in panel.columns]

    if not state_cols:
        raise ValueError("No state features found in panel; check option feature builds.")

    return panel, state_cols
