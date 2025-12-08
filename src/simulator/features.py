# simulator/features.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

# ---------- helpers ----------
# --- at top of _prep_base ---
def _prep_base(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "date" in x:
        x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
        x = x.sort_values("date").reset_index(drop=True)
    if "put_call" in x:
        x["put_call"] = (x["put_call"].astype(str)
                           .str.strip().str.upper().str[0]
                           .map({"C":"C","P":"P"}))
    if "delta" in x and x["delta"].abs().quantile(0.99) > 2:
        x["delta"] = x["delta"] / 100.0  # vendor ±100 → [-1,1]
    if "delta" in x and x["delta"].abs().max() > 1.01:
        raise ValueError("Delta appears out of [-1,1]. Convert percent deltas to fraction before feature selection.")
    return x

def _pick_atm_tolerant(
    df: pd.DataFrame,
    target_dte: int,
    base_delta=0.50,
    cols=("iv",),
):
    x = _prep_base(df)
    for dte_tol, d_tol in [(15, 0.15), (25, 0.20), (35, 0.25)]:
        z = x[x["tenor_d"].sub(target_dte).abs() <= dte_tol].copy()
        if z.empty: continue
        z["abs_delta"] = z["delta"].abs()
        z = z[z["abs_delta"].sub(base_delta).abs() <= d_tol]
        if z.empty: continue
        z["dte_diff"] = (z["tenor_d"] - target_dte).abs()
        z["atm_diff"] = (z["abs_delta"] - base_delta).abs()
        z["spread"]   = (z["ask"] - z["bid"]) if {"ask","bid"}.issubset(z.columns) else 0.0
        y = (z.sort_values(["date","dte_diff","atm_diff","spread"])
               .drop_duplicates("date"))
        if len(y):
            for col in cols:
                if col not in y.columns:
                    y[col] = np.nan
            return y[["date", *cols]]
    # fallback: nearest-by-delta within widest DTE
    z = x[x["tenor_d"].sub(target_dte).abs() <= 35].copy()
    if z.empty:
        return pd.DataFrame(columns=["date", *cols])
    z["abs_delta"] = z["delta"].abs()
    z["atm_diff"]  = (z["abs_delta"] - base_delta).abs()
    z["spread"]    = (z["ask"] - z["bid"]) if {"ask","bid"}.issubset(z.columns) else 0.0
    y = (z.sort_values(["date","atm_diff","spread"])
           .drop_duplicates("date"))
    for col in cols:
        if col not in y.columns:
            y[col] = np.nan
    return y[["date", *cols]]

def _pick_25d_wings_by_date(df: pd.DataFrame, target_dte=30):
    x = _prep_base(df); x["abs_delta"] = x["delta"].abs()
    out = None
    for dte_tol, dlt in [(15, 0.05), (25, 0.08), (35, 0.10)]:
        z = x[x["tenor_d"].sub(target_dte).abs() <= dte_tol].copy()
        if z.empty: continue
        z = z[z["abs_delta"].sub(0.25).abs() <= dlt]
        if z.empty: continue
        z["dte_diff"] = (z["tenor_d"] - target_dte).abs()
        z["d_diff"]   = (z["abs_delta"] - 0.25).abs()
        z["spread"]   = (z["ask"] - z["bid"]) if {"ask","bid"}.issubset(z.columns) else 0.0
        best = (z.sort_values(["date","put_call","dte_diff","d_diff","spread"])
                 .groupby(["date","put_call"], as_index=False)
                 .first())
        wide = (best.pivot(index="date", columns="put_call", values="iv")
                     .rename(columns={"P":"iv_put25_30d", "C":"iv_call25_30d"}))
        out = wide if out is None else out.combine_first(wide)

    if out is None:
        z = x[x["tenor_d"].sub(target_dte).abs() <= 35].copy()
        if z.empty:
            return pd.DataFrame(columns=["date","iv_put25_30d","iv_call25_30d","iv_skew_30d"])
        z["d_diff"] = (z["abs_delta"] - 0.25).abs()
        z["spread"] = (z["ask"] - z["bid"]) if {"ask","bid"}.issubset(z.columns) else 0.0
        best = (z.sort_values(["date","put_call","d_diff","spread"])
                 .groupby(["date","put_call"], as_index=False)
                 .first())
        out = (best.pivot(index="date", columns="put_call", values="iv")
                   .rename(columns={"P":"iv_put25_30d", "C":"iv_call25_30d"}))

    out = out.reset_index()
    out["iv_skew_30d"] = out["iv_put25_30d"] - out["iv_call25_30d"]
    return out[["date","iv_put25_30d","iv_call25_30d","iv_skew_30d"]]

# ---------- public API ----------
def make_option_features(clean_dir: Path, prefix: str):
    """
    From cleaned parquet parts -> daily features:
      iv_atm_30d_{prefix}, iv_atm_91d_{prefix}, iv_ts_slope_{prefix},
      iv_put25_30d_{prefix}, iv_call25_30d_{prefix}, iv_skew_30d_{prefix},
      iv_atm_60d_{prefix}, iv_put25_60d_{prefix}, iv_call25_60d_{prefix}, iv_skew_60d_{prefix}
    """
    dset = ds.dataset(clean_dir, format="parquet")
    cols = ["date","tenor_d","put_call","bid","ask","iv","delta"]
    base = dset.to_table(columns=[c for c in cols if c in dset.schema.names]).to_pandas()

    atm30 = _pick_atm_tolerant(base, 30, cols=("iv","delta"))
    if "delta" not in atm30.columns:
        atm30["delta"] = np.nan
    iv30 = atm30[["date","iv"]].rename(columns={"iv": f"iv_atm_30d_{prefix}"})
    delta30 = atm30[["date","delta"]].rename(columns={"delta": f"delta_atm_30d_{prefix}"})
    iv60 = _pick_atm_tolerant(base, 60, cols=("iv","delta")).rename(columns={"iv": f"iv_atm_60d_{prefix}", "delta": f"delta_atm_60d_{prefix}"})
    iv91 = _pick_atm_tolerant(base, 91).rename(columns={"iv": f"iv_atm_91d_{prefix}"})
    feats = iv30.merge(iv91, on="date", how="outer")
    feats = feats.merge(delta30, on="date", how="left")
    feats = feats.merge(iv60[["date", f"iv_atm_60d_{prefix}"]], on="date", how="left")
    feats[f"iv_ts_slope_{prefix}"] = feats[f"iv_atm_91d_{prefix}"] - feats[f"iv_atm_30d_{prefix}"]

    wings = _pick_25d_wings_by_date(base, target_dte=30).rename(columns={
        "iv_put25_30d": f"iv_put25_30d_{prefix}",
        "iv_call25_30d": f"iv_call25_30d_{prefix}",
        "iv_skew_30d": f"iv_skew_30d_{prefix}",
    })
    wings60 = _pick_25d_wings_by_date(base, target_dte=60).rename(columns={
        "iv_put25_30d": f"iv_put25_60d_{prefix}",
        "iv_call25_30d": f"iv_call25_60d_{prefix}",
        "iv_skew_30d": f"iv_skew_60d_{prefix}",
    })

    out = (feats.merge(wings, on="date", how="left")
                 .merge(wings60, on="date", how="left")
                 .sort_values("date").reset_index(drop=True))
    return out
