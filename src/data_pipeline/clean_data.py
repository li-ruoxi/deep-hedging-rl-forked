from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq




OM_COLS = [
    "secid","date","exdate","cp_flag","strike_price","best_bid","best_offer",
    "volume","open_interest","impl_volatility","delta","gamma","vega","theta",
    "optionid","root","ticker","index_flag","issuer","exercise_style"
]

def csv_to_parquet_parts_om(csv_path: Path, out_dir: Path, chunksize=2_000_000):
    import pandas as pd
    part = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=lambda c: c in OM_COLS,
        parse_dates=["date","exdate"],
        dtype={
            "cp_flag":"category", "ticker":"category", "root":"category",
            "index_flag":"category", "exercise_style":"category", "issuer":"category"
        },
        low_memory=False, chunksize=chunksize
    ):
        # --- normalize dates ---
        for c in ("date","exdate"):
            chunk[c] = pd.to_datetime(chunk[c], errors="coerce").dt.tz_localize(None).dt.normalize()

        # --- rename to canonical ---
        chunk = chunk.rename(columns={
            "exdate":"expiry",
            "cp_flag":"put_call",
            "strike_price":"strike",
            "best_bid":"bid",
            "best_offer":"ask",
            "impl_volatility":"iv",
        })

        # --- underlying ---
        if "underlying" not in chunk:
            chunk["underlying"] = chunk["ticker"].astype("string")
            # fallback if ticker missing:
            mask = chunk["underlying"].isna()
            if mask.any() and "root" in chunk:
                chunk.loc[mask, "underlying"] = chunk.loc[mask, "root"].astype("string")

        # --- compute mid if missing ---
        chunk["mid"] = (chunk["bid"].astype("float32") + chunk["ask"].astype("float32")) / 2.0

        # --- strike unit auto-fix (handles ×1000 dumps) ---
        s = pd.to_numeric(chunk["strike"], errors="coerce")
        if s.max() and s.max() > 100000:   # heuristically detect milli-dollars
            chunk["strike"] = s / 1000.0
        else:
            chunk["strike"] = s.astype("float32")

        # --- downcast numerics for memory ---
        for c in ("bid","ask","mid","last","iv","delta","gamma","vega","theta"):
            if c in chunk:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce", downcast="float")
        for c in ("open_interest","volume"):
            if c in chunk:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce", downcast="unsigned")

        # --- write this chunk as its own part ---
        part += 1
        chunk.to_parquet(out_dir / f"part_{part:04d}.parquet", index=False)


# pick files and convert
def pick_csv(dirpath: Path) -> Path:
    files = sorted(dirpath.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {dirpath}")
    return max(files, key=lambda p: p.stat().st_size)

# Load options data from a directory of parquet parts

CANONICAL_ORDER = [
    "date","underlying","put_call","expiry","strike",
    "bid","ask","mid","last","iv","delta","gamma","vega","theta",
    "open_interest","volume","secid","optionid","root","ticker",
    "index_flag","issuer","exercise_style"
]

def load_options_dir(dirpath: Path, columns=None) -> pd.DataFrame:
    dset = ds.dataset(dirpath, format="parquet")
    tbl = dset.to_table(columns=None)  # all, we've already harmonized at write
    df  = tbl.to_pandas()
    # final guards
    for c in ("date","expiry"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None).dt.normalize()
    # order columns (keep extras at end)
    cols = [c for c in CANONICAL_ORDER if c in df.columns] + [c for c in df.columns if c not in CANONICAL_ORDER]
    return df[cols]

def qc(df, name):
    print(f"[{name}] rows={len(df):,}  {df['date'].min().date()}→{df['date'].max().date()}")
    s = (df["ask"] - df["bid"])
    print(f"  neg_spread%={(s<0).mean():.3%}  zero_spread%={(s==0).mean():.3%}  iv>5%={(df['iv']>5).mean():.3%}")

#################################
# --- Cleaning the dataset ---
#################################

from data_pipeline.config import ROOT, PROCESSED_DIR
SRC_BASE = ROOT / "data" / "processed" / "options_parquet"
DST_BASE = ROOT / "data" / "processed" / "cleaned"

# --- canonical columns to KEEP ---
KEEP = [
    "date","underlying","put_call","expiry","strike",
    "bid","ask","mid","iv","delta","gamma","vega","theta",
    "open_interest","volume","tenor_d"
]

# --- aliases we may see from OptionMetrics ---
ALIASES = {
    "exdate": "expiry",
    "cp_flag": "put_call",
    "strike_price": "strike",
    "best_bid": "bid",
    "best_offer": "ask",
    "impl_volatility": "iv",
}

# we might read some metadata to build "underlying" if needed
META_CANDIDATES = ["ticker","root","exercise_style","index_flag","issuer"]

# read candidates = keep + aliases keys + meta
CANDIDATES = set(KEEP) | set(ALIASES.keys()) | set(META_CANDIDATES)

def _read_part_selective(path: Path) -> pd.DataFrame:
    # list present columns without loading the whole file
    pf = pq.ParquetFile(path)
    names = set(pf.schema.names)
    cols = [c for c in CANDIDATES if c in names]
    tbl = pf.read(columns=cols)
    return tbl.to_pandas()

def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    x = df.rename(columns={k:v for k,v in ALIASES.items() if k in df.columns}).copy()

    # underlying
    if "underlying" not in x.columns:
        if "ticker" in x: x["underlying"] = x["ticker"].astype("string")
        elif "root" in x: x["underlying"] = x["root"].astype("string")
    x["underlying"] = x["underlying"].str.upper()

    # dates
    for c in ("date","expiry"):
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce").dt.tz_localize(None).dt.normalize()

    # mid
    if "mid" not in x.columns and {"bid","ask"} <= set(x.columns):
        x["mid"] = (pd.to_numeric(x["bid"], errors="coerce") +
                    pd.to_numeric(x["ask"], errors="coerce")) / 2.0

    # strike unit sanity (some dumps x1000)
    if "strike" in x.columns:
        s = pd.to_numeric(x["strike"], errors="coerce")
        x["strike"] = (s/1000.0) if (s.max() and s.max() > 100_000) else s.astype("float32")

    # tenor + hygiene filters
    x["tenor_d"] = (x["expiry"] - x["date"]).dt.days
    x = x[(x["bid"] > 0) & (x["ask"] > 0) & (x["ask"] >= x["bid"])]
    x = x[(x["iv"] > 0) & (x["iv"] <= 5.0)]
    x = x[(x["delta"] >= -1.05) & (x["delta"] <= 1.05)]
    x = x[(x["tenor_d"] >= 4) & (x["tenor_d"] <= 730)]

    # downcast for size
    for c in ("bid","ask","mid","iv","delta","gamma","vega","theta"):
        if c in x: x[c] = pd.to_numeric(x[c], errors="coerce", downcast="float")
    for c in ("open_interest","volume","tenor_d"):
        if c in x: x[c] = pd.to_numeric(x[c], errors="coerce", downcast="unsigned")

    # final select/order
    keep = [c for c in KEEP if c in x.columns]
    return x[keep].reset_index(drop=True)

def clean_options_dir(symbol: str, strict_liquidity: bool = False) -> Path:
    src_dir = SRC_BASE / symbol
    dst_dir = DST_BASE / symbol
    dst_dir.mkdir(parents=True, exist_ok=True)

    parts = sorted(src_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts in {src_dir}")

    out_i = 0
    for p in parts:
        chunk = _read_part_selective(p)
        chunk = _clean_chunk(chunk)

        if strict_liquidity:
            if {"open_interest","volume"}.issubset(chunk.columns):
                chunk = chunk[(chunk["open_interest"] > 0) & (chunk["volume"] > 0)]

        out_i += 1
        chunk.to_parquet(dst_dir / f"part_{out_i:04d}.parquet",
                         index=False, compression="zstd")

    # quick manifest
    import pyarrow.dataset as ds
    dset = ds.dataset(dst_dir, format="parquet")
    tbl  = dset.to_table(columns=["date"]) if "date" in dset.schema.names else None
    df   = tbl.to_pandas() if tbl is not None else pd.DataFrame(columns=["date"])
    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    else:
        dates = pd.Series(dtype="datetime64[ns]")
    summary = {
        "symbol": symbol,
        "parts": out_i,
        "rows": len(df),
        "date_min": str(dates.min().date()) if len(dates) else None,
        "date_max": str(dates.max().date()) if len(dates) else None,
        "path": str(dst_dir),
    }
    pd.Series(summary).to_json(dst_dir / "_manifest.json", indent=2)
    return dst_dir

# Function to check for NaNs in a parquet file
def check_nans_in_parquet(directory):
    parquet_files = sorted(directory.glob("part_*.parquet"))
    for file in parquet_files:
        df = pq.read_table(file).to_pandas()
        nan_summary = df.isna().sum()
        print(f"File: {file.name}")
        print(nan_summary[nan_summary > 0])  # Show only columns with NaNs
        print("-" * 40)
