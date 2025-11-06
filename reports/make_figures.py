from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

# All plots share a unified rocket_r aesthetic for consistency across the paper.
def _style() -> dict[str, tuple[float, float, float]]:
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("rocket_r", 6)
    return {
        "train": palette[0],
        "valid": palette[2],
        "test": palette[4],
        "policy": palette[1],
        "spy": palette[4],
        "bars_policy": palette[1],
        "bars_spy": palette[4],
        "drawdown": palette[5],
    }


def nav_from_bps(bps: np.ndarray) -> np.ndarray:
    r = np.asarray(bps, dtype=float) / 1e4
    return (1.0 + r).cumprod()


def plot_nav_curves(model_dir: Path, out_path: Path) -> None:
    colors = _style()
    splits = ["train", "valid", "test"]
    plt.figure(figsize=(10, 4), dpi=150)
    for split in splits:
        csv = model_dir / f"{split}_bps.csv"
        if not csv.exists():
            continue
        rewards = pd.read_csv(csv)[pd.read_csv(csv).columns[0]].to_numpy()
        equity = nav_from_bps(rewards)
        plt.plot(equity, label=split.capitalize(), linewidth=2.0, color=colors.get(split))
    plt.title("Cumulative NAV (deterministic rollouts)")
    plt.xlabel("Steps")
    plt.ylabel("NAV (× initial)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def test_nav_vs_spy(model_dir: Path, panel_csv: Path, config_json: Path, out_path: Path) -> None:
    colors = _style()
    cfg = json.loads(Path(config_json).read_text())
    valid_end = pd.Timestamp(cfg["valid_end"])
    window = int(cfg["window"])

    panel = pd.read_csv(panel_csv, parse_dates=["date"]).sort_values("date")
    mask_test = panel["date"] > valid_end
    spy_ret = panel.loc[mask_test, "close_spy"].pct_change().shift(-1).dropna().to_numpy()

    rewards_csv = model_dir / "test_bps.csv"
    rewards = pd.read_csv(rewards_csv)[pd.read_csv(rewards_csv).columns[0]].to_numpy() / 1e4
    spy_ret = spy_ret[window:window + len(rewards)]

    nav_model = (1.0 + rewards).cumprod()
    nav_spy = (1.0 + spy_ret).cumprod()

    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(nav_model, label="GAE policy", color=colors["policy"], linewidth=2.0)
    plt.plot(nav_spy, label="SPY (close-to-close)", color=colors["spy"], linewidth=2.0)
    plt.title("Test NAV: Policy vs SPY")
    plt.xlabel("Steps")
    plt.ylabel("NAV (× initial)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def sharpe_comparison_bars(metrics_csv: Path, out_path: Path) -> None:
    colors = _style()
    df = pd.read_csv(metrics_csv)
    subset = df[df["policy"].isin(["gae_policy", "long_spy"])][["policy", "split", "sharpe"]]
    pivot = subset.pivot(index="policy", columns="split", values="sharpe").reindex(["gae_policy", "long_spy"])
    pivot = pivot[[c for c in ["train", "valid", "test"] if c in pivot.columns]]

    ax = pivot.T.plot(kind="bar", figsize=(10, 5), width=0.7,
                      color=[colors["bars_policy"], colors["bars_spy"]])
    ax.set_title("Sharpe Comparison: Policy vs SPY")
    ax.set_ylabel("Sharpe (ann.)")
    ax.set_xlabel("Split")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["GAE policy", "Long SPY"], frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def rolling_sharpe_plot(model_dir: Path, panel_csv: Path, config_json: Path, out_path: Path, win: int = 252) -> None:
    colors = _style()
    cfg = json.loads(Path(config_json).read_text())
    valid_end = pd.Timestamp(cfg["valid_end"])
    window = int(cfg["window"])

    panel = pd.read_csv(panel_csv, parse_dates=["date"]).sort_values("date")
    mask_test = panel["date"] > valid_end
    spy_ret = panel.loc[mask_test, "close_spy"].pct_change().shift(-1).dropna().to_numpy()[window:]
    rewards = pd.read_csv(model_dir / "test_bps.csv")[lambda d: d.columns[0]].to_numpy() / 1e4

    def rolling_sharpe(series: np.ndarray) -> np.ndarray:
        s = pd.Series(series)
        roll = s.rolling(win)
        mean = roll.mean()
        std = roll.std(ddof=1)
        sharpe = (mean / std) * np.sqrt(252)
        return sharpe.to_numpy()

    sharpe_model = rolling_sharpe(rewards)
    sharpe_spy = rolling_sharpe(spy_ret[:len(rewards)])

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(sharpe_model, label="GAE policy", color=colors["policy"], linewidth=2.0)
    plt.plot(sharpe_spy, label="SPY", color=colors["spy"], linewidth=2.0)
    plt.title(f"Rolling Sharpe (window={win}) – Test")
    plt.xlabel("Step")
    plt.ylabel("Sharpe")
    plt.legend(frameon=False, loc="lower left")
    plt.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def drawdown_plot(model_dir: Path, out_path: Path) -> None:
    colors = _style()
    rewards = pd.read_csv(model_dir / "test_bps.csv")[lambda d: d.columns[0]].to_numpy() / 1e4
    equity = (1.0 + rewards).cumprod()
    drawdown = equity / np.maximum.accumulate(equity) - 1.0

    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(drawdown, color=colors["drawdown"], linewidth=2.0)
    plt.title("Test Max Drawdown Curve (GAE policy)")
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def copy_training_histogram(src: Path, out_path: Path) -> None:
    src = Path(src)
    if not src.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(src.read_bytes())


def data_qa_fig(panel_csv: Path, out_path: Path) -> None:
    colors = _style()
    panel = pd.read_csv(panel_csv, parse_dates=["date"]).sort_values("date")
    df = panel.set_index("date")

    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(df.index, df["vix"], label="VIX", color=colors["policy"], linewidth=1.5)
    plt.plot(df.index, df["rate_10y"], label="10Y rate", color=colors["spy"], linewidth=1.5)
    plt.plot(df.index, df["rv_21d"], label="Realized 21d", color=colors["train"], linewidth=1.5, alpha=0.8)
    plt.plot(df.index, df["hvol_30d"], label="Hist vol 30d", color=colors["drawdown"], linewidth=1.5, alpha=0.8)
    plt.title("Data QA: Volatility and Rate Context")
    plt.xlabel("Date")
    plt.ylabel("Level")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    model_dir = Path("models/gae_run1")
    figs = Path("reports/figures")
    panel_csv = Path("data/processed/cleaned/hedging_panel_spx.csv")
    cfg_json = model_dir / "config.json"

    plot_nav_curves(model_dir, figs / "nav_curves.png")
    test_nav_vs_spy(model_dir, panel_csv, cfg_json, figs / "test_nav_vs_spy.png")
    sharpe_comparison_bars(model_dir / "metrics_comparison.csv", figs / "sharpe_comparison.png")
    rolling_sharpe_plot(model_dir, panel_csv, cfg_json, figs / "rolling_sharpe.png", win=252)
    drawdown_plot(model_dir, figs / "drawdown_curve.png")
    copy_training_histogram(Path("reports/figures/gae_episode_return_distribution.png"),
                            figs / "gae_episode_return_distribution.png")
    data_qa_fig(panel_csv, figs / "data_qa.png")
