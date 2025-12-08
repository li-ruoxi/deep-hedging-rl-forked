import numpy as np
import pandas as pd

from rl_agent.experiment import (
    build_envs,
    augment_panel_with_baseline_features,
    build_aligned_env_bundle,
    make_tuned_baselines,
    BaselineSpec,
)


def _toy_panel(n=60):
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    base = np.linspace(100, 120, n)
    data = {
        "date": dates,
        "close_spy": base,
        "hvol_30d": np.linspace(0.12, 0.32, n),
        "rv_21d": np.linspace(0.10, 0.25, n),
        "delta_atm_30d_spx": np.linspace(-0.2, 0.2, n),
        "ret_fwd": np.sin(np.linspace(0, 3, n)) * 0.01,
        "iv_atm_30d_spx": np.linspace(0.15, 0.25, n),
        "iv_ts_slope_spx": np.linspace(-0.02, 0.02, n),
        "iv_skew_30d_spx": np.linspace(0.01, -0.01, n),
        "vix": np.linspace(12, 20, n),
        "rate_10y": np.linspace(1.5, 2.5, n),
        "rv_21d_dup": np.linspace(0.1, 0.2, n),  # helper for state_cols uniqueness
    }
    panel = pd.DataFrame(data)
    panel["rv_21d"] = panel["rv_21d"]
    return panel


def test_augment_panel_adds_expected_columns():
    panel = _toy_panel()
    augmented = augment_panel_with_baseline_features(panel)
    assert "close_spy_ratio5" in augmented
    assert "delta_atm_30d_spx_ewm5" in augmented
    assert "vol_blend_30d" in augmented
    # ensure original untouched
    assert "close_spy_ratio5" not in panel


def test_build_aligned_env_bundle_appends_features():
    panel = _toy_panel()
    envs = build_envs(
        panel,
        features=["iv_atm_30d_spx", "iv_ts_slope_spx", "iv_skew_30d_spx"],
        train_end="2010-02-10",
        valid_end="2010-03-15",
        window=3,
        txn_cost_bps=1.0,
        pos_limit=1.0,
    )
    augmented = augment_panel_with_baseline_features(envs["panel"])
    new_envs, features = build_aligned_env_bundle(envs, ["close_spy_ratio5"], panel_override=augmented)
    assert "close_spy_ratio5" in features
    assert set(new_envs.keys()) == {"train", "valid", "test"}


def test_make_tuned_baselines_returns_specs():
    panel = _toy_panel(90)
    envs = build_envs(
        panel,
        features=[
            "iv_atm_30d_spx",
            "iv_ts_slope_spx",
            "iv_skew_30d_spx",
            "vix",
            "rate_10y",
            "rv_21d",
            "hvol_30d",
            "rv_21d_dup",
        ],
        train_end="2010-02-28",
        valid_end="2010-04-15",
        window=5,
        txn_cost_bps=1.0,
        pos_limit=2.0,
    )
    specs = make_tuned_baselines(envs)
    assert {"momentum", "delta_hedge", "vol_target"} <= set(specs.keys())
    for spec in specs.values():
        assert isinstance(spec, BaselineSpec)
        obs = spec.envs["train"].reset()
        out = float(spec.policy(obs))
        assert np.isfinite(out)
