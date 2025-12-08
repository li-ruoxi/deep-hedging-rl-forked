import numpy as np
import pandas as pd
import torch

from simulator.features import make_option_features
from simulator.env import HedgingEnv
from simulator.rewards import pnl_only
from rl_agent.train_ac_gae import PolicyValueNet, make_obs_fixer, rollout_episode


def test_make_option_features_on_small_sample(tmp_path):
    clean_dir = tmp_path / "spx_clean"
    clean_dir.mkdir()

    data = [
        # ATM 30d + wings
        {"date": "2020-01-01", "tenor_d": 30, "put_call": "C", "bid": 1.0, "ask": 1.2, "iv": 0.20, "delta": 0.50},
        {"date": "2020-01-01", "tenor_d": 30, "put_call": "P", "bid": 1.0, "ask": 1.1, "iv": 0.21, "delta": -0.25},
        {"date": "2020-01-01", "tenor_d": 30, "put_call": "C", "bid": 1.0, "ask": 1.1, "iv": 0.22, "delta": 0.25},
        # ATM 60d + wings
        {"date": "2020-01-01", "tenor_d": 60, "put_call": "C", "bid": 1.5, "ask": 1.7, "iv": 0.24, "delta": 0.50},
        {"date": "2020-01-01", "tenor_d": 60, "put_call": "P", "bid": 1.4, "ask": 1.6, "iv": 0.25, "delta": -0.25},
        {"date": "2020-01-01", "tenor_d": 60, "put_call": "C", "bid": 1.4, "ask": 1.5, "iv": 0.26, "delta": 0.25},
        # ATM 91d for term-structure slope
        {"date": "2020-01-01", "tenor_d": 91, "put_call": "C", "bid": 1.3, "ask": 1.5, "iv": 0.23, "delta": 0.50},
    ]
    df = pd.DataFrame(data)
    df.to_parquet(clean_dir / "part.parquet", index=False)

    feats = make_option_features(clean_dir, "spx")
    required = [
        "iv_atm_30d_spx",
        "iv_ts_slope_spx",
        "iv_skew_30d_spx",
        "iv_atm_60d_spx",
        "iv_skew_60d_spx",
    ]
    for col in required:
        assert col in feats
        assert feats[col].notna().any()


def test_rollout_episode_zeroes_logp_when_trade_skipped():
    n = 12
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "feature": np.linspace(0.0, 1.0, n),
            "ret_fwd": np.full(n, 0.001),
        }
    )
    env = HedgingEnv(
        df=panel,
        features=["feature"],
        reward_fn=pnl_only,
        window=3,
        txn_cost_bps=0.0,
        rebalance_every=2,  # every other step is held
    )

    to_fixed = make_obs_fixer(window=3, n_features=1)
    policy = PolicyValueNet(input_dim=3, hidden=8)

    rews, logps_t, _, _, _, steps = rollout_episode(env, policy, to_fixed, deterministic=True)
    executed = (np.arange(steps) % 2) == 0
    logps_np = logps_t.detach().cpu().numpy()

    # Non-executed steps should contribute zero log-prob; executed steps should have signal.
    assert np.allclose(logps_np[~executed], 0.0)
    assert not np.allclose(logps_np[executed], 0.0)
