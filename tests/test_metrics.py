from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.env import HedgingEnv
from simulator.rewards import pnl_only
from rl_agent.experiment import deterministic_rewards
from rl_agent.metrics import combo_summary, nav_from_bps


class _ConstPolicy:
    def __init__(self, value: float):
        self.value = float(value)

    def act(self, obs, deterministic: bool = True):
        return (self.value, None, None)


def _make_env(rows: int = 30) -> HedgingEnv:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "feature": np.linspace(0.0, 1.0, rows),
            "ret_fwd": np.full(rows, 0.001),
        }
    )
    return HedgingEnv(
        df=df,
        features=["feature"],
        reward_fn=pnl_only,
        window=5,
        txn_cost_bps=0.0,
    )


def test_combo_summary_matches_manual_blend():
    env = _make_env()
    policy_main = _ConstPolicy(1.0)
    policy_base = _ConstPolicy(0.0)
    blend = 0.3

    main_r = deterministic_rewards(env, policy_main)
    base_r = deterministic_rewards(env, policy_base)
    expected_rewards = blend * main_r + (1.0 - blend) * base_r
    expected_nav = nav_from_bps(expected_rewards)

    summary = combo_summary(env, policy_main, policy_base, blend, split="test")

    assert summary["split"] == "test"
    assert np.allclose(summary["rewards"], expected_rewards)
    assert np.allclose(summary["nav"], expected_nav)
    assert np.isclose(summary["mean_bps"], expected_rewards.mean())
    assert summary["max_drawdown"] <= 0.0
