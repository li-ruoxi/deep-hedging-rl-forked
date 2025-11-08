from __future__ import annotations

import numpy as np
import pandas as pd

from simulator.env import HedgingEnv
from simulator.rewards import pnl_only


def _toy_panel(n: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {
        "date": dates,
        "feature": np.linspace(0.0, 1.0, n),
        "ret_fwd": np.full(n, 0.001),
    }
    return pd.DataFrame(data)


def test_rebalance_every_skips_intervening_steps():
    panel = _toy_panel()
    env = HedgingEnv(
        df=panel,
        features=["feature"],
        reward_fn=pnl_only,
        window=5,
        txn_cost_bps=0.0,
        rebalance_every=3,
    )

    env.reset()
    executed = []
    positions = []
    for action in [1.0, -1.0, 0.5, -0.5, 0.0, 0.8, -0.2]:
        _, _, done, info = env.step(action)
        executed.append(info["executed"])
        positions.append(info["pos"])
        if done:
            break

    # only every third step should execute a trade
    expected = [idx % 3 == 0 for idx in range(len(executed))]
    assert executed == expected[: len(executed)]
    # positions should only change when executed flag is True
    for idx in range(1, len(positions)):
        if executed[idx]:
            assert positions[idx] != positions[idx - 1]
        else:
            assert positions[idx] == positions[idx - 1]


def test_slippage_bps_increases_costs():
    panel = _toy_panel()
    env = HedgingEnv(
        df=panel,
        features=["feature"],
        reward_fn=pnl_only,
        window=5,
        txn_cost_bps=0.0,
        slippage_bps=10.0,
    )
    env.reset()
    _, _, _, info = env.step(1.0)
    # 10 bps per unit means cost = 10 * 1e-4 * |dpos|
    assert np.isclose(info["cost"], 10.0 * 1e-4 * abs(info["dpos"]))
