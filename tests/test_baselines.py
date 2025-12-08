import numpy as np

from simulator.baselines import momentum_policy


def test_momentum_policy_threshold_and_lookback():
    # Build a simple observation window with a gentle up-trend.
    obs = np.linspace(1.0, 1.05, 10, dtype=np.float64).reshape(-1, 1)

    # Default behaves like the legacy single-step signal.
    legacy = momentum_policy()(obs)
    assert legacy > 0.0

    # Larger lookback still produces a long signal for the up-trend.
    long_window = momentum_policy(lookback=5)(obs)
    assert long_window > 0.0

    # A high threshold suppresses trading when momentum is weak.
    gated = momentum_policy(lookback=5, threshold=0.02)(obs)
    assert gated == 0.0
