# GAEP Run 1 – Summary

Artifacts in this folder capture the actor–critic (GAE) hedge policy trained on the SPX options panel. Key facts for reproducibility and handoff are listed below.

Further testing can be done through a feature sweep or changing the hyperparams

## Training configuration

- Features: `iv_atm_30d_spx`, `iv_ts_slope_spx`, `iv_skew_30d_spx`, `vix`, `rate_10y`, `rv_21d`, `hvol_30d`, `hvol_91d`
- Window: 30 (flattened observation = 240 inputs)
- Transaction cost: 1 bp per |Δposition|
- Position limit: ±1.0
- Network: PolicyValueNet(hidden=256)
- Optimiser: Adam, lr = 5e-4, weight_decay = 1e-4
- Entropy bonus: start 0.01 → floor 0.01
- Rollout cap: 150 steps
- Early stopping: 300-step patience (best checkpoint saved at step ≈350)

## Performance snapshot (deterministic rollout)

| Split | Sharpe | Max DD | Turnover |
|-------|-------:|-------:|---------:|
| Train | 0.085 | -12.40% | 69.72 |
| Valid | 0.410 | -01.00% | 17.00 |
| Test  | 0.376 | -02.80% | 40.18 |

Supporting files:

- `best.pt` – policy weights
- `config.json` – training config used by `train_ac_gae.train`
- `results.json`, `*_bps.csv` – deterministic reward curves per split
- `evaluation_summary.csv` – post-training aggregate metrics
- `nav_curves.png`, `test_reward_hist.png` – plots generated in `04_results.ipynb`

## Loading the policy

```python
from pathlib import Path
import pandas as pd

from rl_agent.deploy import load_policy

panel = pd.read_parquet("data/processed/hedging_panel.parquet")  # example path
policy, cfg, envs = load_policy(Path("models/gae_run1"), panel, device="cpu")
```

The helper returns the restored policy, the saved configuration, and train/valid/test environments ready for deterministic or stochastic rollouts. Use `envs["env_te"].rollout(lambda obs: policy.act(obs, deterministic=True)[0])` to replay the policy on the test split or feed it into a downstream backtest harness.

