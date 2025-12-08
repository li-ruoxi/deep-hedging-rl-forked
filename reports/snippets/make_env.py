from rl_agent.experiment import make_splits, make_scaler
from simulator.env import HedgingEnv
from simulator.rewards import reward_bps

state_cols = [
    "iv_atm_30d_spx","iv_ts_slope_spx","iv_skew_30d_spx",
    "vix","rate_10y","rv_21d","hvol_30d","hvol_91d",
]
p, m_tr, m_va, m_te, TRAIN_END, VALID_END = make_splits(panel, "2017-12-31", "2019-12-31")
scaler, mu, sg = make_scaler(p, state_cols, m_tr)

env = HedgingEnv(
    df=p[m_te].reset_index(drop=True),
    features=state_cols,
    reward_fn=lambda pnl, info: reward_bps(pnl, info, 1e4),
    window=30, txn_cost_bps=1.0, pos_limit=1.0,
    scaler=scaler, hold_on_nan=True,
)
obs = env.reset()
