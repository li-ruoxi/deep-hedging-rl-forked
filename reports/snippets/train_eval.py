from rl_agent.train_ac_gae import train
from rl_agent.experiment import deterministic_rewards

policy, (tr, va, te) = train(
    panel=p,
    state_cols=state_cols,
    train_end=str(TRAIN_END.date()), valid_end=str(VALID_END.date()),
    window=30, txn_cost_bps=1.0, pos_limit=1.0,
    hidden=256, lr=5e-4, steps=2500,
    entropy_start=0.01, entropy_floor=0.01,
    save_ckpt="models/gae_run1/best.pt",
    save_config="models/gae_run1/config.json",
    save_outdir="models/gae_run1",
)

r_test = deterministic_rewards(env, policy)
