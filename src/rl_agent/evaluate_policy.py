#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from simulator.env import HedgingEnv
from simulator.rewards import reward_bps
from rl_agent.trainer import evaluate_env
from rl_agent.train_ac_gae import PolicyValueNet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--ckpt", required=True, help="path to .pt saved by train_ac_gae")
    p.add_argument("--config", required=True, help="config.json saved by train_ac_gae")
    p.add_argument("--device", default="cpu")
    return p.parse_args()

def main():
    args = parse_args()
    panel = pd.read_csv(args.data_path)
    cfg = json.loads(Path(args.config).read_text())

    # splits
    panel["date"] = pd.to_datetime(panel["date"])
    TRAIN_END = pd.Timestamp(cfg["train_end"])
    VALID_END = pd.Timestamp(cfg["valid_end"])
    m_tr = panel["date"] <= TRAIN_END
    m_va = (panel["date"] > TRAIN_END) & (panel["date"] <= VALID_END)
    m_te = panel["date"] > VALID_END

    mu = panel.loc[m_tr, cfg["features"]].mean()
    sg = panel.loc[m_tr, cfg["features"]].std(ddof=1).replace(0, np.nan).fillna(1.0)
    scaler = lambda obs: (obs - mu.values) / sg.values

    def make_env(mask):
        return HedgingEnv(
            df=panel.loc[mask].reset_index(drop=True),
            features=cfg["features"],
            reward_fn=lambda pnl, info: reward_bps(pnl, info, scale=1e4),
            window=cfg["window"],
            txn_cost_bps=cfg["txn_cost_bps"],
            scaler=scaler,
            hold_on_nan=True,
            pos_limit=cfg["pos_limit"],
        )

    env_tr, env_va, env_te = make_env(m_tr), make_env(m_va), make_env(m_te)
    input_dim = env_tr.reset().size
    net = PolicyValueNet(input_dim=input_dim, hidden=cfg["hidden"]).to(args.device)
    state_dict = torch.load(args.ckpt, map_location=args.device)
    net.load_state_dict(state_dict)

    tr = evaluate_env(env_tr, net, deterministic=True)
    va = evaluate_env(env_va, net, deterministic=True)
    te = evaluate_env(env_te, net, deterministic=True)
    print(json.dumps({"train": tr, "valid": va, "test": te}, indent=2))

if __name__ == "__main__":
    main()
