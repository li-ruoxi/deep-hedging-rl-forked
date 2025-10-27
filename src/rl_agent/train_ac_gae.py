#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assumes:
  from simulator.env import HedgingEnv
  from rl_agent.trainer import evaluate_env

Actor–Critic (GAE) training for HedgingEnv.
- No-leakage splits & scaler via rl_agent.experiment.build_envs
- Policy + Value network, GAE advantages
- Entropy + cosine LR schedules, grad clipping
- Best checkpoint by VALID Sharpe; optional artifact bundle

Run:
  python -m rl_agent.train_ac_gae \
    --data_path data/processed/cleaned/hedging_panel_spx.csv \
    --features vix,rate_10y,iv_atm_30d_spx,iv_ts_slope_spx,iv_skew_30d_spx \
    --window 45 --txn_cost_bps 1.0 --pos_limit 2.0 \
    --hidden 128 --lr 1e-3 --entropy_start 0.5 --steps 8000 \
    --save_ckpt models/run1/best.pt --save_config models/run1/config.json
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ------------ Shared utilities (your repo) -------------
from rl_agent.experiment import build_envs, deterministic_rewards, ArtifactLogger
from rl_agent.trainer import evaluate_env

# =======================================================
# Utils
# =======================================================
def set_seed(seed: int = 42):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def linear_anneal(step, total, start, end=0.0):
    t = min(step / max(total, 1), 1.0)
    return start + (end - start) * t

def cosine_lr(base_lr, step, total):
    # Cosine schedule from base_lr → ~0
    return base_lr * (0.5 * (1.0 + math.cos(math.pi * min(step, total) / max(total, 1))))

# =======================================================
# Model
# =======================================================
class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, init_log_std: float = -0.3):
        super().__init__()
        self.input_dim = input_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.v  = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.tensor([init_log_std], dtype=torch.float32))
        self.obs_fix = None  # set in train() -> policy.obs_fix = to_fixed

    def _to_tensor(self, x, device="cpu"):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=device)
        return x.to(device)

    def forward(self, x: torch.Tensor):
        # returns (mu, std, v) with gradients
        if x.dim() == 1:
            x = x.view(1, -1)
        else:
            x = x.reshape(x.size(0), -1)
        h  = self.backbone(x)
        mu = torch.tanh(self.mu(h))         # [B,1]
        v  = self.v(h).squeeze(-1)          # [B]
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, v

    def act(self, x, deterministic: bool = False, device: str = "cpu"):
        if self.obs_fix is not None and isinstance(x, np.ndarray):
            x = self.obs_fix(x)  # -> fixed 1D length input_dim
        x = self._to_tensor(x, device=device)

        mu, std, _ = self.forward(x)
        a = mu if deterministic else (mu + std * torch.randn_like(std))

        # ↓↓↓ RETURN DETACHED NUMPY (important for evaluate_env/env.step)
        a_out   = a.clamp(-1, 1)[0].detach().cpu().numpy()
        mu_out  = mu[0].detach().cpu().numpy()
        std_out = std[0].detach().cpu().numpy()
        return a_out, mu_out, std_out


# =======================================================
# GAE + Rollout
# =======================================================
@torch.no_grad()

def compute_gae(rews, vals, dones, gamma=0.99, lam=0.95):
    """
    rews: np.array shape [T]
    vals: np.array shape [T+1] (bootstrap at T already appended)
    dones: np.array[bool] shape [T]
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        delta = rews[t] + (0 if dones[t] else gamma * vals[t + 1]) - vals[t]
        gae = delta + gamma * lam * (0 if dones[t] else gae)
        adv[t] = gae
    ret = adv + vals[:-1]
    return adv, ret

def make_obs_fixer(window: int, n_features: int, dtype=np.float32):
    input_dim = window * n_features

    def to_fixed(obs):
        arr = np.asarray(obs, dtype=dtype)

        # Case A: 1D vector (n_features,)
        if arr.ndim == 1:
            out = np.zeros((window, n_features), dtype=dtype)
            out[-1] = arr[:n_features]
            return out.reshape(-1)

        # Case B: 2D matrix (T, n_features) or (n_features, T)
        if arr.ndim == 2:
            h, w = arr.shape
            if w == n_features:
                mat = arr
            elif h == n_features and w != n_features:
                # transpose if user/env gave (n_features, T)
                mat = arr.T
                h, w = mat.shape
            else:
                # fallback: keep last axis as features if reasonable
                if n_features in arr.shape:
                    ax = 1 if arr.shape[1] == n_features else 0
                    mat = arr if ax == 1 else arr.swapaxes(0, 1)
                    h, w = mat.shape
                else:
                    # flatten and right-pad
                    flat = arr.reshape(-1)
                    out = np.zeros((input_dim,), dtype=dtype)
                    k = min(input_dim, flat.size)
                    out[-k:] = flat[-k:]
                    return out

            out = np.zeros((window, n_features), dtype=dtype)
            k = min(window, h)
            out[-k:] = mat[-k:, :n_features]
            return out.reshape(-1)

        # Case C: anything else → flatten, then right-pad/truncate
        flat = arr.reshape(-1)
        out = np.zeros((input_dim,), dtype=dtype)
        k = min(input_dim, flat.size)
        out[-k:] = flat[-k:]
        return out

    return to_fixed

# in rollout_episode
def rollout_episode(env, policy: PolicyValueNet, to_fixed, device="cpu",
                    deterministic=False, max_steps: int | None = None):
    obs = env.reset()
    rews, dones = [], []
    logps_t, states_t, vals_t = [], [], []
    steps = 0
    pos_limit = float(getattr(env, "pos_limit", 1.0))

    while True:
        x_np = to_fixed(obs)
        x    = torch.as_tensor(x_np, dtype=torch.float32, device=device)

        # get mu, std, v with grad; draw action separately
        mu, std, v = policy.forward(x)
        dist = torch.distributions.Normal(mu, std)
        z = mu if deterministic else dist.rsample()
        logp = dist.log_prob(z).sum()
        action_tensor = torch.clamp(z, -pos_limit, pos_limit)

        action = action_tensor.detach().cpu().numpy()
        obs_next, r, done, _ = env.step(action)
        rews.append(float(r)); dones.append(bool(done))
        logps_t.append(logp); states_t.append(x); vals_t.append(v.detach())

        obs = obs_next
        steps += 1
        if done or (max_steps is not None and steps >= max_steps):
            break

    if dones and not dones[-1]:
        with torch.no_grad():
            x_boot = torch.as_tensor(to_fixed(obs), dtype=torch.float32, device=device)
            _, _, v_boot = policy.forward(x_boot.reshape(1, -1))
            bootstrap_v = float(v_boot.squeeze().cpu().item())
    else:
        bootstrap_v = 0.0

    vals_boot = np.asarray([vv.cpu().item() for vv in vals_t] + [bootstrap_v], dtype=np.float32)
    return (
        np.asarray(rews, np.float32),
        torch.stack(logps_t),
        vals_boot,
        np.asarray(dones, bool),
        torch.stack(states_t),
        steps,
    )


# =======================================================
# Training
# =======================================================
def train(panel: pd.DataFrame,
          state_cols: list[str],
          train_end="2017-12-31",
          valid_end="2019-12-31",
          window=45,
          txn_cost_bps=1.0,
          pos_limit=2.0,
          hidden=128,
          lr=1e-3,
          steps=8000,
          entropy_start=0.5,
          max_rollout_steps: int | None = None,
          track_history: bool = False,
          return_history: bool = False,
          device="cpu",
          seed=42,
          save_ckpt: str | None = None,
          save_config: str | None = None,
          save_outdir: str | None = None):
    set_seed(seed)

    return_history = return_history or track_history

    # ----- envs / scaler (no leakage) -----
    envs = build_envs(panel, state_cols, train_end, valid_end,
                      window=window, txn_cost_bps=txn_cost_bps, pos_limit=pos_limit)
    env_tr, env_va, env_te = envs["env_tr"], envs["env_va"], envs["env_te"]
    n_features = len(state_cols)
    input_dim  = window * n_features
    to_fixed   = make_obs_fixer(window, n_features)
    print(f"Env(train) ready. input_dim={input_dim} (window={window}, n_features={n_features})")

    # ----- model + optimizer -----
    policy = PolicyValueNet(input_dim=input_dim, hidden=hidden).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    policy.obs_fix = to_fixed

    best = {'sharpe': -np.inf, 'sdict': None}
    gamma, lam = 0.99, 0.95

    history = None
    if return_history:
        history = {
            "step": [],
            "episode_return": [],
            "episode_len": [],
            "entropy_coef": [],
            "policy_loss": [],
            "value_loss": [],
            "eval_step": [],
            "train_sharpe": [],
            "valid_sharpe": [],
        }

    for step in range(1, steps + 1):
        # schedules
        ent_horizon = int(0.35 * steps)               # faster fade
        ent_coef = linear_anneal(step, ent_horizon, start=entropy_start, end=0.0)
        #ent_coef = linear_anneal(step, steps, start=entropy_start, end=0.0)
        for g in opt.param_groups:
            g['lr'] = cosine_lr(lr, step, steps)

        # rollout on TRAIN
        rews, logps_t, vals_boot, dones, states_t, ep_len = rollout_episode(
            env_tr, policy, to_fixed=to_fixed, device=device,
            deterministic=False, max_steps=max_rollout_steps
        )

        # GAE
        adv, ret = compute_gae(rews, vals_boot, dones, gamma=gamma, lam=lam)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)

        # recompute policy/value on saved states WITH gradient
        mu_batch, std_batch, v_pred = policy.forward(states_t)
        # policy loss (use stored logps_t OR recompute; logps_t is fine)
        if adv_t.numel() > 1:
            denom = adv_t.std(unbiased=False)
            if denom > 1e-6:
                adv_norm = (adv_t - adv_t.mean()) / (denom + 1e-8)
            else:
                adv_norm = adv_t - adv_t.mean()
        else:
            adv_norm = adv_t
        policy_loss = -(logps_t * adv_norm).mean()
        # value loss
        value_loss = 0.5 * ((v_pred - ret_t) ** 2).mean()
        # analytic Gaussian entropy per state: 0.5*log(2πeσ^2)
        entropy = (0.5 * (1.0 + math.log(2 * math.pi)) + torch.log(std_batch)).sum(dim=1).mean()

        loss = policy_loss + value_loss - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if history is not None:
            history["step"].append(step)
            history["episode_return"].append(float(np.sum(rews)))
            history["episode_len"].append(int(ep_len))
            history["entropy_coef"].append(float(ent_coef))
            history["policy_loss"].append(float(policy_loss.detach().cpu().item()))
            history["value_loss"].append(float(value_loss.detach().cpu().item()))

        if step % 50 == 0:
            tr = evaluate_env(env_tr, policy, deterministic=True)
            va = evaluate_env(env_va, policy, deterministic=True)
            print(f"[{step:05d}] "
                  f"Train Sharpe {tr['sharpe']:.3f} | "
                  f"Valid Sharpe {va['sharpe']:.3f} | "
                  f"LR {opt.param_groups[0]['lr']:.2e} | Ent {ent_coef:.3f}")

            if history is not None:
                history["eval_step"].append(step)
                history["train_sharpe"].append(float(tr['sharpe']))
                history["valid_sharpe"].append(float(va['sharpe']))

            if va['sharpe'] > best['sharpe']:
                best['sharpe'] = va['sharpe']
                best['sdict'] = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                if save_ckpt:
                    Path(save_ckpt).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best['sdict'], save_ckpt)
                    if save_config:
                        cfg = dict(
                            features=state_cols, window=window, txn_cost_bps=txn_cost_bps,
                            pos_limit=pos_limit, hidden=hidden, lr=lr, steps=steps,
                            entropy_start=entropy_start, train_end=train_end, valid_end=valid_end,
                        )
                        Path(save_config).write_text(json.dumps(cfg, indent=2))

    if best['sdict'] is not None:
        policy.load_state_dict(best['sdict'])
        print(f"Loaded best checkpoint (valid Sharpe={best['sharpe']:.3f}).")

    # Final evaluation
    tr = evaluate_env(env_tr, policy, deterministic=True)
    va = evaluate_env(env_va, policy, deterministic=True)
    te = evaluate_env(env_te, policy, deterministic=True)
    print("FINAL:", json.dumps({"train": tr, "valid": va, "test": te}, indent=2))

    # Optional artifact bundle (results + curves + config)
    if save_outdir:
        out = Path(save_outdir); out.mkdir(parents=True, exist_ok=True)
        logger = ArtifactLogger(out)
        curves = {
            "train": deterministic_rewards(env_tr, policy),
            "valid": deterministic_rewards(env_va, policy),
            "test":  deterministic_rewards(env_te, policy),
        }
        logger.save(
            split_metrics={"train": tr, "valid": va, "test": te},
            curves=curves,
            config=dict(
                features=state_cols, window=window, lr=lr, steps=steps,
                train_end=train_end, valid_end=valid_end,
                txn_cost_bps=txn_cost_bps, pos_limit=pos_limit, hidden=hidden
            ),
        )

    if history is not None:
        history["final"] = {"train": tr, "valid": va, "test": te}

    if return_history:
        return policy, (tr, va, te), history
    return policy, (tr, va, te)

# =======================================================
# CLI
# =======================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--features", type=str, required=True,
                   help="Comma-separated feature list")
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--txn_cost_bps", type=float, default=1.0)
    p.add_argument("--pos_limit", type=float, default=2.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--entropy_start", type=float, default=0.5)
    p.add_argument("--max_rollout_steps", type=int, default=0,
                   help="If >0, truncate each training rollout to this many steps")
    p.add_argument("--track_history", action="store_true",
                   help="Record training metrics and return them from train()")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress training logs")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_end", type=str, default="2017-12-31")
    p.add_argument("--valid_end", type=str, default="2019-12-31")
    p.add_argument("--save_ckpt", type=str, default="", help="Path to save best .pt")
    p.add_argument("--save_config", type=str, default="", help="Path to save run config .json")
    p.add_argument("--save_outdir", type=str, default="", help="If set, save results/curves/config here")
    return p.parse_args()

def main():
    args = parse_args()
    panel = pd.read_csv(args.data_path)
    feats = [c.strip() for c in args.features.split(",") if c.strip()]
    missing = [c for c in feats if c not in panel.columns]
    if missing:
        raise ValueError(f"Missing features in panel: {missing}")
    train(
        panel=panel,
        state_cols=feats,
        train_end=args.train_end,
        valid_end=args.valid_end,
        window=args.window,
        txn_cost_bps=args.txn_cost_bps,
        pos_limit=args.pos_limit,
        hidden=args.hidden,
        lr=args.lr,
        steps=args.steps,
        entropy_start=args.entropy_start,
        max_rollout_steps=(args.max_rollout_steps or None),
        track_history=args.track_history,
        device=args.device,
        seed=args.seed,
        save_ckpt=(args.save_ckpt or None),
        save_config=(args.save_config or None),
        save_outdir=(args.save_outdir or (str(Path(args.save_ckpt).parent) if args.save_ckpt else None)),
    )

if __name__ == "__main__":
    main()
