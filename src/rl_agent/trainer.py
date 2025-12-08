# src/rl_agent/trainer.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .policy_nn import PolicyNetwork, _flatten_obs


@dataclass
class TrainConfig:
    gamma: float = 0.99
    entropy_coef: float = 0.0    # small bonus like 1e-3 if you want exploration
    lr: float = 1e-4
    max_steps: int | None = None # None = run until env done
    max_rollout_steps: int | None = None
    max_episodes: int | None = None
    device: str = "cpu"
    print_every: int = 25
    normalize_returns: bool = True


def _discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Reward-to-go (backward cumulative sum with discount)."""
    y = np.zeros_like(x, dtype=np.float64)
    run = 0.0
    for t in reversed(range(len(x))):
        run = x[t] + gamma * run
        y[t] = run
    return y


@torch.no_grad()
def evaluate_env(env, policy: PolicyNetwork, deterministic: bool = True) -> Dict[str, float]:
    """Deterministic rollout to get simple metrics."""
    obs = env.reset()
    rets: List[float] = []
    while True:
        a, _, _ = policy.act(obs, deterministic=deterministic)
        obs, r, done, info = env.step(a)
        rets.append(float(r))
        if done:
            break

    r = np.asarray(rets, dtype=np.float64)
    mean, std = r.mean(), r.std(ddof=1) if r.size > 1 else 0.0
    sharpe = (mean / std * np.sqrt(252)) if std > 0 else 0.0
    return {"mean": mean, "std": std, "sharpe": sharpe, "steps": len(r)}


def train_reinforce(env,
                    policy: PolicyNetwork,
                    cfg: TrainConfig) -> Dict[str, List[float]]:
    """
    Minimal REINFORCE (episodic policy gradient) with reward-to-go.
    Works with your env: obs (window x feats) -> action in [-1,1].
    """
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    history: Dict[str, List[float]] = {"episode_return": [], "episode_len": [], "eval_sharpe": []}

    total_eps = cfg.max_episodes or 10_000_000
    for ep in range(1, total_eps + 1):
        obs = env.reset()
        pos_limit = float(getattr(env, "pos_limit", 1.0))
        policy.pos_limit = pos_limit
        logps: List[torch.Tensor] = []
        entrs: List[torch.Tensor] = []
        rewards: List[float] = []

        steps = 0
        while True:
            # forward
            x = _flatten_obs(obs).to(cfg.device).unsqueeze(0)
            mu = policy.forward(x)
            std = policy.log_std.exp().clamp(1e-4, 5.0)
            dist = torch.distributions.Normal(mu, std)
            z = dist.rsample()
            a = torch.tanh(z)

            # env step
            act = float(a.squeeze().detach().cpu().item() * pos_limit)
            obs, r, done, info = env.step(act)

            # log prob with tanh correction + entropy
            logp = dist.log_prob(z) - torch.log1p(-a.pow(2) + 1e-6) - math.log(pos_limit)
            logp = logp.sum(dim=-1)  # (1,)
            ent = dist.entropy().sum(dim=-1)

            logps.append(logp)
            entrs.append(ent)
            rewards.append(float(r))
            steps += 1

            if done:
                break
            if cfg.max_steps and steps >= cfg.max_steps:
                break
            if cfg.max_rollout_steps and steps >= cfg.max_rollout_steps:
                break

        # reward-to-go (advantages)
        G = _discount_cumsum(np.asarray(rewards, dtype=np.float64), cfg.gamma)
        if cfg.normalize_returns and G.std(ddof=1) > 0:
            G = (G - G.mean()) / (G.std(ddof=1) + 1e-8)

        # policy loss
        logps_t = torch.stack(logps)           # (T,)
        ent_t   = torch.stack(entrs)           # (T,)
        G_t     = torch.from_numpy(G).float().to(cfg.device)

        loss_pg = -(logps_t * G_t).mean()
        loss_ent = -(cfg.entropy_coef * ent_t.mean())
        loss = loss_pg + loss_ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        # logging
        ep_ret = float(np.sum(rewards))
        history["episode_return"].append(ep_ret)
        history["episode_len"].append(steps)

        if ep % cfg.print_every == 0:
            eval_metrics = evaluate_env(env, policy, deterministic=True)
            history["eval_sharpe"].append(eval_metrics["sharpe"])
            print(f"[EP {ep:04d}] "
                  f"R:{ep_ret: .4f}  T:{steps:4d}  "
                  f"Eval Sharpe:{eval_metrics['sharpe']:.3f}  "
                  f"Mean Ret:{eval_metrics['mean']*1e4: .2f} bp")

    return history

# Re-export canonical helpers to avoid duplication
from simulator.rewards import reward_bps  # noqa: E402,F401
from rl_agent.experiment import deterministic_rewards  # noqa: E402,F401
