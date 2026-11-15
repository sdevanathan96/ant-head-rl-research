"""MAPPO and IPPO (Yu et al. 2022; de Witt et al. 2020).

Both train a stochastic per-ant policy with the clipped PPO objective on the
TEAM reward. They differ in the critic:

    MAPPO  one centralised critic V(s) on the global state  ("shared critic")
    IPPO   a decentralised critic V(o_i) on each ant's own view

Acting is decentralised in both: each ant samples from π(a | o_i).

Credit (how the reward becomes each ant's advantage):

    credit="blend"  (default) each ant's return mixes its own reward with an equal share of the
                    team's: 0.5·own_i + 0.5·team/n. With MAPPO's single V(s) the critic fits the
                    mean of these per-ant returns.
    credit="team"   standard MAPPO: every ant gets the same advantage from the team reward
                    (scaled by 1/n, the mean per-ant reward, so the critic's loss scale matches).
    critic_per_ant  MAPPO only: V(s) has one output per ant, each fitted to that ant's own return.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .networks import AgentNet, mlp


@dataclass
class PPOConfig:
    centralized_critic: bool = True   # True = MAPPO, False = IPPO
    share: bool = True
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    rollout_steps: int = 128          # steps per env per update
    num_envs: int = 8
    agent_id: bool = True             # append a one-hot ant id to shared networks
    credit: str = "blend"             # blend | team (see module docstring)
    critic_per_ant: bool = False      # MAPPO: one V(s) output per ant instead of one shared value


class PPOLearner:
    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, n_actions: int,
                 cfg: PPOConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = cfg = cfg or PPOConfig()
        self.n, self.device = n_agents, torch.device(device)
        torch.manual_seed(seed)
        self.actor = AgentNet(obs_dim, n_actions, n_agents, cfg.hidden, cfg.share, cfg.agent_id).to(self.device)
        if cfg.credit not in ("blend", "team"):
            raise ValueError(f"unknown credit {cfg.credit!r} (expected 'blend' or 'team')")
        if cfg.centralized_critic:
            heads = n_agents if cfg.critic_per_ant else 1
            self.critic: nn.Module = mlp(state_dim, cfg.hidden, heads).to(self.device)
        else:
            self.critic = AgentNet(obs_dim, 1, n_agents, cfg.hidden, cfg.share, cfg.agent_id).to(self.device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.lr, eps=1e-5)

    def _values(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """[..., 1] (centralised) or [..., n] (decentralised, or centralised with critic_per_ant)."""
        if self.cfg.centralized_critic:
            return self.critic(state)
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray, greedy: bool = False):
        """obs [E, n, d], state [E, S] → actions [E, n], logp [E, n], values [E, 1|n]."""
        o = torch.as_tensor(obs, device=self.device)
        s = torch.as_tensor(state, device=self.device)
        dist = Categorical(logits=self.actor(o))
        a = dist.probs.argmax(-1) if greedy else dist.sample()   # each ant samples independently
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), self._values(o, s).cpu().numpy()

    def update(self, roll: dict) -> dict:
        """roll: obs [T,E,n,d], state [T,E,S], actions/logp [T,E,n], team [T,E], done [T,E],
        values [T,E,k], last_values [E,k]  (k = 1 for MAPPO, n for IPPO or critic_per_ant).
        Without per-ant "rewards" in the roll, the team credit is used."""
        c = self.cfg
        T, E = roll["team"].shape
        values = roll["values"]
        k = values.shape[-1]
        if c.credit == "team" or "rewards" not in roll:
            step_r = np.repeat(roll["team"][:, :, None] / self.n, k, axis=-1)          # [T,E,k]
        else:
            step_r = 0.5 * roll["rewards"] + 0.5 * (roll["team"][:, :, None] / self.n)  # [T,E,n]
        reps = step_r.shape[-1] // k                     # 1-output critic + per-ant returns → repeat
        v_ag = np.repeat(values, reps, axis=-1)
        last_v_ag = np.repeat(roll["last_values"], reps, axis=-1)
        adv = np.zeros_like(v_ag)
        gae = np.zeros_like(last_v_ag)
        for t in reversed(range(T)):
            next_v = last_v_ag if t == T - 1 else v_ag[t + 1]
            not_done = (1.0 - roll["done"][t])[:, None]
            delta = step_r[t] + c.gamma * next_v * not_done - v_ag[t]
            gae = delta + c.gamma * c.gae_lambda * not_done * gae
            adv[t] = gae
        ret_ag = adv + v_ag
        returns = ret_ag.mean(axis=-1, keepdims=True) if k == 1 else ret_ag
        flat = lambda x: torch.as_tensor(x.reshape(T * E, *x.shape[2:]), device=self.device)
        obs, state, actions, old_logp = flat(roll["obs"]), flat(roll["state"]), flat(roll["actions"]), flat(roll["logp"])
        adv_t, ret_t = flat(adv.astype(np.float32)), flat(returns.astype(np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        n_samples, mb = T * E, max(1, T * E // c.minibatches)
        for _ in range(c.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, mb):
                idx = perm[start:start + mb]
                dist = Categorical(logits=self.actor(obs[idx]))
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv_t[idx] if adv_t.shape[-1] == self.n else adv_t[idx].expand_as(ratio)
                pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a).mean()
                v_loss = (self._values(obs[idx], state[idx]) - ret_t[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + c.value_coef * v_loss - c.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, c.grad_clip)
                self.opt.step()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
        k = c.epochs * ((n_samples + mb - 1) // mb)
        return {key: v / k for key, v in stats.items()}

    def state_dict(self) -> dict:
        return {"cfg": asdict(self.cfg), "actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
