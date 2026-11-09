"""Value-based multi-agent learners: IQL, VDN and QMIX.

All three train per-ant Q-networks from a shared replay buffer of transitions.
They differ only in how the per-ant values are combined in the loss:

    IQL   each ant's own TD loss on its own reward (or, as an ablation, the team reward)
    VDN   one TD loss on Q_tot = Σ_i Q_i, trained on the team reward
    QMIX  one TD loss on Q_tot = mixer(Q_1..Q_n; s), trained on the team reward

VDN and QMIX are "train together, act alone": the mixer (and QMIX's global
state) is only used in training; acting uses each ant's own Q on its own view.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .networks import AgentNet, QMixer, VDNMixer


@dataclass
class ValueConfig:
    mixer: str = "none"          # none (IQL) | vdn | qmix
    reward: str = "own"          # IQL only: own | team   (VDN/QMIX always use the team reward)
    share: bool = True           # parameter sharing across ants
    hidden: int = 128
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 2_000  # env steps before the first update
    train_every: int = 2          # env steps per gradient step
    target_update: int = 500      # gradient steps between target-network syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 100_000      # env steps to anneal epsilon
    double_q: bool = True
    grad_clip: float = 10.0
    mixer_embed: int = 32
    agent_id: bool = True        # append a one-hot ant id to the shared Q-network


class ReplayBuffer:
    def __init__(self, capacity: int, n: int, obs_dim: int, state_dim: int):
        self.cap, self.i, self.full = capacity, 0, False
        self.obs = np.zeros((capacity, n, obs_dim), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.next_state = np.zeros_like(self.state)
        self.actions = np.zeros((capacity, n), np.int64)
        self.rewards = np.zeros((capacity, n), np.float32)
        self.team = np.zeros(capacity, np.float32)
        self.terminal = np.zeros(capacity, np.float32)

    def __len__(self):
        return self.cap if self.full else self.i

    def add(self, obs, state, actions, rewards, team, next_obs, next_state, terminal):
        k = self.i
        self.obs[k], self.state[k], self.actions[k] = obs, state, actions
        self.rewards[k], self.team[k] = rewards, team
        self.next_obs[k], self.next_state[k], self.terminal[k] = next_obs, next_state, terminal
        self.i = (k + 1) % self.cap
        self.full = self.full or self.i == 0

    def sample(self, batch: int, rng: np.random.Generator, device) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, len(self), size=batch)
        names = ("obs", "state", "actions", "rewards", "team", "next_obs", "next_state", "terminal")
        return {k: torch.as_tensor(getattr(self, k)[idx], device=device) for k in names}


class ValueLearner:
    def __init__(self, n_agents: int, obs_dim: int, state_dim: int, n_actions: int,
                 cfg: ValueConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = cfg = cfg or ValueConfig()
        self.n, self.n_actions, self.device = n_agents, n_actions, torch.device(device)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.q = AgentNet(obs_dim, n_actions, n_agents, cfg.hidden, cfg.share, cfg.agent_id).to(self.device)
        self.mixer = {"none": None, "vdn": VDNMixer(),
                      "qmix": QMixer(n_agents, state_dim, cfg.mixer_embed)}[cfg.mixer]
        if self.mixer is not None:
            self.mixer.to(self.device)
        self.q_target = copy.deepcopy(self.q)
        self.mixer_target = copy.deepcopy(self.mixer)
        params = list(self.q.parameters()) + (list(self.mixer.parameters()) if self.mixer is not None else [])
        self.params = params
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, n_agents, obs_dim, state_dim)
        self.updates = 0

    def epsilon(self, env_step: int) -> float:
        c = self.cfg
        frac = min(1.0, env_step / max(1, c.eps_decay))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float = 0.0) -> np.ndarray:
        q = self.q(torch.as_tensor(obs, device=self.device).unsqueeze(0))[0]
        greedy = q.argmax(-1).cpu().numpy()
        if eps <= 0.0:
            return greedy
        explore = self.rng.random(self.n) < eps                  # each ant explores independently
        return np.where(explore, self.rng.integers(0, self.n_actions, size=self.n), greedy)

    def store(self, *transition):
        self.buffer.add(*transition)

    def ready(self, env_step: int) -> bool:
        return (len(self.buffer) >= max(self.cfg.batch_size, self.cfg.learning_starts)
                and env_step % self.cfg.train_every == 0)

    def update(self) -> dict:
        c = self.cfg
        b = self.buffer.sample(c.batch_size, self.rng, self.device)
        q_all = self.q(b["obs"])                                            # [B, n, A]
        q_taken = q_all.gather(-1, b["actions"].unsqueeze(-1)).squeeze(-1)  # [B, n]
        with torch.no_grad():
            q_next_target = self.q_target(b["next_obs"])
            if c.double_q:
                a_star = self.q(b["next_obs"]).argmax(-1, keepdim=True)
                q_next = q_next_target.gather(-1, a_star).squeeze(-1)
            else:
                q_next = q_next_target.max(-1).values
            not_done = 1.0 - b["terminal"]
        if self.mixer is None:
            r = b["rewards"] if c.reward == "own" else b["team"].unsqueeze(-1).expand_as(q_taken)
            target = r + c.gamma * not_done.unsqueeze(-1) * q_next
            pred = q_taken
        else:
            pred = self.mixer(q_taken, b["state"])
            with torch.no_grad():
                target = b["team"] + c.gamma * not_done * self.mixer_target(q_next, b["next_state"])
        loss = F.smooth_l1_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, c.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % c.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            if self.mixer is not None:
                self.mixer_target.load_state_dict(self.mixer.state_dict())
        return {"loss": loss.item(), "q_mean": q_taken.mean().item()}

    def state_dict(self) -> dict:
        return {"cfg": asdict(self.cfg), "q": self.q.state_dict(),
                "mixer": self.mixer.state_dict() if self.mixer is not None else None}

    def load_state_dict(self, sd: dict):
        self.q.load_state_dict(sd["q"])
        self.q_target.load_state_dict(sd["q"])
        if self.mixer is not None and sd.get("mixer") is not None:
            self.mixer.load_state_dict(sd["mixer"])
            self.mixer_target.load_state_dict(sd["mixer"])
