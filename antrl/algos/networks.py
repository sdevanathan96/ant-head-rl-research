"""Networks shared by the learners."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class AgentNet(nn.Module):
    """Maps each ant's observation to an output (Q-values, logits or a value).

    ``share=True``  one network for every ant (parameter sharing); a one-hot ant id
                    is appended so ants can still specialise.
    ``share=False`` a separate network per ant (fully independent learners).
    Input ``[..., n_agents, obs_dim]`` → output ``[..., n_agents, out_dim]``.
    """

    def __init__(self, obs_dim: int, out_dim: int, n_agents: int, hidden: int = 128, share: bool = True):
        super().__init__()
        self.n, self.share, self.obs_dim = n_agents, share, obs_dim
        self_dim = 6 if obs_dim >= 60 else (7 if obs_dim >= 20 else obs_dim)
        self.self_dim = self_dim
        if share:
            self.net = mlp(obs_dim + n_agents, hidden, out_dim)
            self.self_net = mlp(self_dim + n_agents, max(16, hidden // 2), out_dim, depth=1)
        else:
            self.nets = nn.ModuleList([
                nn.ModuleDict({
                    "full": mlp(obs_dim, hidden, out_dim),
                    "self": mlp(self_dim, max(16, hidden // 2), out_dim, depth=1),
                })
                for _ in range(n_agents)
            ])
        self.register_buffer("ids", torch.eye(n_agents))

    def _extract_self(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_dim >= 60:
            return obs[..., -6:]
        if self.obs_dim >= 20:
            return obs[..., :7]
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        sf = self._extract_self(obs)
        if self.share:
            ids = self.ids.expand(*obs.shape[:-2], self.n, self.n)
            return 0.15 * self.net(torch.cat([obs, ids], dim=-1)) + self.self_net(torch.cat([sf, ids], dim=-1))
        return torch.stack([
            0.15 * m["full"](obs[..., i, :]) + m["self"](sf[..., i, :])
            for i, m in enumerate(self.nets)
        ], dim=-2)


class VDNMixer(nn.Module):
    """VDN (Sunehag et al. 2018): Q_tot = Σ_i Q_i."""

    def forward(self, qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return qs.sum(dim=-1)


class QMixer(nn.Module):
    """QMIX (Rashid et al. 2018): Q_tot = f(Q_1..Q_n; s) with ∂Q_tot/∂Q_i ≥ 0.

    Hypernetworks conditioned on the global state produce the mixing weights;
    taking their absolute value keeps the mix monotonic in every ant's Q, so each
    ant's greedy action is also the team's greedy action.
    """

    def __init__(self, n_agents: int, state_dim: int, embed: int = 32, hyper_hidden: int = 64):
        super().__init__()
        self.n, self.embed = n_agents, embed
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, n_agents * embed))
        self.hyper_b1 = nn.Linear(state_dim, embed)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
                                      nn.Linear(hyper_hidden, embed))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(b, self.n, self.embed)
        b1 = self.hyper_b1(state).view(b, 1, self.embed)
        hidden = F.elu(torch.bmm(qs.view(b, 1, self.n), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(b, self.embed, 1)
        b2 = self.hyper_b2(state).view(b, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(b)
