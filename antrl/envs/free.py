"""Free world: continuous 2-D space, noisy headings, force-summed transport.

A piece moves by the vector sum of its holders' pulls divided by its weight
(capped at walking speed): two ants pulling the same way carry it at full speed,
a lone ant drags it at half speed, and opposite pulls cancel — the coordination
problem real ants solve through the load itself.

Each ant senses the nearest few ants, crumbs, pieces and big food within
``sense_radius`` (relative offsets), plus its own state and the way home.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import MOVES, ForageConfig, ForageEnv


@dataclass
class FreeConfig(ForageConfig):
    size: float = 10.0
    nest_radius: float = 1.0
    reach: float = 0.6
    speed: float = 0.5
    heading_noise: float = 0.25   # radians, uniform ±
    sense_radius: float = 3.0
    k_ants: int = 3
    k_crumbs: int = 3
    k_pieces: int = 2
    k_big: int = 1


class FreeForage(ForageEnv):
    def __init__(self, config: FreeConfig | None = None, seed: int | None = None):
        self.fcfg = config or FreeConfig()
        self.L = self.fcfg.size
        super().__init__(self.fcfg, seed)

    # geometry
    def _nest_position(self):
        return np.array([self.L / 2, self.L / 2])

    def _min_spawn_dist(self):
        return 2.5, 3.5

    def _sample_position(self, min_nest_dist, taken):
        for _ in range(10_000):
            p = self.rng.uniform(0.5, self.L - 0.5, size=2)
            if np.linalg.norm(p - self.nest) >= min_nest_dist and all(np.linalg.norm(p - q) >= 1.0 for q in taken):
                return p
        raise RuntimeError("arena too small to place food")

    def _near(self, a, b):
        return bool(np.linalg.norm(a - b) < self.fcfg.reach)

    def _at_nest(self, p):
        return bool(np.linalg.norm(p - self.nest) < self.fcfg.nest_radius)

    def _heading(self, action):
        dx, dy = MOVES[action]
        th = np.arctan2(dy, dx) + self.rng.uniform(-self.fcfg.heading_noise, self.fcfg.heading_noise)
        return np.array([np.cos(th), np.sin(th)])

    def _move_ant(self, p, action):
        return np.clip(p + self.fcfg.speed * self._heading(action), 0.0, self.L)

    def _move_piece(self, p, pulls):
        force = np.sum([self._heading(a) for a in pulls], axis=0) / self.cfg.piece_weight
        norm = np.linalg.norm(force)
        if norm > 1.0:
            force /= norm
        if norm < 0.1:                       # (near-)cancelling pulls: the load does not budge
            return p.copy()
        return np.clip(p + self.fcfg.speed * force, 0.0, self.L)

    def _dist_to_nest(self, p):
        return float(np.linalg.norm(p - self.nest) / self.fcfg.speed)

    def _spawn_piece_at(self, big_pos):
        """Cut pieces drop just outside cutting reach, on the nest side of the big item."""
        home = self.nest - big_pos
        home = home / (np.linalg.norm(home) + 1e-9)
        jitter = self.rng.uniform(-0.15, 0.15, size=2)
        return np.clip(big_pos + 1.4 * self.fcfg.reach * home + jitter, 0.0, self.L)

    # observations
    def _slots(self, origin, points, extras, k):
        """k nearest points within sense radius → [present, dx/R, dy/R, *extra] per slot."""
        R = self.fcfg.sense_radius
        width = 3 + (extras.shape[1] if extras is not None else 0)
        out = np.zeros((k, width), dtype=np.float32)
        if len(points):
            d = points - origin
            dist = np.linalg.norm(d, axis=1)
            idx = [j for j in np.argsort(dist) if dist[j] <= R][:k]
            for s, j in enumerate(idx):
                out[s, 0] = 1.0
                out[s, 1:3] = d[j] / R
                if extras is not None:
                    out[s, 3:] = extras[j]
        return out.ravel()

    def _observe(self, i):
        c, f = self.cfg, self.fcfg
        me = self.ant_pos[i]
        others = [j for j in range(self.n_agents) if j != i]
        ant_pts = self.ant_pos[others]
        ant_ext = np.array([[float(self.carrying[j] or self.holding[j] >= 0)] for j in others]).reshape(-1, 1)
        ants = self._slots(me, ant_pts, ant_ext, f.k_ants)
        if not c.observe_teammates:
            ants[:] = 0.0
        crumbs = self._slots(me, self.crumb_pos[self.crumb_alive], None, f.k_crumbs)
        live = np.flatnonzero(self.piece_state == 1)
        piece_ext = np.array([[len(self.piece_holders[p]) / c.piece_weight] for p in live]).reshape(-1, 1)
        pieces = self._slots(me, self.piece_pos[live], piece_ext, f.k_pieces)
        bigs = np.flatnonzero(self.big_left > 0)
        big = self._slots(me, self.big_pos[bigs], (self.big_left[bigs] / c.pieces_per_big)[:, None], f.k_big)
        return np.concatenate([self._self_features(i), ants, crumbs, pieces, big])

    def _self_features(self, i):
        holding = self.holding[i] >= 0
        partners = len(self.piece_holders[self.holding[i]]) / self.cfg.piece_weight if holding else 0.0
        home = self.nest - self.ant_pos[i]
        own = np.array([float(self.carrying[i]), float(holding), partners, home[0] / self.L, home[1] / self.L,
                        np.linalg.norm(home) / self.L, self.t / self.cfg.max_steps], dtype=np.float32)
        return np.concatenate([own, self._tug_features(i)])

    def _state(self):
        c, L = self.cfg, self.L
        ants = np.concatenate([self.ant_pos / L, self.carrying[:, None], (self.holding >= 0)[:, None]], axis=1)
        crumbs = np.concatenate([self.crumb_pos / L, self.crumb_alive[:, None]], axis=1)
        big = np.concatenate([self.big_pos / L, (self.big_left / c.pieces_per_big)[:, None]], axis=1)
        holders = np.array([len(h) / c.piece_weight for h in self.piece_holders])[:, None]
        pieces = np.concatenate([self.piece_pos / L, (self.piece_state == 1)[:, None], holders], axis=1)
        return np.concatenate([ants.ravel(), crumbs.ravel(), big.ravel(), pieces.ravel(),
                               [self.t / c.max_steps]]).astype(np.float32)
