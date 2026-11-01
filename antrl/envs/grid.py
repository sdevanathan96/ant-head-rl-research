"""Grid world: discrete cells, partial egocentric views, two-ant pieces.

A piece moves one cell only when at least ``piece_weight`` holders pull in the
SAME direction in the same step — disagreeing holders stall it. Each ant sees a
(2r+1)×(2r+1) window around itself plus its own state and the direction home
(ants path-integrate their way back to the nest).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import MOVES, ForageConfig, ForageEnv

# observation / state channels
CH_ANTS, CH_CRUMBS, CH_PIECES, CH_BIG, CH_NEST, CH_WALL = range(6)


@dataclass
class GridConfig(ForageConfig):
    size: int = 10
    view_radius: int = 2


class GridForage(ForageEnv):
    def __init__(self, config: GridConfig | None = None, seed: int | None = None):
        self.gcfg = config or GridConfig()
        self.size = self.gcfg.size
        self.r = self.gcfg.view_radius
        super().__init__(self.gcfg, seed)

    # geometry
    def _nest_position(self):
        c = self.size // 2
        return np.array([c, c], dtype=float)

    def _min_spawn_dist(self):
        return 3.0, 4.0

    def _sample_position(self, min_nest_dist, taken):
        used = {tuple(q) for q in taken}
        for _ in range(10_000):
            p = self.rng.integers(0, self.size, size=2).astype(float)
            if np.abs(p - self.nest).sum() >= min_nest_dist and tuple(p) not in used:
                return p
        raise RuntimeError("grid too small to place food")

    def _near(self, a, b):
        return bool(np.array_equal(a, b))

    def _dist(self, a, b) -> float:
        return float(np.abs(a - b).sum())

    def _at_nest(self, p):
        return bool(np.array_equal(p, self.nest))

    def _move_ant(self, p, action):
        d = np.array(MOVES[action])
        return np.clip(p + d, 0, self.size - 1)

    def _move_piece(self, p, pulls):
        counts = np.bincount(pulls, minlength=5)
        best = int(counts.argmax())
        if counts[best] >= self.cfg.piece_weight and (counts == counts[best]).sum() == 1:
            return self._move_ant(p, best)
        return p.copy()

    def _spawn_piece_at(self, big_pos):
        """Cut pieces drop on the neighbouring cell nearest the nest, so carrying can start
        while the big item is still being cut (WORK on the big item's own cell always cuts)."""
        options = [big_pos + d for d in (np.array(m) for m in MOVES.values())
                   if (0 <= big_pos + d).all() and (big_pos + d < self.size).all()]
        dist = [np.abs(o - self.nest).sum() for o in options]
        best = [o for o, d in zip(options, dist) if d == min(dist)]
        return best[self.rng.integers(len(best))].copy()

    # observations
    def _layers(self, viewer: int | None = None) -> np.ndarray:
        """Full-map channels; ``viewer`` (if given) is left out of the ant channel."""
        g = np.zeros((6, self.size, self.size), dtype=np.float32)
        for i, (x, y) in enumerate(self.ant_pos.astype(int)):
            if i != viewer:
                g[CH_ANTS, x, y] += 1.0
        for (x, y) in self.crumb_pos[self.crumb_alive].astype(int):
            g[CH_CRUMBS, x, y] = 1.0
        for p in np.flatnonzero(self.piece_state == 1):
            x, y = self.piece_pos[p].astype(int)
            g[CH_PIECES, x, y] = max(g[CH_PIECES, x, y], 0.5 + 0.5 * len(self.piece_holders[p]) / self.cfg.piece_weight)
        for b in np.flatnonzero(self.big_left > 0):
            x, y = self.big_pos[b].astype(int)
            g[CH_BIG, x, y] = self.big_left[b] / self.cfg.pieces_per_big
        nx, ny = self.nest.astype(int)
        g[CH_NEST, nx, ny] = 1.0
        return g

    def _observe(self, i):
        r, n = self.r, self.size
        g = self._layers(viewer=i)
        if not self.cfg.observe_teammates:
            g[CH_ANTS] = 0.0
        g[CH_WALL] = 0.0
        pad = np.zeros((6, n + 2 * r, n + 2 * r), dtype=np.float32)
        pad[CH_WALL] = 1.0
        pad[:, r:r + n, r:r + n] = g
        pad[CH_WALL, r:r + n, r:r + n] = 0.0
        x, y = self.ant_pos[i].astype(int)
        window = pad[:, x:x + 2 * r + 1, y:y + 2 * r + 1]
        return np.concatenate([window.ravel(), self._self_features(i)])

    def _self_features(self, i):
        holding = self.holding[i] >= 0
        partners = len(self.piece_holders[self.holding[i]]) / self.cfg.piece_weight if holding else 0.0
        target = self._target_pos(i)
        d = target - self.ant_pos[i]
        if abs(d[0]) >= abs(d[1]) and d[0] != 0:
            step_dir = np.array([np.sign(d[0]), 0.0], dtype=np.float32)
        elif d[1] != 0:
            step_dir = np.array([0.0, np.sign(d[1])], dtype=np.float32)
        else:
            step_dir = np.zeros(2, dtype=np.float32)
        return np.array([float(self.carrying[i]), float(holding), partners, step_dir[0], step_dir[1],
                         self.t / self.cfg.max_steps], dtype=np.float32)

    def _state(self):
        g = self._layers()[:CH_WALL].ravel()
        ants = np.concatenate([self.ant_pos / self.size, self.carrying[:, None],
                               (self.holding >= 0)[:, None]], axis=1).ravel()
        return np.concatenate([g, ants, [self.t / self.cfg.max_steps]]).astype(np.float32)

    def render_ascii(self) -> str:
        """Small text view for debugging: N nest, c crumb, B big food, p piece, a ant (A = loaded)."""
        rows = [["." for _ in range(self.size)] for _ in range(self.size)]
        put = lambda p, ch: rows[self.size - 1 - int(p[1])].__setitem__(int(p[0]), ch)
        put(self.nest, "N")
        for k in np.flatnonzero(self.crumb_alive):
            put(self.crumb_pos[k], "c")
        for b in np.flatnonzero(self.big_left > 0):
            put(self.big_pos[b], "B")
        for p in np.flatnonzero(self.piece_state == 1):
            put(self.piece_pos[p], "p")
        for i in range(self.n_agents):
            put(self.ant_pos[i], "A" if (self.carrying[i] or self.holding[i] >= 0) else "a")
        return "\n".join("".join(r) for r in rows)
