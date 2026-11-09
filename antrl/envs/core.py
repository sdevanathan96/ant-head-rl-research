"""Shared mechanics of the ant foraging worlds.

Both worlds (grid and free) run the same four jobs:

    1. find food            crumbs and big food are scattered away from the nest
    2. bring it home        an ant carrying a crumb delivers it by reaching the nest
    3. cut big food         WORK at a big food item adds cut progress; every
                            ``cut_work`` units release one transportable piece
    4. carry together       a piece is heavy: it only moves when its holders pull,
                            and ``piece_weight`` holders are needed for full speed

Subclasses supply the geometry (positions, movement, proximity, observations).

Actions (same in both worlds)::

    0 STAY   stand still; a holder that stays lets go of its piece
             (with ``hold_on_stay`` a holder that stays keeps holding)
    1 NORTH  2 SOUTH  3 EAST  4 WEST   move (or, when holding a piece, pull it)
    5 WORK   pick up a crumb / cut big food / grab a piece / keep holding
    6 DROP   only with ``hold_on_stay``: let go of a held piece

Rewards: every step returns one reward per ant (credited to the ant that earned
it) and a team reward equal to their sum, so independent learners can train on
their own reward and cooperative learners on the team reward.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STAY, NORTH, SOUTH, EAST, WEST, WORK = range(6)
DROP = 6                                   # extra action, only when ``hold_on_stay`` is on
N_ACTIONS = 6
ACTION_NAMES = ("stay", "north", "south", "east", "west", "work")
MOVES = {NORTH: (0.0, 1.0), SOUTH: (0.0, -1.0), EAST: (1.0, 0.0), WEST: (-1.0, 0.0)}


@dataclass
class ForageConfig:
    n_ants: int = 4
    n_crumbs: int = 4
    n_big: int = 1
    pieces_per_big: int = 4
    cut_work: int = 2              # units of WORK needed to cut one piece free
    piece_weight: int = 2          # holders needed to move a piece at full speed
    crumb_value: float = 1.0
    piece_value: float = 2.0
    max_steps: int = 100
    observe_teammates: bool = True  # False = "blind to teammates" (stage-1 baseline)
    shaping: float = 0.0           # bonus for pickups, cuts and grabs (0 = delivery reward only)
    step_cost: float = 0.0         # per-ant cost per step
    hold_on_stay: bool = False     # STAY keeps hold of a piece; a 7th action, DROP, lets go
    feel_tug: bool = False         # holders sense their partners' last pulls through the load (+2 obs)


@dataclass
class EpisodeStats:
    """Running totals for one episode; ``summary()`` is what evaluation reports."""
    n_ants: int
    team_value: float = 0.0
    crumbs: int = 0
    pieces: int = 0
    cut_work: int = 0
    grabs: int = 0
    stalled_pulls: int = 0         # holder-steps spent pulling a piece that did not move
    moving_pulls: int = 0          # holder-steps spent pulling a piece that moved
    per_ant_value: np.ndarray = field(default=None)
    per_ant_cut: np.ndarray = field(default=None)

    def __post_init__(self):
        self.per_ant_value = np.zeros(self.n_ants)
        self.per_ant_cut = np.zeros(self.n_ants, dtype=int)

    def summary(self) -> dict:
        share = self.per_ant_value / self.team_value if self.team_value > 0 else np.zeros(self.n_ants)
        return {
            "team_value": self.team_value,
            "crumbs": self.crumbs,
            "pieces": self.pieces,
            "cut_work": self.cut_work,
            "grabs": self.grabs,
            "stalled_pulls": self.stalled_pulls,
            "moving_pulls": self.moving_pulls,
            "per_ant_value": self.per_ant_value.tolist(),
            "per_ant_cut": self.per_ant_cut.tolist(),
            "per_ant_share": share.tolist(),
            "gini": gini(self.per_ant_value),
        }


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative contributions: 0 = equal shares, → 1 = one ant does it all."""
    x = np.asarray(x, dtype=float)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


class ForageEnv:
    """Base class: entities, the shared step logic, and bookkeeping."""

    n_actions = N_ACTIONS

    def __init__(self, config: ForageConfig | None = None, seed: int | None = None):
        self.cfg = config or ForageConfig()
        self.n_agents = self.cfg.n_ants
        self.n_actions = N_ACTIONS + (1 if self.cfg.hold_on_stay else 0)
        self.rng = np.random.default_rng(seed)
        self.reset(seed)
        self.obs_dim = self._observe_all().shape[1]
        self.state_dim = self._state().shape[0]

    # ── geometry hooks (implemented by GridForage / FreeForage) ──────────────
    def _nest_position(self) -> np.ndarray: raise NotImplementedError
    def _sample_position(self, min_nest_dist: float, taken: list) -> np.ndarray: raise NotImplementedError
    def _near(self, a: np.ndarray, b: np.ndarray) -> bool: raise NotImplementedError
    def _at_nest(self, p: np.ndarray) -> bool: raise NotImplementedError
    def _move_ant(self, p: np.ndarray, action: int) -> np.ndarray: raise NotImplementedError
    def _move_piece(self, p: np.ndarray, pulls: list[int]) -> np.ndarray: raise NotImplementedError
    def _spawn_piece_at(self, big_pos: np.ndarray) -> np.ndarray: raise NotImplementedError
    def _observe(self, i: int) -> np.ndarray: raise NotImplementedError
    def _state(self) -> np.ndarray: raise NotImplementedError
    def _min_spawn_dist(self) -> tuple[float, float]: raise NotImplementedError  # (crumbs, big food)

    def _dist_to_nest(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(p - self.nest))

    def _tug_features(self, i: int) -> np.ndarray:
        """Sum of the other holders' pull directions on ant i's piece last step (local load sensing)."""
        return self.tug[i].astype(np.float32) if self.cfg.feel_tug else np.zeros(0, dtype=np.float32)

    # ── episode ──────────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.t = 0
        self.nest = self._nest_position()
        crumb_d, big_d = self._min_spawn_dist()
        self.ant_pos = np.repeat(self.nest[None, :], c.n_ants, axis=0).astype(float)
        self.carrying = np.zeros(c.n_ants, dtype=bool)          # carrying a crumb
        self.holding = np.full(c.n_ants, -1, dtype=int)         # index of held piece, -1 = none
        taken: list[np.ndarray] = []
        for d, n in ((crumb_d, c.n_crumbs), (big_d, c.n_big)):
            for _ in range(n):
                taken.append(self._sample_position(d, taken))
        self.crumb_pos = np.array(taken[:c.n_crumbs]).reshape(-1, 2)
        self.crumb_alive = np.ones(c.n_crumbs, dtype=bool)
        self.big_pos = np.array(taken[c.n_crumbs:]).reshape(-1, 2)
        self.big_left = np.full(c.n_big, c.pieces_per_big, dtype=int)
        self.big_progress = np.zeros(c.n_big, dtype=int)
        n_pieces = c.n_big * c.pieces_per_big
        self.piece_pos = np.zeros((n_pieces, 2))
        self.piece_state = np.zeros(n_pieces, dtype=int)        # 0 uncut · 1 in the world · 2 delivered
        self.piece_holders: list[list[int]] = [[] for _ in range(n_pieces)]
        self.crumb_best_dist = np.full(c.n_ants, np.inf)
        self.piece_best_dist = np.full(n_pieces, np.inf)
        self.piece_grabbed = np.zeros((n_pieces, c.n_ants), dtype=bool)
        self.tug = np.zeros((c.n_ants, 2))
        self.stats = EpisodeStats(c.n_ants)
        return self._observe_all(), self._state(), {}

    def step(self, actions):
        c = self.cfg
        actions = np.asarray(actions, dtype=int)
        assert actions.shape == (self.n_agents,), actions.shape
        own = np.full(self.n_agents, -c.step_cost)
        order = self.rng.permutation(self.n_agents)             # fair tie-breaks for shared targets

        # 1. WORK: pick up a crumb, cut big food, grab a piece (holders keep holding)
        for i in order:
            if actions[i] != WORK or self.carrying[i] or self.holding[i] >= 0:
                continue
            k = self._crumb_here(self.ant_pos[i])
            if k >= 0:
                self.crumb_alive[k] = False
                self.carrying[i] = True
                self.crumb_best_dist[i] = self._dist_to_nest(self.ant_pos[i])
                own[i] += c.shaping
                continue
            b = self._big_here(self.ant_pos[i])
            if b >= 0:
                self.big_progress[b] += 1
                self.stats.cut_work += 1
                self.stats.per_ant_cut[i] += 1
                own[i] += 0.5 * c.shaping
                continue
            p = self._loose_piece_here(self.ant_pos[i])
            if p >= 0:
                had_holder = len(self.piece_holders[p]) > 0
                self.piece_holders[p].append(i)
                self.holding[i] = p
                self.ant_pos[i] = self.piece_pos[p].copy()
                self.stats.grabs += 1
                if not self.piece_grabbed[p, i]:
                    self.piece_grabbed[p, i] = True
                    own[i] += (2.0 * c.shaping) if had_holder else c.shaping

        # 2. STAY (or DROP, with hold_on_stay) releases a held piece
        release = DROP if c.hold_on_stay else STAY
        for i in range(self.n_agents):
            if actions[i] == release and self.holding[i] >= 0:
                self.piece_holders[self.holding[i]].remove(i)
                self.holding[i] = -1

        # 3. free ants (and crumb carriers) walk
        for i in range(self.n_agents):
            if self.holding[i] < 0 and actions[i] in MOVES:
                self.ant_pos[i] = self._move_ant(self.ant_pos[i], int(actions[i]))
                if c.shaping > 0 and self.carrying[i]:
                    d_now = self._dist_to_nest(self.ant_pos[i])
                    if d_now < self.crumb_best_dist[i]:
                        own[i] += 0.5 * c.shaping * (self.crumb_best_dist[i] - d_now)
                        self.crumb_best_dist[i] = d_now

        # 4. pieces move by their holders' pulls
        self.tug = np.zeros((self.n_agents, 2))
        for p in np.flatnonzero(self.piece_state == 1):
            holders = self.piece_holders[p]
            for h in holders:
                self.tug[h] = sum((np.array(MOVES[int(actions[j])]) for j in holders
                                   if j != h and actions[j] in MOVES), np.zeros(2))
            pulls = [int(actions[h]) for h in holders if actions[h] in MOVES]
            if not pulls:
                continue
            new = self._move_piece(self.piece_pos[p], pulls)
            moved = not np.allclose(new, self.piece_pos[p])
            if moved:
                self.stats.moving_pulls += len(pulls)
                if c.shaping > 0:
                    d_now = self._dist_to_nest(new)
                    if d_now < self.piece_best_dist[p]:
                        prog = self.piece_best_dist[p] - d_now
                        self.piece_best_dist[p] = d_now
                        for h in holders:
                            own[h] += 2.0 * c.shaping * prog
            else:
                self.stats.stalled_pulls += len(pulls)
            self.piece_pos[p] = new
            for h in holders:
                self.ant_pos[h] = new.copy()

        # 5. finished cuts release pieces
        for b in range(c.n_big):
            while self.big_progress[b] >= c.cut_work and self.big_left[b] > 0:
                self.big_progress[b] -= c.cut_work
                self.big_left[b] -= 1
                p = b * c.pieces_per_big + (c.pieces_per_big - 1 - self.big_left[b])
                self.piece_pos[p] = self._spawn_piece_at(self.big_pos[b])
                self.piece_state[p] = 1
                self.piece_best_dist[p] = self._dist_to_nest(self.piece_pos[p])

        # 6. deliveries at the nest
        for i in range(self.n_agents):
            if self.carrying[i] and self._at_nest(self.ant_pos[i]):
                self.carrying[i] = False
                self.crumb_best_dist[i] = np.inf
                own[i] += c.crumb_value
                self.stats.crumbs += 1
                self.stats.team_value += c.crumb_value
                self.stats.per_ant_value[i] += c.crumb_value
        for p in np.flatnonzero(self.piece_state == 1):
            holders = self.piece_holders[p]
            if holders and self._at_nest(self.piece_pos[p]):
                for h in holders:
                    own[h] += c.piece_value / len(holders)
                    self.stats.per_ant_value[h] += c.piece_value / len(holders)
                    self.holding[h] = -1
                self.piece_holders[p] = []
                self.piece_state[p] = 2
                self.stats.pieces += 1
                self.stats.team_value += c.piece_value

        self.t += 1
        terminated = self._all_delivered()
        truncated = self.t >= c.max_steps and not terminated
        info = {"terminated": terminated, "truncated": truncated}
        if terminated or truncated:
            info["episode"] = self.stats.summary()
        return self._observe_all(), self._state(), own, float(own.sum()), terminated or truncated, info

    # ── helpers ──────────────────────────────────────────────────────────────
    def _observe_all(self) -> np.ndarray:
        return np.stack([self._observe(i) for i in range(self.n_agents)]).astype(np.float32)

    def _crumb_here(self, pos) -> int:
        for k in np.flatnonzero(self.crumb_alive):
            if self._near(pos, self.crumb_pos[k]):
                return int(k)
        return -1

    def _big_here(self, pos) -> int:
        for b in np.flatnonzero(self.big_left > 0):
            if self.big_progress[b] < self.big_left[b] * self.cfg.cut_work and self._near(pos, self.big_pos[b]):
                return int(b)
        return -1

    def _loose_piece_here(self, pos) -> int:
        candidates = [int(p) for p in np.flatnonzero(self.piece_state == 1)
                      if len(self.piece_holders[p]) < self.cfg.piece_weight and self._near(pos, self.piece_pos[p])]
        if not candidates:
            return -1
        candidates.sort(key=lambda p: (-len(self.piece_holders[p]), p))
        return candidates[0]

    def _all_delivered(self) -> bool:
        return (not self.crumb_alive.any() and not self.carrying.any()
                and not (self.big_left > 0).any() and not (self.piece_state == 1).any())

    def max_team_value(self) -> float:
        c = self.cfg
        return c.n_crumbs * c.crumb_value + c.n_big * c.pieces_per_big * c.piece_value
