"""What the holders of a piece do, step by step.

Used by ``scripts/coordination.py`` (saved runs) and by ``python -m antrl.train --coord``
(at every evaluation, during training). Rollouts run on held-out seeds, either greedily or
with each ant sampling its own action as it does in training.

For every step in which a piece has 2 holders, it records what each holder chose (before the step
is applied) and reports, as shares of those co-holding steps:

    same_obs     the holders' observations are identical, so a shared policy can tell them
                 apart only by the one-hot ant id it appends
    agree        both pulled the same way           split    both pulled, different ways
    wait         one pulled, the other only held on hold     neither pulled, both held on
    release      at least one holder let go

and, over the pulls themselves:

    nestward     the pull points towards the nest (positive dot product with the home vector)
    diag_*       steps where the nest is diagonal from the piece (two nest-ward moves on the grid):
                 diag_steps (share of co-holding steps), diag_agree, straight_agree,
                 larger_axis (share of diagonal pulls along the larger home-vector component)
    x_share_by_ant  on diagonal steps, each ant id's share of pulls along x (east/west); very
                 different values per ant mean the id, not the situation, picks the axis

and, for steps in which a piece has a single holder: solo_steps_per_ep and the shares of that
holder's choices: solo_pull (on the grid this always stalls; in the free world it drags the piece
at half speed), solo_wait (holds on) and solo_release.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch

from .algos.ppo import PPOLearner
from .envs import make_env
from .envs.core import DROP, EAST, MOVES, STAY, WEST

# The scalar shares that `python -m antrl.train --coord` logs at every evaluation.
LOG_KEYS = ("co_holding_steps_per_ep", "agree", "split", "wait", "hold", "release", "nestward",
            "solo_steps_per_ep", "solo_pull", "solo_wait", "solo_release")


@contextmanager
def preserved_rng(learner):
    """Restore torch's random state and the learner's own generator (ε-greedy) on exit, so that
    sampling actions here leaves a training run that calls this unchanged."""
    state = learner.rng.bit_generator.state if hasattr(learner, "rng") else None
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        if state is not None:
            learner.rng.bit_generator.state = state


def analyse_policy(world: str, env_kw: dict, learner, episodes: int = 50, seed: int = 10_000,
                   sample: bool = False, eps: float | None = None) -> dict:
    """Roll out ``learner`` on ``episodes`` held-out seeds and summarise what the holders do.

    sample=False acts greedily. sample=True lets each ant sample as in training: PPO samples from
    π(a | o_i); value-based learners act ε-greedily with ``eps`` (default: the final ε).
    The random state of torch and of the learner is restored afterwards, so calling this in the
    middle of training leaves the training run unchanged.
    """
    if isinstance(learner, PPOLearner):
        def act(obs, state):
            return learner.act(obs[None], state[None], greedy=not sample)[0][0]
    else:
        e = (learner.cfg.eps_end if eps is None else eps) if sample else 0.0

        def act(obs, state):
            return learner.act(obs, eps=e)
    with preserved_rng(learner):
        return _analyse(world, env_kw, act, episodes, seed)


def _analyse(world: str, env_kw: dict, act, episodes: int, seed: int) -> dict:
    env = make_env(world, seed=seed, **env_kw)
    release_action = DROP if env.cfg.hold_on_stay else STAY
    c = defaultdict(float)
    x_pulls, all_pulls = defaultdict(float), defaultdict(float)
    for k in range(episodes):
        obs, state, _ = env.reset(seed=seed + k)
        done = False
        while not done:
            a = act(obs, state)
            for p in np.flatnonzero(env.piece_state == 1):
                holders = list(env.piece_holders[p])
                if len(holders) == 1:
                    x = int(a[holders[0]])
                    c["solo_steps"] += 1
                    c["solo_pull" if x in MOVES else "solo_release" if x == release_action else "solo_wait"] += 1
                if len(holders) < 2:
                    continue
                c["steps"] += 1
                c["same_obs"] += float(np.array_equal(obs[holders[0]], obs[holders[1]]))
                acts = [int(a[h]) for h in holders]
                pulls = [(h, x) for h, x in zip(holders, acts) if x in MOVES]
                if any(x == release_action for x in acts):
                    c["release"] += 1
                elif len(pulls) == 2:
                    c["agree" if pulls[0][1] == pulls[1][1] else "split"] += 1
                else:
                    c["wait" if len(pulls) == 1 else "hold"] += 1
                home = env.nest - env.piece_pos[p]
                diagonal = world == "grid" and abs(home[0]) > 0.5 and abs(home[1]) > 0.5
                if len(pulls) == 2 and world == "grid" and not any(x == release_action for x in acts):
                    key = "diag" if diagonal else "straight"
                    c[f"{key}_pairs"] += 1
                    c[f"{key}_agree"] += float(pulls[0][1] == pulls[1][1])
                for h, x in pulls:
                    c["pulls"] += 1
                    c["nestward"] += float(np.dot(MOVES[x], home) > 0)
                    if diagonal:
                        along_x = x in (EAST, WEST)
                        x_pulls[h] += float(along_x)
                        all_pulls[h] += 1
                        if abs(home[0]) != abs(home[1]):
                            c["diag_pulls_untied"] += 1
                            c["larger_axis"] += float(along_x == (abs(home[0]) > abs(home[1])))
                if diagonal:
                    c["diag_steps"] += 1
            obs, state, _, _, done, _ = env.step(a)
    s = max(c["steps"], 1)
    out = {"co_holding_steps_per_ep": c["steps"] / episodes}
    for key in ("same_obs", "agree", "split", "wait", "hold", "release", "diag_steps"):
        out[key] = c[key] / s
    out["nestward"] = c["nestward"] / max(c["pulls"], 1)
    out["diag_agree"] = c["diag_agree"] / c["diag_pairs"] if c["diag_pairs"] else float("nan")
    out["straight_agree"] = c["straight_agree"] / c["straight_pairs"] if c["straight_pairs"] else float("nan")
    out["larger_axis"] = c["larger_axis"] / c["diag_pulls_untied"] if c["diag_pulls_untied"] else float("nan")
    out["solo_steps_per_ep"] = c["solo_steps"] / episodes
    for key in ("solo_pull", "solo_wait", "solo_release"):
        out[key] = c[key] / max(c["solo_steps"], 1)
    out["x_share_by_ant"] = [round(x_pulls[i] / all_pulls[i], 2) if all_pulls[i] else None
                             for i in range(env.n_agents)]
    return out
