"""Mechanics of the four jobs in both worlds."""
import numpy as np
import pytest

from antrl.envs import GridConfig, GridForage, FreeConfig, FreeForage, gini
from antrl.envs.core import EAST, NORTH, STAY, WEST, WORK


def grid(**kw):
    cfg = GridConfig(**{"n_ants": 2, "n_crumbs": 1, "n_big": 1, "size": 10, **kw})
    env = GridForage(cfg, seed=0)
    env.reset(seed=0)
    return env


def place(env, ant, pos):
    env.ant_pos[ant] = np.array(pos, dtype=float)


def test_shapes_and_determinism():
    a, b = grid(), grid()
    oa, sa, _ = a.reset(seed=7)
    ob, sb, _ = b.reset(seed=7)
    assert oa.shape == (2, a.obs_dim) and sa.shape == (a.state_dim,)
    assert np.array_equal(oa, ob) and np.array_equal(sa, sb)


def test_food_never_overlaps():
    env = GridForage(GridConfig(n_crumbs=12, n_big=2, size=10), seed=1)
    for s in range(20):
        env.reset(seed=s)
        cells = [tuple(p) for p in np.concatenate([env.crumb_pos, env.big_pos])]
        assert len(cells) == len(set(cells))


def test_crumb_pickup_and_delivery_credits_the_carrier():
    env = grid()
    place(env, 0, env.crumb_pos[0])
    env.step([WORK, STAY])
    assert env.carrying[0]
    place(env, 0, env.nest + [1, 0])
    _, _, own, team, _, _ = env.step([WEST, STAY])
    assert own[0] == pytest.approx(1.0) and own[1] == 0.0 and team == pytest.approx(1.0)
    assert env.stats.crumbs == 1 and env.stats.per_ant_value[0] == 1.0


def test_cutting_releases_a_piece_after_cut_work():
    env = grid(cut_work=2)
    place(env, 0, env.big_pos[0])
    env.step([WORK, STAY])
    assert env.big_left[0] == 4 and (env.piece_state == 1).sum() == 0
    env.step([WORK, STAY])
    assert env.big_left[0] == 3 and (env.piece_state == 1).sum() == 1


@pytest.mark.parametrize("world", ["grid", "free"])
def test_pieces_can_be_carried_while_cutting_continues(world):
    """Cut pieces land beside the big item: grabbing must not wait for the whole item to be cut."""
    if world == "grid":
        env = grid(cut_work=1)
    else:
        env = FreeForage(FreeConfig(n_ants=2, n_crumbs=1, cut_work=1), seed=0)
        env.reset(seed=0)
    place(env, 0, env.big_pos[0])
    env.step([WORK, STAY])                         # first piece cut free
    p = int(np.flatnonzero(env.piece_state == 1)[0])
    assert env.big_left[0] == 3
    assert not env._near(env.piece_pos[p], env.big_pos[0])
    assert env.cfg.n_big == 1 and np.linalg.norm(env.piece_pos[p] - env.nest) < np.linalg.norm(env.big_pos[0] - env.nest)
    place(env, 1, env.piece_pos[p])
    env.step([WORK, WORK])                         # ant 0 keeps cutting, ant 1 grabs the piece
    assert env.holding[1] == p and env.big_left[0] == 2


def _two_holders_on_a_piece(env, cell):
    env.piece_state[0] = 1
    env.piece_pos[0] = np.array(cell, dtype=float)
    place(env, 0, cell)
    place(env, 1, cell)
    env.step([WORK, WORK])
    assert env.piece_holders[0] == [0, 1] or sorted(env.piece_holders[0]) == [0, 1]


def test_grid_piece_needs_two_ants_pulling_the_same_way():
    env = grid()
    start = np.array([2.0, 2.0])
    _two_holders_on_a_piece(env, start)
    env.step([EAST, WORK])                         # one puller: stuck
    assert np.array_equal(env.piece_pos[0], start)
    env.step([EAST, NORTH])                        # disagreeing: stuck
    assert np.array_equal(env.piece_pos[0], start)
    assert env.stats.stalled_pulls == 3
    env.step([EAST, EAST])                         # agreeing: moves, carriers move with it
    assert np.array_equal(env.piece_pos[0], start + [1, 0])
    assert np.array_equal(env.ant_pos[0], env.piece_pos[0]) and np.array_equal(env.ant_pos[1], env.piece_pos[0])


def test_piece_delivery_splits_value_and_stay_releases():
    env = grid()
    _two_holders_on_a_piece(env, env.nest + [-1, 0])
    _, _, own, team, _, _ = env.step([EAST, EAST])
    assert team == pytest.approx(2.0) and own == pytest.approx([1.0, 1.0])
    assert (env.holding == -1).all() and env.piece_state[0] == 2
    env2 = grid()
    _two_holders_on_a_piece(env2, [2, 2])
    env2.step([STAY, WORK])
    assert env2.holding[0] == -1 and env2.piece_holders[0] == [1]


def test_partial_view_and_blind_teammates():
    env = grid(view_radius=1)
    obs = env._observe(0)
    assert obs.shape == (6 * 9 + 6,)
    place(env, 1, env.ant_pos[0] + [1, 0])
    seen = env._observe(0)[:9].reshape(3, 3)       # ant channel
    assert seen.sum() == 1.0
    blind = grid(view_radius=1, observe_teammates=False)
    place(blind, 1, blind.ant_pos[0] + [1, 0])
    assert blind._observe(0)[:9].sum() == 0.0


def test_episode_ends_at_max_steps():
    env = grid(max_steps=5)
    done = False
    for _ in range(5):
        *_, done, info = env.step([STAY, STAY])
    assert done and info["truncated"] and "episode" in info


def test_free_world_pulls_add_as_vectors():
    env = FreeForage(FreeConfig(n_ants=2, heading_noise=0.0, speed=0.5), seed=0)
    env.reset(seed=0)
    env.piece_state[0] = 1
    p0 = np.array([3.0, 3.0])
    env.piece_pos[0] = p0
    place(env, 0, p0)
    place(env, 1, p0)
    env.step([WORK, WORK])
    env.step([EAST, WEST])                          # opposite pulls cancel
    assert np.allclose(env.piece_pos[0], p0)
    env.step([EAST, EAST])                          # aligned: full speed
    assert np.allclose(env.piece_pos[0], p0 + [0.5, 0.0])
    env.step([EAST, WORK])                          # one puller: half speed
    assert np.allclose(env.piece_pos[0], p0 + [0.75, 0.0])


def test_free_world_observation_is_local():
    env = FreeForage(FreeConfig(n_ants=2, sense_radius=1.0), seed=0)
    env.reset(seed=0)
    env.crumb_pos[:] = env.ant_pos[0] + [5.0, 0.0]  # every crumb out of range
    k = env.fcfg
    crumb_slots = env._observe(0)[7 + 4 * k.k_ants: 7 + 4 * k.k_ants + 3 * k.k_crumbs]
    assert crumb_slots.sum() == 0.0


def test_gini():
    assert gini(np.array([1, 1, 1, 1])) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 4])) == pytest.approx(0.75)
    assert gini(np.zeros(4)) == 0.0


def test_grab_release_shaping_does_not_farm_reward():
    """Alternating WORK (grab) and STAY (release) must not accumulate positive shaping reward."""
    env = grid(shaping=0.1)
    env.piece_state[0] = 1
    env.piece_pos[0] = np.array([2.0, 2.0])
    place(env, 0, [2.0, 2.0])
    total = 0.0
    for t in range(10):
        act = WORK if t % 2 == 0 else STAY
        _, _, own, _, _, _ = env.step([act, STAY])
        total += own[0]
    assert total <= 0.0


def test_transport_shaping_rewards_progress_toward_nest():
    """Pulling a held piece closer to the nest yields positive shaping; pulling away yields negative."""
    env = grid(shaping=0.1)
    start = env.nest + [-2, 0]
    _two_holders_on_a_piece(env, start)
    _, _, own_toward, team_toward, _, _ = env.step([EAST, EAST])
    assert (own_toward > 0.0).all() and team_toward > 0.0
    _, _, own_away, team_away, _, _ = env.step([WEST, WEST])
    assert (own_away < 0.0).all() and team_away < 0.0


def test_overcutting_is_prevented_on_final_piece():
    """Multiple ants cutting on the last remaining piece cannot exceed total cut_work."""
    env = grid(cut_work=2)
    env.big_left[0] = 1
    env.big_progress[0] = 1
    place(env, 0, env.big_pos[0])
    place(env, 1, env.big_pos[0])
    env.step([WORK, WORK])
    assert env.stats.cut_work == 1 and env.big_left[0] == 0

