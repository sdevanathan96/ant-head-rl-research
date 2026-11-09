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


def test_honest_observations_without_hardcoded_targets():
    """Observations must use path-integrated home vector and have no _target_pos oracle."""
    env = grid()
    assert not hasattr(env, "_target_pos")
    place(env, 0, [2, 3])
    sf = env._self_features(0)
    expected_home = (env.nest - np.array([2.0, 3.0])) / env.size
    assert np.allclose(sf[3:5], expected_home)


def test_grab_release_does_not_farm_shaping_reward():
    """Grabbing a piece awards shaping at most once per ant per piece; repeated WORK/STAY yields 0 extra."""
    env = grid(shaping=0.1)
    env.piece_state[0] = 1
    env.piece_pos[0] = np.array([2.0, 2.0])
    place(env, 0, [2.0, 2.0])
    rewards = []
    for t in range(10):
        act = WORK if t % 2 == 0 else STAY
        _, _, own, _, _, _ = env.step([act, STAY])
        rewards.append(own[0])
    assert rewards[0] == pytest.approx(0.1)
    assert all(r == pytest.approx(0.0) for r in rewards[1:])


def test_monotone_transport_shaping_rewards_only_new_nest_progress():
    """Moving a held piece closer to the nest gives positive shaping once; oscillating back and forth gives 0."""
    env = grid(shaping=0.1)
    start = env.nest + [-2, 0]
    _two_holders_on_a_piece(env, start)
    env.piece_best_dist[0] = env._dist_to_nest(start)
    _, _, own_closer, _, _, _ = env.step([EAST, EAST])
    assert (own_closer > 0.0).all()
    _, _, own_away, _, _, _ = env.step([WEST, WEST])
    assert np.allclose(own_away, 0.0)
    _, _, own_repeat, _, _, _ = env.step([EAST, EAST])
    assert np.allclose(own_repeat, 0.0)


def test_hold_on_stay_keeps_the_piece_and_drop_lets_go():
    """With hold_on_stay, STAY no longer drops a held piece; the extra DROP action does."""
    from antrl.envs.core import DROP
    env = grid(hold_on_stay=True)
    assert env.n_actions == 7 and grid().n_actions == 6
    _two_holders_on_a_piece(env, [2, 2])
    env.step([STAY, STAY])
    assert (env.holding[:2] == 0).all()
    env.step([DROP, STAY])
    assert env.holding[0] == -1 and env.holding[1] == 0


def test_feel_tug_senses_only_partner_pulls_through_the_load():
    """feel_tug adds 2 features: the summed pull directions of the *other* holders last step."""
    env = grid(feel_tug=True)
    assert env.obs_dim == grid().obs_dim + 2
    _two_holders_on_a_piece(env, [2, 2])
    obs, *_ = env.step([EAST, WORK])                 # ant 0 pulls east alone; ant 1 keeps holding
    assert np.allclose(obs[1][-2:], [1.0, 0.0])      # ant 1 feels its partner's pull
    assert np.allclose(obs[0][-2:], [0.0, 0.0])      # ant 0 does not feel its own pull
    env.step([STAY, STAY])                           # ant 0 lets go
    assert np.allclose(env._tug_features(1), [0.0, 0.0])
