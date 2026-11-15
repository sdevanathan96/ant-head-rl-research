"""Algorithm-level checks: mixer properties, losses and end-to-end updates."""
import numpy as np
import torch

from antrl.algos import AgentNet, PPOConfig, PPOLearner, QMixer, ValueConfig, ValueLearner, VDNMixer
from antrl.envs import make_env
from antrl.train import evaluate


def test_qmix_is_monotonic_in_every_agent_q():
    torch.manual_seed(0)
    mixer = QMixer(n_agents=4, state_dim=10)
    qs = torch.randn(64, 4, requires_grad=True)
    mixer(qs, torch.randn(64, 10)).sum().backward()
    assert (qs.grad >= 0).all()


def test_vdn_is_the_sum():
    qs = torch.randn(8, 3)
    assert torch.allclose(VDNMixer()(qs, None), qs.sum(-1))


def test_agent_net_sharing():
    shared = AgentNet(5, 6, 3, hidden=16, share=True)
    separate = AgentNet(5, 6, 3, hidden=16, share=False)
    x = torch.randn(2, 3, 5)
    assert shared(x).shape == separate(x).shape == (2, 3, 6)
    assert len(separate.nets) == 3


def _fill(learner, env, n=400):
    obs, state, _ = env.reset(seed=0)
    for _ in range(n):
        a = learner.act(obs, eps=1.0)
        nobs, nstate, r, team, done, info = env.step(a)
        learner.store(obs, state, a, r, team, nobs, nstate, float(info["terminated"]))
        obs, state = (env.reset()[:2] if done else (nobs, nstate))


def test_value_learners_update_and_round_trip():
    env = make_env("grid", seed=0, n_ants=2, size=7)
    for mixer, reward, share in (("none", "own", False), ("none", "team", False), ("vdn", "team", True), ("qmix", "team", True)):
        cfg = ValueConfig(mixer=mixer, reward=reward, share=share, hidden=32, batch_size=32, learning_starts=32)
        learner = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, cfg)
        _fill(learner, env)
        stats = [learner.update() for _ in range(5)]
        assert all(np.isfinite(s["loss"]) for s in stats)
        clone = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, cfg, seed=1)
        clone.load_state_dict(learner.state_dict())
        obs = env.reset(seed=3)[0]
        assert np.array_equal(clone.act(obs), learner.act(obs))


def test_iql_own_reward_target_uses_each_ants_reward():
    """A transition where only ant 0 is rewarded: IQL-own pushes Q_0 up, not Q_1."""
    env = make_env("grid", seed=0, n_ants=2, size=7)
    cfg = ValueConfig(mixer="none", reward="own", share=False, hidden=16, batch_size=8, learning_starts=1, lr=1e-2)
    learner = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, cfg)
    obs, state, _ = env.reset(seed=0)
    for _ in range(8):
        learner.store(obs, state, np.array([0, 0]), np.array([1.0, 0.0]), 1.0, obs, state, 1.0)
    for _ in range(50):
        learner.update()
    with torch.no_grad():
        q = learner.q(torch.as_tensor(obs).unsqueeze(0))[0, :, 0]
    assert q[0] > 0.5 and abs(float(q[1])) < 0.3


def test_ppo_update_runs_for_both_critics():
    env = make_env("free", seed=0, n_ants=2)
    for central in (True, False):
        learner = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                             PPOConfig(centralized_critic=central, hidden=32, rollout_steps=16, num_envs=2))
        T, E = 16, 2
        obs = np.random.rand(T, E, env.n_agents, env.obs_dim).astype(np.float32)
        state = np.random.rand(T, E, env.state_dim).astype(np.float32)
        a, logp, v = learner.act(obs.reshape(T * E, env.n_agents, -1), state.reshape(T * E, -1))
        k = v.shape[-1]
        roll = {"obs": obs, "state": state, "actions": a.reshape(T, E, -1), "logp": logp.reshape(T, E, -1),
                "values": v.reshape(T, E, k), "team": np.random.rand(T, E).astype(np.float32),
                "done": np.zeros((T, E), np.float32), "last_values": v.reshape(T, E, k)[-1]}
        stats = learner.update(roll)
        assert all(np.isfinite(x) for x in stats.values())
        assert k == (1 if central else env.n_agents)


def test_evaluate_reports_shares():
    env = make_env("grid", seed=0, n_ants=3, size=7)
    learner = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, ValueConfig(hidden=16))
    out = evaluate("grid", {"n_ants": 3, "size": 7}, learner, episodes=2)
    assert len(out["per_ant_share"]) == 3 and 0.0 <= out["gini"] <= 1.0


def test_ppo_samples_each_ant_independently_even_while_holding():
    """No hand-coded holder logic: a lone holder can still sample STAY (which lets go)."""
    env = make_env("grid", seed=0, n_ants=2, size=7)
    env.piece_state[0] = 1
    env.piece_pos[0] = np.array([2.0, 2.0])
    env.ant_pos[:] = np.array([2.0, 2.0])
    obs, state, *_ = env.step([5, 0])                # ant 0 grabs, ant 1 stays off the piece
    assert env.holding[0] == 0 and env.holding[1] == -1
    learner = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, PPOConfig(hidden=16))
    a, _, _ = learner.act(np.repeat(obs[None], 512, 0), np.repeat(state[None], 512, 0))
    assert (a[:, 0] == 0).any()                      # STAY is never masked out


def test_agent_id_switch():
    obs = torch.zeros(5, 3, 8)
    with_id, without = AgentNet(8, 4, 3, hidden=16), AgentNet(8, 4, 3, hidden=16, agent_id=False)
    assert not torch.allclose(with_id(obs)[:, 0], with_id(obs)[:, 1])   # ids break symmetry
    assert torch.allclose(without(obs)[:, 0], without(obs)[:, 1])       # identical obs -> identical output


def test_value_replay_treats_timeouts_as_terminal():
    """Same rule as PPO: a timeout transition is stored with terminal = 1."""
    from antrl.train import train_value
    env = make_env("grid", seed=0, max_steps=5, n_ants=2, size=7)
    learner = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                           ValueConfig(hidden=16, learning_starts=10**9))
    seen = []
    learner.store = lambda *t: seen.append(t[-1])
    train_value(env, learner, steps=10, on_eval=lambda *a: None, eval_every=10**9)
    assert seen == [0, 0, 0, 0, 1] * 2


def test_ppo_rollout_treats_timeouts_as_terminal():
    """t / max_steps is observed, so a timeout must end the GAE trace (done = 1)."""
    from antrl.train import train_ppo
    env_kw = {"max_steps": 5, "n_ants": 2, "size": 7}
    env = make_env("grid", seed=0, **env_kw)
    learner = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                         PPOConfig(hidden=16, rollout_steps=10, num_envs=1))
    seen = {}

    def capture(roll):
        seen["done"] = roll["done"][:, 0].tolist()
        return {}

    learner.update = capture
    train_ppo("grid", env_kw, learner, steps=10, seed=0, on_eval=lambda *a: None, eval_every=10**9)
    assert seen["done"] == [0, 0, 0, 0, 1] * 2


def _credit_roll(learner, env, T=8, E=2, own=None, seed=0):
    rng = np.random.default_rng(seed)
    n = env.n_agents
    obs = rng.random((T, E, n, env.obs_dim)).astype(np.float32)
    state = rng.random((T, E, env.state_dim)).astype(np.float32)
    a, logp, v = learner.act(obs.reshape(T * E, n, -1), state.reshape(T * E, -1))
    k = v.shape[-1]
    own = rng.random((T, E, n)).astype(np.float32) if own is None else own
    return {"obs": obs, "state": state, "actions": a.reshape(T, E, n), "logp": logp.reshape(T, E, n),
            "values": v.reshape(T, E, k), "rewards": own, "team": own.sum(-1),
            "done": np.ones((T, E), np.float32), "last_values": v.reshape(T, E, k)[-1]}


def test_team_credit_ignores_how_the_team_reward_is_split():
    """credit=team: only the team total matters; credit=blend: the per-ant split matters too."""
    env = make_env("grid", seed=0, n_ants=2, size=7)
    T, E = 8, 2
    even = np.full((T, E, 2), 0.5, np.float32)
    lopsided = np.stack([np.ones((T, E)), np.zeros((T, E))], -1).astype(np.float32)
    for credit, should_match in (("team", True), ("blend", False)):
        params = []
        for own in (even, lopsided):
            learner = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                                 PPOConfig(hidden=16, credit=credit), seed=0)
            roll = _credit_roll(learner, env, T, E, own=own)
            torch.manual_seed(1)
            learner.update(roll)
            params.append(torch.cat([p.flatten() for p in learner.params]))
        assert torch.equal(params[0], params[1]) == should_match


def test_per_ant_critic_heads_fit_each_ants_own_return():
    env = make_env("grid", seed=0, n_ants=2, size=7)
    learner = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                         PPOConfig(hidden=32, critic_per_ant=True, lr=5e-3), seed=0)
    T, E = 8, 2
    own = np.stack([np.ones((T, E)), np.zeros((T, E))], -1).astype(np.float32)   # only ant 0 earns
    for _ in range(40):
        learner.update(_credit_roll(learner, env, T, E, own=own))
    roll = _credit_roll(learner, env, T, E, own=own)
    v = roll["values"]
    assert v.shape[-1] == 2
    # episodes end every step, so the targets are 0.5·own + 0.5·team/n = 0.75 for ant 0, 0.25 for ant 1
    assert abs(v[..., 0].mean() - 0.75) < 0.1 and abs(v[..., 1].mean() - 0.25) < 0.1
