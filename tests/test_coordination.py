"""Holder coordination and sampled evaluation: measuring either must not change a training run."""
import csv

import torch

from antrl.algos import PPOConfig, PPOLearner, ValueConfig, ValueLearner
from antrl.coordination import LOG_KEYS, analyse_policy
from antrl.envs import make_env
from antrl.train import evaluate, main

TINY = ["--world", "grid", "--steps", "128", "--eval-every", "64", "--eval-episodes", "2",
        "--env", "n_ants=2", "size=7", "max_steps=20"]
LEARNER = {"ippo": ["rollout_steps=16", "num_envs=2", "hidden=16"],
           "iql": ["learning_starts=32", "batch_size=16", "hidden=16"]}


def _train(tmp_path, algo, coord):
    out = tmp_path / ("coord" if coord else "plain")
    main(TINY + ["--algo", algo, "--set", *LEARNER[algo], "--out", str(out)] + (["--coord"] if coord else []))
    run = out / f"grid-{algo}-s0"
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    return torch.load(run / "model.pt"), rows


def _params(sd):
    """Every network tensor in a learner's state_dict, keyed by its path."""
    return {f"{part}.{k}": t for part, d in sd.items() if isinstance(d, dict) and part != "cfg"
            for k, t in d.items()}


def test_coord_logging_leaves_training_unchanged(tmp_path):
    for algo in ("ippo", "iql"):
        plain_model, plain_rows = _train(tmp_path / algo, algo, coord=False)
        coord_model, coord_rows = _train(tmp_path / algo, algo, coord=True)
        assert _params(plain_model) and _params(plain_model).keys() == _params(coord_model).keys()
        for name, t in _params(plain_model).items():
            assert torch.equal(t, _params(coord_model)[name]), (algo, name)
        assert [r["team_value"] for r in plain_rows] == [r["team_value"] for r in coord_rows]
        for tag in ("g", "s"):
            for key in LOG_KEYS:
                assert f"{tag}_{key}" in coord_rows[-1]
            assert 0.0 <= float(coord_rows[-1][f"{tag}_agree"]) <= 1.0
        assert "g_agree" not in plain_rows[-1]


def test_value_learner_sampling_restores_its_random_state():
    env = make_env("grid", seed=0, n_ants=2, size=7)
    learner = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, ValueConfig(hidden=16))
    before = learner.rng.bit_generator.state
    analyse_policy("grid", {"n_ants": 2, "size": 7}, learner, episodes=2, sample=True, eps=0.5)
    assert learner.rng.bit_generator.state == before


def test_sampled_evaluation_restores_random_state_and_greedy_is_unchanged():
    env = make_env("grid", seed=0, n_ants=2, size=7)
    kw = {"n_ants": 2, "size": 7}
    ppo = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, PPOConfig(hidden=16), seed=0)
    value = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions, ValueConfig(hidden=16))
    for learner in (ppo, value):
        greedy = evaluate("grid", kw, learner, episodes=2)
        torch_state, own = torch.get_rng_state(), getattr(learner, "rng", None)
        own_state = own.bit_generator.state if own is not None else None
        evaluate("grid", kw, learner, episodes=2, sample=True)
        assert torch.equal(torch.get_rng_state(), torch_state)
        if own is not None:
            assert own.bit_generator.state == own_state
        assert evaluate("grid", kw, learner, episodes=2) == greedy
