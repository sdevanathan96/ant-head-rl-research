"""Train and evaluate the ant foraging learners.

    python -m antrl.train --world grid --algo qmix --steps 300000 --seed 0
    python -m antrl.train --world free --algo mappo --steps 500000 --env shaping=0.1

Presets (stage 1 = each ant alone, stage 2 = loosened policies):

    iql       stage 1 baseline: own Q-network, own reward, blind to teammates
    iql_team  ablation: independent learners on the shared team reward
    vdn       Q_tot = Σ Q_i, team reward, shared parameters, sense teammates
    qmix      monotonic mixer on the global state, team reward, shared parameters
    ippo      PPO with a decentralised critic V(o_i), team reward
    mappo     PPO with a centralised critic V(s), team reward

Every run writes runs/<world>-<algo>-s<seed>/{config.json, metrics.csv, model.pt}.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from .algos.ppo import PPOConfig, PPOLearner
from .algos.value_based import ValueConfig, ValueLearner
from .envs import make_env

PRESETS = {
    "iql":      ("value", {"mixer": "none", "reward": "own", "share": False}, {"observe_teammates": False}),
    "iql_team": ("value", {"mixer": "none", "reward": "team", "share": False}, {"observe_teammates": False}),
    "vdn":      ("value", {"mixer": "vdn", "share": True}, {"observe_teammates": True}),
    "qmix":     ("value", {"mixer": "qmix", "share": True}, {"observe_teammates": True}),
    "ippo":     ("ppo", {"centralized_critic": False, "share": True}, {"observe_teammates": True}),
    "mappo":    ("ppo", {"centralized_critic": True, "share": True}, {"observe_teammates": True}),
}


def _parse_overrides(pairs: list[str], dataclass_type) -> dict:
    """``["lr=1e-3", "share=false"]`` → typed kwargs for ``dataclass_type``."""
    types = {f.name: f.type for f in fields(dataclass_type)}
    out = {}
    for pair in pairs or []:
        key, val = pair.split("=", 1)
        if key not in types:
            raise SystemExit(f"unknown option {key!r} for {dataclass_type.__name__}")
        t = str(types[key])
        if "bool" in t:
            out[key] = val.lower() in ("1", "true", "yes", "on")
        elif "int" in t:
            out[key] = int(float(val))
        elif "float" in t:
            out[key] = float(val)
        else:
            out[key] = val
    return out


def build(world: str, algo: str, seed: int, env_overrides: dict, algo_overrides: dict, device: str):
    kind, algo_kw, env_kw = PRESETS[algo]
    env_kw = {**env_kw, **env_overrides}
    env = make_env(world, seed=seed, **env_kw)
    if kind == "value":
        learner = ValueLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                               ValueConfig(**{**algo_kw, **algo_overrides}), device, seed)
    else:
        learner = PPOLearner(env.n_agents, env.obs_dim, env.state_dim, env.n_actions,
                             PPOConfig(**{**algo_kw, **algo_overrides}), device, seed)
    return kind, env, env_kw, learner


def evaluate(world: str, env_kw: dict, learner, episodes: int = 20, seed: int = 10_000) -> dict:
    """Greedy rollouts on held-out seeds → mean episode metrics."""
    env = make_env(world, seed=seed, **env_kw)
    summaries = []
    for k in range(episodes):
        obs, state, _ = env.reset(seed=seed + k)
        done = False
        while not done:
            if isinstance(learner, PPOLearner):
                a, _, _ = learner.act(obs[None], state[None], greedy=True)
                a = a[0]
            else:
                a = learner.act(obs, eps=0.0)
            obs, state, _, _, done, info = env.step(a)
        summaries.append(info["episode"])
    out = {key: float(np.mean([s[key] for s in summaries]))
           for key in ("team_value", "crumbs", "pieces", "cut_work", "grabs", "stalled_pulls", "moving_pulls", "gini")}
    out["team_value_std"] = float(np.std([s["team_value"] for s in summaries]))
    out["per_ant_share"] = np.mean([s["per_ant_share"] for s in summaries], axis=0).round(3).tolist()
    out["max_team_value"] = env.max_team_value()
    return out


def train_value(env, learner: ValueLearner, steps: int, on_eval, eval_every: int):
    obs, state, _ = env.reset()
    last = {}
    for step in range(1, steps + 1):
        eps = learner.epsilon(step)
        actions = learner.act(obs, eps)
        next_obs, next_state, rewards, team, done, info = env.step(actions)
        learner.store(obs, state, actions, rewards, team, next_obs, next_state, float(info["terminated"]))
        obs, state = next_obs, next_state
        if done:
            obs, state, _ = env.reset()
        if learner.ready(step):
            last = learner.update()
        if step % eval_every == 0 or step == steps:
            on_eval(step, {**last, "epsilon": eps})


def train_ppo(world, env_kw, learner: PPOLearner, steps: int, seed: int, on_eval, eval_every: int):
    c = learner.cfg
    envs = [make_env(world, seed=seed + 1 + e, **env_kw) for e in range(c.num_envs)]
    obs = np.stack([e.reset(seed=seed + 1 + k)[0] for k, e in enumerate(envs)])
    state = np.stack([e._state() for e in envs])
    step, next_eval, last = 0, eval_every, {}
    while step < steps:
        buf = {k: [] for k in ("obs", "state", "actions", "logp", "values", "team", "done")}
        for _ in range(c.rollout_steps):
            a, logp, v = learner.act(obs, state)
            buf["obs"].append(obs); buf["state"].append(state)
            buf["actions"].append(a); buf["logp"].append(logp); buf["values"].append(v)
            team_r, dones, new_obs, new_state = [], [], [], []
            for e, env in enumerate(envs):
                o, s, _, r, d, _ = env.step(a[e])
                if d:
                    o, s, _ = env.reset()
                team_r.append(r); dones.append(float(d)); new_obs.append(o); new_state.append(s)
            buf["team"].append(np.array(team_r, np.float32)); buf["done"].append(np.array(dones, np.float32))
            obs, state = np.stack(new_obs), np.stack(new_state)
            step += c.num_envs
        roll = {k: np.stack(v) for k, v in buf.items()}
        roll["last_values"] = learner.act(obs, state)[2]
        last = learner.update(roll)
        if step >= next_eval or step >= steps:
            on_eval(step, last)
            next_eval += eval_every


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=["grid", "free"], default="grid")
    ap.add_argument("--algo", choices=sorted(PRESETS), default="iql")
    ap.add_argument("--steps", type=int, default=200_000, help="environment steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--env", nargs="*", default=[], metavar="KEY=VAL", help="environment overrides")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="learner overrides")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 lets runs share cores)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    from .envs import FreeConfig, GridConfig
    kind = PRESETS[args.algo][0]
    env_over = _parse_overrides(args.env, GridConfig if args.world == "grid" else FreeConfig)
    algo_over = _parse_overrides(args.set, ValueConfig if kind == "value" else PPOConfig)
    kind, env, env_kw, learner = build(args.world, args.algo, args.seed, env_over, algo_over, args.device)

    run = Path(args.out) / f"{args.world}-{args.algo}-s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({
        "world": args.world, "algo": args.algo, "seed": args.seed, "steps": args.steps,
        "env": {**asdict(env.cfg)}, "learner": asdict(learner.cfg)}, indent=1))
    t0 = time.time()
    csv_path = run / "metrics.csv"
    header_written = False

    def on_eval(step, train_stats):
        nonlocal header_written
        ev = evaluate(args.world, env_kw, learner, args.eval_episodes)
        row = {"step": step, "minutes": round((time.time() - t0) / 60, 2), **{k: v for k, v in ev.items() if k != "per_ant_share"},
               "per_ant_share": " ".join(map(str, ev["per_ant_share"])), **train_stats}
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if not header_written:
                w.writeheader()
                header_written = True
            w.writerow(row)
        torch.save(learner.state_dict(), run / "model.pt")
        print(f"[{args.world}/{args.algo}] step {step:>8,}  team food {ev['team_value']:5.2f}/{ev['max_team_value']:.0f}"
              f"  crumbs {ev['crumbs']:.2f}  pieces {ev['pieces']:.2f}  gini {ev['gini']:.2f}"
              f"  shares {ev['per_ant_share']}", flush=True)

    if kind == "value":
        train_value(env, learner, args.steps, on_eval, args.eval_every)
    else:
        train_ppo(args.world, env_kw, learner, args.steps, args.seed, on_eval, args.eval_every)
    print(f"done in {(time.time() - t0) / 60:.1f} min → {run}")


if __name__ == "__main__":
    main()
