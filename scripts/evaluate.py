"""Re-evaluate saved runs with the current metrics (e.g. after adding a new episode statistic).

    python scripts/evaluate.py runs/fix-compare/honest/free-*            # 50 held-out episodes each
    python scripts/evaluate.py --sample runs/fix-compare/honest/free-*   # each ant samples its action

Each run directory needs config.json and model.pt (the checkpoint from its last evaluation).
Writes <run>/eval.json and prints one line per run. The held-out seeds and the greedy policy
match training-time evaluation, so team_value should equal the last row of metrics.csv.
With --sample, each ant samples its own action as in training (PPO from π, value-based
ε-greedy at the final ε) and the result goes to <run>/eval_sampled.json instead.
"""
import csv
import json
import sys
from pathlib import Path

import torch

from antrl.train import build, evaluate


def reevaluate(run: Path, episodes: int = 50, sample: bool = False) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    _, _, env_kw, learner = build(cfg["world"], cfg["algo"], cfg["seed"], cfg["env"], cfg["learner"], "cpu")
    learner.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    ev = evaluate(cfg["world"], env_kw, learner, episodes, sample=sample)
    rows = list(csv.DictReader((run / "metrics.csv").open()))
    ev["logged_team_value"] = float(rows[-1]["team_value"]) if rows else None
    ev["step"] = int(rows[-1]["step"]) if rows else None
    (run / ("eval_sampled.json" if sample else "eval.json")).write_text(json.dumps(ev, indent=1))
    return ev


def main(paths, episodes=50, sample=False):
    torch.set_num_threads(1)
    for p in map(Path, paths):
        ev = reevaluate(p, episodes, sample)
        flag = "" if sample or ev["logged_team_value"] is None or abs(ev["team_value"] - ev["logged_team_value"]) < 1e-6 \
            else f"  (metrics.csv says {ev['logged_team_value']:.2f})"
        print(f"{p.name:<20} team {ev['team_value']:5.2f}  pieces {ev['pieces']:.2f}  paired {ev['pieces_paired']:.2f}"
              f"  carry team/solo {ev['carry_team']:.2f}/{ev['carry_solo']:.2f}{flag}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    episodes = next((int(a.split("=", 1)[1]) for a in args if a.startswith("--episodes=")), 50)
    sample = "--sample" in args
    main([a for a in args if not a.startswith("--")], episodes, sample)
