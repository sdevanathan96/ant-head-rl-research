"""How do the two holders of a piece coordinate? Rollouts of saved runs on 50 held-out episodes.

    python scripts/coordination.py runs/fix-compare/honest/grid-ippo-s*            # greedy
    python scripts/coordination.py --sample runs/fix-compare/honest/grid-ippo-s*   # each ant samples

Writes coordination.json (greedy) or coordination_sampled.json (--sample: each ant samples its own
action, as in training) into each run folder. The metrics (agree, split, wait, hold, release,
nestward, the diagonal-step shares, x_share_by_ant and the lone-holder shares) are defined in
antrl/coordination.py, which `python -m antrl.train --coord` also uses during training.
"""
import json
import sys
from pathlib import Path

import torch

from antrl.coordination import analyse_policy
from antrl.train import build


def load(run: Path):
    cfg = json.loads((run / "config.json").read_text())
    _, _, env_kw, learner = build(cfg["world"], cfg["algo"], cfg["seed"], cfg["env"], cfg["learner"], "cpu")
    learner.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return cfg["world"], env_kw, learner


def analyse(run: Path, episodes: int = 50, seed: int = 10_000, sample: bool = False) -> dict:
    world, env_kw, learner = load(run)
    return analyse_policy(world, env_kw, learner, episodes, seed, sample=sample)


def main(args):
    torch.set_num_threads(1)
    sample = "--sample" in args
    for p in (Path(a) for a in args if a != "--sample"):
        r = analyse(p, sample=sample)
        (p / ("coordination_sampled.json" if sample else "coordination.json")).write_text(json.dumps(r, indent=1))
        print(f"{p.name:<18} co-hold/ep {r['co_holding_steps_per_ep']:5.1f}  same_obs {r['same_obs']:.2f}  "
              f"agree {r['agree']:.2f} split {r['split']:.2f} wait {r['wait']:.2f} hold {r['hold']:.2f} "
              f"release {r['release']:.2f}  nestward {r['nestward']:.2f}  diag {r['diag_steps']:.2f} "
              f"diag_agree {r['diag_agree']:.2f} straight_agree {r['straight_agree']:.2f} "
              f"larger_axis {r['larger_axis']:.2f}  x_share_by_ant {r['x_share_by_ant']}  |  lone holder/ep "
              f"{r['solo_steps_per_ep']:5.1f}: pull {r['solo_pull']:.2f} wait {r['solo_wait']:.2f} "
              f"release {r['solo_release']:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
