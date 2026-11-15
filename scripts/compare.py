"""Summarise finished runs into one table (last evaluation of each run, mean ± sd over seeds).

    python scripts/compare.py runs/

Runs logged before ``pieces_paired`` existed take it from ``eval.json`` (written by
scripts/evaluate.py); without either, the cell shows "–".
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(root="runs"):
    groups = defaultdict(list)
    for f in sorted(Path(root).glob("**/metrics.csv")):
        rows = list(csv.DictReader(f.open()))
        if not rows:
            continue
        world, algo, _seed = f.parent.name.rsplit("-", 2)
        last = dict(rows[-1])
        extra = f.parent / "eval.json"
        if "pieces_paired" not in last and extra.exists():
            last["pieces_paired"] = json.loads(extra.read_text())["pieces_paired"]
        groups[(str(f.parent.parent.relative_to(root)), world, algo)].append(last)
    if not groups:
        sys.exit(f"no metrics.csv under {root}")
    cols = ("team_value", "crumbs", "pieces", "pieces_paired", "cut_work", "moving_pulls", "stalled_pulls", "gini")
    print(f"{'group':<16}{'world':<6}{'algo':<10}{'n':>3}  " + "".join(f"{c:>17}" for c in cols) + "   steps")
    for (group, world, algo), rows in sorted(groups.items()):
        cells = []
        for c in cols:
            if any(c not in r for r in rows):
                cells.append(f"{'–':>17}")
                continue
            v = np.array([float(r[c]) for r in rows])
            cells.append(f"{v.mean():>10.2f} ± {v.std():<4.2f}")
        print(f"{group:<16}{world:<6}{algo:<10}{len(rows):>3}  " + "".join(cells) + f"   {rows[0]['step']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
