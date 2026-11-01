"""Summarise finished runs into one table (last evaluation of each run, mean ± sd over seeds).

    python scripts/compare.py runs/
"""
import csv
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
        groups[(str(f.parent.parent.relative_to(root)), world, algo)].append(rows[-1])
    if not groups:
        sys.exit(f"no metrics.csv under {root}")
    cols = ("team_value", "crumbs", "pieces", "cut_work", "stalled_pulls", "gini")
    print(f"{'group':<14}{'world':<6}{'algo':<10}{'n':>3}  " + "".join(f"{c:>17}" for c in cols) + "   steps")
    for (group, world, algo), rows in sorted(groups.items()):
        cells = []
        for c in cols:
            v = np.array([float(r[c]) for r in rows])
            cells.append(f"{v.mean():>10.2f} ± {v.std():<4.2f}")
        print(f"{group:<14}{world:<6}{algo:<10}{len(rows):>3}  " + "".join(cells) + f"   {rows[0]['step']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
