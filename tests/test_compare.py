"""scripts/compare.py: greedy rows from metrics.csv, sampled rows from eval_sampled.json."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("compare", Path(__file__).parents[1] / "scripts" / "compare.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

COLS = ("team_value", "crumbs", "pieces", "pieces_paired", "cut_work", "moving_pulls", "stalled_pulls", "gini")


def _run(root, name, greedy, sampled=None):
    run = root / "group" / name
    run.mkdir(parents=True)
    (run / "metrics.csv").write_text("step," + ",".join(COLS) + "\n1000," + ",".join([str(greedy)] * len(COLS)) + "\n")
    if sampled is not None:
        (run / "eval_sampled.json").write_text(json.dumps({**{c: sampled for c in COLS}, "step": 1000}))


def _team_cell(out, algo):
    line = next(l for l in out.splitlines() if f" {algo} " in l)
    return line.split()[4:7]                        # n, mean, "±", sd → mean ± sd of team_value


def test_compare_reads_greedy_metrics_and_sampled_evals(tmp_path, capsys):
    _run(tmp_path, "grid-ippo-s0", greedy=10, sampled=11)
    _run(tmp_path, "grid-ippo-s1", greedy=8, sampled=12)
    _run(tmp_path, "grid-iql-s0", greedy=3)          # no sampled re-score: left out of --sample

    compare.main(str(tmp_path))
    greedy = capsys.readouterr().out
    assert _team_cell(greedy, "ippo") == ["9.00", "±", "1.00"]
    assert " iql " in greedy

    compare.main(str(tmp_path), sample=True)
    sampled = capsys.readouterr().out
    assert _team_cell(sampled, "ippo") == ["11.50", "±", "0.50"]
    assert " iql " not in sampled
