# Training Ants to Work Together — RL Environments

Multi-agent reinforcement learning for ant-like foraging. A colony of ant agents
learns to forage first **alone** (independent learners), then with **loosened
policies** that train together but still act alone (centralised training,
decentralised execution).

This repo implements the plan from the project's research-stage update:
two worlds, four jobs, a stage-1 baseline, stage-2 cooperative learners, and
metrics that score the team **and** every ant.

## The worlds

| | Grid world (`grid`) | Free world (`free`) |
|---|---|---|
| Space | discrete cells (default 10×10) | continuous arena (default 10×10), noisy headings |
| What an ant sees | a (2r+1)² window around itself (default r = 2) | the nearest few ants / crumbs / pieces / big food within `sense_radius` |
| Moving a piece | moves one cell only if **2 holders pull the same way** | moves by the **vector sum** of its holders' pulls ÷ weight: opposite pulls cancel, one ant drags at half speed |

Every ant also knows its own load, the direction home (ants path-integrate
back to the nest) and the time step, and can feel whether a partner is holding
its piece. Nothing in an observation points to food or assigns a role.

**The four jobs** (identical rules in both worlds):

1. **Find food**: crumbs and one big food item spawn away from the nest.
2. **Bring it home**: an ant carrying a crumb delivers it by reaching the nest.
3. **Cut big food**: `WORK` at the big item adds cut progress; every `cut_work` units releases a piece.
4. **Carry together**: pieces are heavy (`piece_weight = 2`) and move only when pulled.

**Actions**: `STAY` (also lets go of a held piece) · `NORTH` `SOUTH` `EAST` `WEST` (walk, or pull a held piece) · `WORK` (pick up a crumb / cut / grab a piece / keep holding).

**Optional mechanics** (off by default, set with `--env`):

| Switch | Effect |
|---|---|
| `hold_on_stay=true` | `STAY` keeps hold of a piece; a 7th action, `DROP`, lets go |
| `feel_tug=true` | each holder also senses its partners' pull directions from the last step, through the load (+2 features) |

**Rewards**: each step returns a reward per ant (credited to whoever earned it:
the crumb carrier, or the piece's holders split evenly) and a team reward equal
to their sum. Default values: crumb 1, piece 2, so the maximum per episode is
4 crumbs + 4 pieces = 12. Optional `shaping` adds small bonuses for pickups and
cuts, a grab bonus paid once per ant per piece, and progress credit when a
carried crumb or piece reaches a new closest distance to the nest.

## The learners

| Preset | Stage | Network | Reward | Teammates | Combination |
|---|---|---|---|---|---|
| `iql` | 1 · each ant alone | one Q-net per ant | own | blind | independent TD losses |
| `iql_team` | ablation | one Q-net per ant | team | blind | independent TD losses |
| `vdn` | 2 · loosened | shared Q-net (+ ant id) | team | sensed | Q_tot = Σ Q_i (Sunehag et al. 2018) |
| `qmix` | 2 · loosened | shared Q-net (+ ant id) | team | sensed | monotonic mixer on the global state, ∂Q_tot/∂Q_i ≥ 0 (Rashid et al. 2018) |
| `ippo` | 2 · comparison | shared policy | team | sensed | PPO, decentralised critic V(o_i) |
| `mappo` | 2 · loosened | shared policy | team | sensed | PPO, centralised critic V(s) (Yu et al. 2022) |

Value-based learners use replay, double-Q targets and periodically synced target
networks. PPO learners use GAE (advantage on `0.5·own + 0.5·team/n` reward),
clipped objectives and parallel environments; a timeout ends the episode, since
the time step is observed. Acting is always decentralised: each ant uses only
its own observation and samples its own action. Shared networks get a one-hot
ant id by default (`--set agent_id=false` removes it). Two opt-in PPO credit
options: `--set credit=team` (every ant's advantage from the team reward, as in
standard MAPPO) and `--set critic_per_ant=true` (MAPPO's V(s) with one output per
ant).

## What gets measured

Every evaluation (greedy policy, held-out seeds) reports:

- `team_value`: food value delivered per episode (max 12 by default)
- `crumbs`, `pieces`, `cut_work`, `grabs`: progress on each job
- `stalled_pulls` vs `moving_pulls`: how often holders pulled without the load moving (on the grid this includes a lone holder pulling while it waits for a partner)
- `pieces_paired`: delivered pieces that 2+ ants pulled together for at least half their path; `carry_team` / `carry_solo`: how far pieces moved under 2+ pullers vs one (in the free world one ant can drag a piece alone)
- `per_ant_share` and `gini`: each ant's share of the delivered value, and its inequality (0 = equal; high = some ants coast while others haul)

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                   # mechanics + algorithm tests

# stage 1 baseline, then a loosened learner, on the grid world
.venv/bin/python -m antrl.train --world grid --algo iql  --steps 300000
.venv/bin/python -m antrl.train --world grid --algo qmix --steps 300000

# the main result in RESULTS.md: IPPO learns cooperative transport in ~4–6 min
.venv/bin/python -m antrl.train --world grid --algo ippo --steps 1000000 --eval-every 100000 --eval-episodes 50 --env shaping=0.1

# free world, MAPPO, with reward shaping and 3 ants
.venv/bin/python -m antrl.train --world free --algo mappo --steps 500000 --env shaping=0.1 n_ants=3

# summarise every run under runs/
.venv/bin/python scripts/compare.py runs/
```

Any environment field can be overridden with `--env key=value` (see `GridConfig`,
`FreeConfig`), and any learner field with `--set key=value` (see `ValueConfig`,
`PPOConfig`). Each run writes `runs/<world>-<algo>-s<seed>/` with
`config.json`, `metrics.csv` and `model.pt`.

## Layout

```
antrl/envs/core.py         shared rules: the four jobs, rewards, episode stats, Gini
antrl/envs/grid.py         grid world (partial egocentric view, same-direction pulls)
antrl/envs/free.py         free world (continuous, force-summed transport, local sensing)
antrl/algos/networks.py    per-ant networks (shared / separate), VDN and QMIX mixers
antrl/algos/value_based.py IQL, VDN, QMIX
antrl/algos/ppo.py         IPPO, MAPPO
antrl/train.py             presets, training loops, evaluation, logging
scripts/compare.py         results table across runs and seeds
scripts/evaluate.py        re-score saved runs with the current metrics
scripts/coordination.py    what the two holders of a piece do, step by step
tests/                     environment mechanics and algorithm checks
```

## Status

The environments and all six learners run end to end. The test suite checks the
mechanics (cutting, same-direction and force-summed transport, delivery credit,
partial views, the optional switches) and the algorithms (QMIX monotonicity,
VDN sum, per-ant IQL targets, PPO updates for both critics and both credit
options, independent action sampling, timeouts as terminal states for both
learner families) and the paired-carry metrics.

### Results at a glance

1M environment steps unless marked, 50 held-out greedy episodes per seed, mean ± SD over seeds. Full tables, learning curves and caveats are in [RESULTS.md](RESULTS.md).

| Learner | Seeds | Grid team food (/12) | Grid pieces (/4) | Free team food (/12) | Free pieces: paired / delivered | Free at 3M steps |
|---|---:|---:|---:|---:|---:|---:|
| `ippo` | 5 | **11.75 ± 0.08** | **3.90** | **8.49 ± 1.04** | **1.94 / 2.66** | **10.69 ± 0.49** |
| `mappo` | 5 | 11.42 ± 0.39 | 3.77 | 7.91 ± 0.78 | 1.58 / 2.44 | 10.16 ± 0.60 |
| `mappo`, one value output per ant | 3 | **11.75 ± 0.11** | 3.89 | 7.46 ± 0.54 | 1.41 / 2.09 | – |
| `mappo`, team reward only | 3 | 11.33 ± 0.23 | 3.69 | 4.14 ± 0.53 | 0.75 / 1.22 | – |
| `vdn` | 3 | 5.51 ± 1.50 | 1.64 | 6.84 ± 1.52 | 1.37 / 2.92 | – |
| `iql` | 3 | 3.29 ± 0.17 | 0 | 7.82 ± 0.88 | 1.55 / 2.53 | – |
| `iql_team` | 3 | 2.73 ± 0.22 | 0 | 3.47 ± 0.49 | 0.30 / 0.93 | – |
| `qmix` | 3 | 0.08 ± 0.10 | 0 | 0.37 ± 0.52 | 0 / 0 | – |

A piece is **paired** when 2+ ants pulled it together for at least half of its path. On the grid every delivered piece is paired, since one ant can't move a piece there.

What the numbers show:

- **The grid is the test of cooperation.**
  - IPPO and MAPPO learn the full task from honest local observations, including two-ant transport of nearly every piece.
  - VDN learns some transport. IQL and IQL-team forage crumbs only.
  - QMIX rises, then collapses, in both worlds. The cause has not been diagnosed.
- **Team food in the free world doesn't measure cooperation.** One ant can drag a piece at half speed, so IQL, blind to its teammates, scores about as much as MAPPO. The paired count separates the learners: 73% of IPPO's deliveries are paired, against 47–61% for VDN and IQL (seeds 0–2).
- **How pairs coordinate.**
  - The two holders of a piece always see identical observations, so a shared policy coordinates them by learning to ignore the ant id and pull toward the nest. Holders agree on 93–94% of grid steps.
  - Most stalled pulls are a lone holder waiting for its partner.
- **Credit matters for MAPPO.**
  - Each ant's own reward in the advantage is important: with the team reward alone, the free-world score halves.
  - One value output per ant learns the grid fastest.
- **Longer training helps in the free world.** IPPO is still improving at 3M steps.

### Progress

| Date | Commit | What happened |
|---|---|---|
| Nov 1 | `a3196e4`, `a216048` | Built both worlds, the six learners and the metrics. First results: every learner foraged crumbs, none carried a piece (best: MAPPO 3.50 / 12). |
| Nov 1 | `1166e33` | Reported 12 / 12 for QMIX, IPPO and MAPPO. **Withdrawn:** a target oracle in the observation gave each ant a role and a step direction. |
| Nov 8 | `e7cfa42` | Removed the oracle and fixed reward farming in the shaping. Its transport results relied on hand-coded holder synchronisation and are **superseded**. |
| Nov 8 | `e478b7b` | Fixed the PPO timeout bug, the main blocker, and removed the hand-coded synchronisation. IPPO and MAPPO learn transport honestly: grid 11.77 and 11.48 at 1M steps. |
| Nov 15 | this update | All six learners at 1M steps in both worlds, with a timeout fix for the value-based learners. Added the paired-carry metric, the coordination analysis, the MAPPO credit variants, 3M-step free-world runs, and 5 seeds for IPPO and MAPPO. |

### Next steps

- Diagnose QMIX's collapse and tune the value-based learners.
- Measure agreement during training, not only in greedy evaluation.
- Train the free world past 3M steps.
- Run the MAPPO credit variants with more seeds.
- Try recurrent agents for the partial views.

## References

- Papoudakis et al., *Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks*, NeurIPS 2021 Datasets & Benchmarks.
- Sunehag et al., *Value-Decomposition Networks for Cooperative Multi-Agent Learning*, AAMAS 2018.
- Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep MARL*, ICML 2018.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.
- Feinerman et al., *The physics of cooperative transport in groups of ants*, Nature Physics 2018.
