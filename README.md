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
ant id by default (`--set agent_id=false` removes it).

## What gets measured

Every evaluation (greedy policy, held-out seeds) reports:

- `team_value`: food value delivered per episode (max 12 by default)
- `crumbs`, `pieces`, `cut_work`, `grabs`: progress on each job
- `stalled_pulls` vs `moving_pulls`: how often holders pulled without the load moving (a coordination failure)
- `per_ant_share` and `gini`: each ant's share of the delivered value, and its inequality (0 = equal; high = some ants coast while others haul)

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                   # mechanics + algorithm tests

# stage 1 baseline, then a loosened learner, on the grid world
.venv/bin/python -m antrl.train --world grid --algo iql  --steps 300000
.venv/bin/python -m antrl.train --world grid --algo qmix --steps 300000

# the main result in RESULTS.md: IPPO learns cooperative transport in ~6 min
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
tests/                     environment mechanics and algorithm checks
```

## Status

The environments and all six learners run end to end. The test suite checks the
mechanics (cutting, same-direction and force-summed transport, delivery credit,
partial views, the optional switches) and the algorithms (QMIX monotonicity,
VDN sum, per-ant IQL targets, PPO updates for both critics, independent action
sampling, timeouts as terminal states).

Results are in [RESULTS.md](RESULTS.md). On the grid world, IPPO and MAPPO learn
the full task, including two-ant transport of nearly every piece, from honest
local observations: 11.77 and 11.48 of 12 food per episode after 1M steps
(3 seeds). In the free world they reach 8.55 and 7.85 of 12 at 1M steps and are
still improving. Earlier results from a target oracle and from hand-coded holder
synchronisation are withdrawn there.

Next steps: the value-based learners at 1M steps, longer free-world training
with a per-delivery count of holders, more seeds, standard MAPPO credit (team
reward or per-ant value heads), and recurrent agents for the partial views.

## References

- Papoudakis et al., *Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks*, NeurIPS 2021 Datasets & Benchmarks.
- Sunehag et al., *Value-Decomposition Networks for Cooperative Multi-Agent Learning*, AAMAS 2018.
- Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep MARL*, ICML 2018.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.
- Feinerman et al., *The physics of cooperative transport in groups of ants*, Nature Physics 2018.
