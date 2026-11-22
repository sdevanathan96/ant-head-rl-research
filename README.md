# Training Ants to Work Together — RL Environments

## Contributors

| Name | Role |
|---|---|
| Sudhanva Devanathan | Environments, learners, training and evaluation |

## Project brief

Multi-agent reinforcement learning for ant-like foraging. A colony of ant agents
learns to forage first **alone** (independent learners), then with **loosened
policies** that train together but still act alone (centralised training,
decentralised execution).

The project has two worlds (a grid and a continuous arena) with the same four
jobs, a stage-1 baseline of independent learners, stage-2 cooperative learners,
and metrics that score the team **and** every ant.

## Problem statement

Ants move food too heavy for one ant by carrying it together, each acting on
what it senses around itself (Feinerman et al. 2018). Can agents that see only
their own surroundings learn this from reward alone, with no role assigned and
no scripted coordination? And which multi-agent learners manage it?

Team food alone can't answer the second question. Where one ant can drag a load
by itself, ants that ignore each other score nearly as well as ants that
cooperate. So the project also measures whether each piece was actually carried
by two ants, and what the two holders do step by step.

## End goal

- Environments in which cooperative transport has to be learned: nothing in an
  observation points to food or assigns a role, and no holder behaviour is
  hand-coded.
- A fair comparison, over several seeds and in both worlds, of independent
  learners (IQL) against cooperative ones (VDN, QMIX, IPPO, MAPPO).
- Metrics that separate cooperation from solo effort (paired pieces, the
  team/solo carry split, each ant's share and its Gini), and an account of how
  the learned pairs coordinate.
- Success: learners that come close to the maximum score from local
  observations, with the paired-piece count confirming two-ant transport, and
  results that anyone can reproduce from this folder.

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

The value-based defaults (learning rate 5e-4, target sync every 500 updates)
are unchanged, but they are not the best settings. QMIX diverges with them. The
tuned settings in RESULTS.md §8 are `--set lr=2.5e-4 target_update=2000` for VDN
and QMIX, and `--set lr=2.5e-4` for IQL and IQL-team.

## What gets measured

Every evaluation (greedy policy, held-out seeds) reports:

- `team_value`: food value delivered per episode (max 12 by default)
- `crumbs`, `pieces`, `cut_work`, `grabs`: progress on each job
- `stalled_pulls` vs `moving_pulls`: how often holders pulled without the load moving (on the grid this includes a lone holder pulling while it waits for a partner)
- `pieces_paired`: delivered pieces that 2+ ants pulled together for at least half their path; `carry_team` / `carry_solo`: how far pieces moved under 2+ pullers vs one (in the free world one ant can drag a piece alone)
- `per_ant_share` and `gini`: each ant's share of the delivered value, and its inequality (0 = equal; high = some ants coast while others haul)

With `--coord`, every evaluation also records what the two holders of a piece do (both pull the same way, split, one waits, let go), once greedily and once with each ant sampling its action as in training. `scripts/evaluate.py --sample` scores a saved model with sampled actions instead of greedy ones. Neither changes a training run.

## Quick start

Run every command from this folder (`projects/learning-environments/ant-foraging/`).
The results were produced on CPU with Python 3.14.7, PyTorch 2.15.0 and NumPy 2.5.4.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                   # mechanics + algorithm tests

# stage 1 baseline, then a loosened learner, on the grid world
.venv/bin/python -m antrl.train --world grid --algo iql  --steps 300000
.venv/bin/python -m antrl.train --world grid --algo qmix --steps 300000 --set lr=2.5e-4 target_update=2000

# the main result in RESULTS.md: IPPO learns cooperative transport in ~4–6 min
.venv/bin/python -m antrl.train --world grid --algo ippo --steps 1000000 --eval-every 100000 --eval-episodes 50 --env shaping=0.1

# free world, MAPPO, with reward shaping and 3 ants
.venv/bin/python -m antrl.train --world free --algo mappo --steps 500000 --env shaping=0.1 n_ants=3

# summarise every run under runs/, or the committed results behind RESULTS.md
.venv/bin/python scripts/compare.py runs/
.venv/bin/python scripts/compare.py results/fix-compare
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
antrl/coordination.py      what the holders of a piece do, step by step (greedy or sampled)
scripts/compare.py         results table across runs and seeds
scripts/evaluate.py        re-score saved runs with the current metrics
scripts/coordination.py    the coordination analysis for saved runs
tests/                     environment mechanics and algorithm checks
results/fix-compare/       evaluation files of every run in RESULTS.md, and ten headline models
```

## Status

The environments and all six learners run end to end. The test suite checks the
mechanics (cutting, same-direction and force-summed transport, delivery credit,
partial views, the optional switches) and the algorithms (QMIX monotonicity,
VDN sum, per-ant IQL targets, the value-based update diagnostics, PPO updates
for both critics and both credit options, independent action sampling, timeouts
as terminal states for both learner families), the paired-carry metrics, and
that measuring coordination or sampled scores leaves a training run unchanged.

### Results at a glance

1M environment steps unless marked, 50 held-out episodes per seed, mean ± SD over seeds. Scores are greedy, except the "sampled" column, where each ant samples its action as in training. Full tables, learning curves and caveats are in [RESULTS.md](RESULTS.md).

| Learner | Seeds | Grid team food (/12) | Grid pieces (/4) | Free team food (/12) | Free pieces: paired / delivered | Free, sampled | Free at 10M steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ippo` | 5 | **11.75 ± 0.08** | **3.90** | 8.49 ± 1.04 | 1.94 / 2.66 | **9.59 ± 0.45** | 10.91 ± 0.52 |
| `mappo` | 5 | 11.42 ± 0.39 | 3.77 | 7.91 ± 0.78 | 1.58 / 2.44 | 9.43 ± 0.46 | **11.31 ± 0.27** |
| `mappo`, one value output per ant | 5 | 11.72 ± 0.19 | 3.88 | 6.72 ± 1.23 | 1.14 / 1.79 | 8.20 ± 0.74 | – |
| `mappo`, team reward only | 5 | 11.10 ± 0.40 | 3.60 | 3.26 ± 1.15 | 0.60 / 0.95 | 4.76 ± 1.49 | – |
| `qmix`, tuned | 3 | 8.44 ± 1.35 | 2.66 | 5.35 ± 2.49 | 0.63 / 1.07 | 5.47 ± 2.51 | – |
| `vdn`, tuned | 3 | 8.23 ± 0.14 | 2.65 | **9.63 ± 0.16** | 1.59 / 3.46 | 9.48 ± 0.27 | – |
| `iql`, tuned | 3 | 4.52 ± 0.29 | 0.58 | 9.17 ± 0.59 | **2.02 / 3.04** | 9.45 ± 0.46 | – |
| `iql_team`, tuned | 3 | 3.16 ± 0.06 | 0 | 5.40 ± 0.73 | 0.87 / 1.80 | 5.43 ± 0.57 | – |
| `vdn` | 3 | 5.51 ± 1.50 | 1.64 | 6.84 ± 1.52 | 1.37 / 2.92 | 6.87 ± 1.18 | – |
| `iql` | 3 | 3.29 ± 0.17 | 0 | 7.82 ± 0.88 | 1.55 / 2.53 | 7.59 ± 0.60 | – |
| `iql_team` | 3 | 2.73 ± 0.22 | 0 | 3.47 ± 0.49 | 0.30 / 0.93 | 3.51 ± 0.53 | – |
| `qmix` | 3 | 0.08 ± 0.10 | 0 | 0.37 ± 0.52 | 0 / 0 | 0.41 ± 0.58 | – |

A piece is **paired** when 2+ ants pulled it together for at least half of its path. On the grid every delivered piece is paired, since one ant can't move a piece there. The 10M-step runs use seeds 0–2. "Tuned" rows use a smaller learning rate, plus a 4× slower target sync for VDN and QMIX (RESULTS.md §8); untuned rows use the defaults. PPO was not tuned.

What the numbers show:

- **The grid is the test of cooperation.**
  - IPPO and MAPPO learn the full task from honest local observations, including two-ant transport of nearly every piece.
  - Tuned, VDN and QMIX carry about 2.7 of 4 pieces (8.2–8.4 / 12), well behind PPO. Their holders pull the same way on 65–71% of steps, against 93–94% for PPO.
  - IQL and IQL-team forage mostly crumbs. Tuned IQL moves an occasional piece, when two ants that can't see each other happen to pull the same way.
- **QMIX's collapse is diagnosed and fixed.** With the target network synced every 500 updates, the team value estimate overshoots early. QMIX's mixer absorbs the overshoot and diverges. Syncing every 2,000 updates fixes it: grid QMIX goes from 0.08 to 8.44.
- **The free world doesn't separate cooperative from independent learners at 1M steps.** One ant can drag a piece at half speed.
  - With default settings, IQL (blind to its teammates) scores about as much as MAPPO greedily. IPPO pairs more of its deliveries (73%, against 47–61% for VDN and IQL), and PPO scores clearly more when sampled (9.4–9.6 against 7.6).
  - Tuned, IQL and VDN reach 9.2–9.6 and match PPO either way. IQL's paired-piece count (2.02) equals IPPO's.
- **Greedy play can deadlock.** Two holders that pull different ways leave the piece where it is, see nearly the same thing next step, and a greedy policy repeats the split. This is why PPO scores higher when sampled before its training settles, and why MAPPO's weakest grid seed (10.82 greedy) scores 11.64 when sampled.
- **How pairs coordinate.**
  - The two holders of a piece always see identical observations, so a shared policy coordinates them by learning to ignore the ant id and pull toward the nest. Holders agree on 93–94% of grid steps.
  - Agreement rises with transport during training. Most stalled pulls are a lone holder waiting for its partner.
- **Credit matters for MAPPO** (5 seeds).
  - Each ant's own reward in the advantage is important: with the team reward alone, the free-world score falls by more than half.
  - One value output per ant gives the best and most consistent MAPPO on the grid and learns it fastest, but trails the default in the free world.
- **Longer training closes the gap in the free world.** MAPPO keeps improving until about 8M steps and levels off at about 11.4 of 12, close to its grid score; IPPO levels off earlier, at about 11. By then 80–83% of delivered pieces are paired.

### Progress

| Date | Commit | What happened |
|---|---|---|
| Nov 1 | `600355b`, `db2195c` | Built both worlds, the six learners and the metrics. First results: every learner foraged crumbs, none carried a piece (best: MAPPO 3.50 / 12). |
| Nov 1 | `bfe4eb3` | Reported 12 / 12 for QMIX, IPPO and MAPPO. **Withdrawn:** a target oracle in the observation gave each ant a role and a step direction. |
| Nov 8 | `e5295c0` | Removed the oracle and fixed reward farming in the shaping. Its transport results relied on hand-coded holder synchronisation and are **superseded**. |
| Nov 8 | `518d820` | Fixed the PPO timeout bug, the main blocker, and removed the hand-coded synchronisation. IPPO and MAPPO learn transport honestly: grid 11.77 and 11.48 at 1M steps. |
| Nov 15 | `3c4e2f3` | All six learners at 1M steps in both worlds, with a timeout fix for the value-based learners. Added the paired-carry metric, the coordination analysis, the MAPPO credit variants, 3M-step free-world runs, and 5 seeds for IPPO and MAPPO. |
| Nov 20 | `d013f33` | Moved into this repo with its full history. Added the contributor, problem and goal sections, the work log, and the evaluation files behind every table. |
| Nov 21 | this update | Free world to 10M steps; seeds 3–4 for the MAPPO credit variants; holder agreement measured during training (`--coord`); sampled evaluation, which shows greedy scores understate PPO in the free world. |
| Nov 22 | this update | Diagnosed QMIX's collapse (fast target sync, amplified by its mixer) and retuned the value-based learners: grid QMIX 0.08 → 8.44, VDN 5.51 → 8.23; free VDN 9.63 and IQL 9.17. PPO still leads on the grid. |

The project started in a personal repository, `sdevanathan96/ant-head-rl-research`. Its history was imported with every file moved into this folder. Messages, authors and dates are unchanged, but the commit IDs differ, so the IDs above are this repo's. Older commit messages still cite the original IDs; [WORK_PROGRESS.md](WORK_PROGRESS.md) lists both.

### Next steps

- Tune PPO with the same care, so the comparison with the tuned value-based learners is fair.
- Train tuned QMIX past 1M steps and add seeds: it is still improving, and its free-world seeds disagree.
- Find out why value-based holders agree less than PPO's (65–71% against 93–94% on the grid).
- Make the free world a sharper test of cooperation, for example by not letting one ant drag a piece alone.
- Decide whether greedy or sampled scores should be the main protocol, or report both throughout.
- Find out why IPPO's free-world seeds stay apart while MAPPO's converge, and why some greedy policies deadlock.
- Run the optional switches and the credit variants at longer budgets in the free world.
- Try recurrent agents for the partial views.

## References

- Papoudakis et al., *Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks*, NeurIPS 2021 Datasets & Benchmarks.
- Sunehag et al., *Value-Decomposition Networks for Cooperative Multi-Agent Learning*, AAMAS 2018.
- Rashid et al., *QMIX: Monotonic Value Function Factorisation for Deep MARL*, ICML 2018.
- Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.
- Feinerman et al., *The physics of cooperative transport in groups of ants*, Nature Physics 2018.
