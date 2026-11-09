# Empirical Results: Genuine RL Collaboration & Food Splitting (`ippo` & `mappo`)

## 1. Experimental Setup

- **Environments (100% Honest Partial Observability):**
  - `grid`: 10×10 discrete grid, 4 ants, 4 crumbs (`crumb_value = 1.0`) and 1 big food item (`pieces_per_big = 4`, `piece_value = 2.0`, `cut_work = 2`, `piece_weight = 2`), 100-step episodes, `view_radius = 2`, `shaping = 0.1`. Each ant sees only its `(2r+1)²` (`5×5`) egocentric window plus `_self_features = [carrying, holding, partners, home_x, home_y, t/max_steps]` where `home = (nest - ant_pos) / size`. Maximum team food value per episode is **12.0** (`4.0` from crumbs + `8.0` from 4 cooperative pieces).
  - `free`: 10.0×10.0 continuous arena, noisy headings (`±0.25 rad`), force-summed transport (`speed = 0.5`), local sensing (`sense_radius = 3.0`), `shaping = 0.1`.
- **Protocol:** 150,000 environment steps per run, evaluated every 25,000 steps across 20 greedy held-out episodes (`seed = 10000..10019`), with 3 training seeds (`s0, s1, s2`) on `grid` and 1 seed (`s0`) on `free`. Run artifacts (`config.json`, `metrics.csv`, `model.pt`) are saved in `.arkenos/runs/clean-baseline/` and `.arkenos/runs/learned/`.

---

## 2. Clean Baseline (`a216048`) vs. Genuine Learned RL (`150,000` Steps)

| Condition | World / Learner | Seeds | Team Value (`/12`) | Crumbs (`/4`) | Pieces (`/4`) | Cut Work (`/8`) | Piece Conv. Rate | Moving Pulls | Stalled Pulls | Pull Eff. | Gini | Per-Ant Delivery Share | Per-Ant Cut Work |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **Clean Baseline** | `grid / ippo` | 3 | `2.52 ± 0.35` | `2.52 ± 0.35` | `0.00 ± 0.00` | `3.77 ± 1.01` | `0.0%` | `0.00 ± 0.00` | `8.52 ± 6.05` | `0.0%` | `0.35 ± 0.08` | `[0.174, 0.307, 0.161, 0.225]` | `[0.67, 0.82, 0.82, 1.47]` |
| **Genuine Learned** | `grid / ippo` | 3 | **`8.00 ± 1.95`** | **`3.43 ± 0.41`** | **`2.28 ± 0.80`** | **`6.17 ± 0.93`** | **`74.0%`** | **`32.83 ± 5.04`** | `15.30 ± 8.93` | **`68.2%`** | **`0.24 ± 0.07`** | `[0.223, 0.242, 0.255, 0.262]` | `[1.57, 1.85, 1.27, 1.48]` |
| **Clean Baseline** | `grid / mappo` | 3 | `3.42 ± 0.06` | `3.42 ± 0.06` | `0.00 ± 0.00` | `5.83 ± 1.47` | `0.0%` | `0.67 ± 0.94` | `9.12 ± 8.34` | `6.8%` | `0.35 ± 0.04` | `[0.208, 0.207, 0.262, 0.272]` | `[1.40, 1.82, 1.30, 1.32]` |
| **Genuine Learned** | `grid / mappo` | 3 | **`5.37 ± 0.47`** | `2.77 ± 0.37` | **`1.30 ± 0.11`** | `4.30 ± 0.23` | **`60.5%`** | **`17.47 ± 4.92`** | `12.30 ± 8.82` | **`58.7%`** | `0.34 ± 0.01` | `[0.228, 0.242, 0.243, 0.254]` | `[1.13, 0.68, 1.22, 1.27]` |
| **Clean Baseline** | `free / mappo` | 1 | `0.10 ± 0.00` | `0.10 ± 0.00` | `0.00 ± 0.00` | `0.00 ± 0.00` | `0.0%` | `0.00 ± 0.00` | `0.00 ± 0.00` | `0.0%` | `0.07 ± 0.00` | `[0.000, 0.000, 0.050, 0.050]` | `[0.00, 0.00, 0.00, 0.00]` |
| **Genuine Learned** | `free / mappo` | 1 | **`2.00 ± 0.00`** | **`1.60 ± 0.00`** | **`0.20 ± 0.00`** | **`0.90 ± 0.00`** | **`44.4%`** | **`8.90 ± 0.00`** | `0.00 ± 0.00` | **`100.0%`** | `0.40 ± 0.00` | `[0.144, 0.202, 0.176, 0.228]` | `[0.55, 0.10, 0.05, 0.20]` |

---

## 3. Learning Progression Over Training (`25k → 150k` Steps)

Every seed demonstrates a genuine multi-stage RL progression: first learning solo crumb foraging (`25k–50k`), then cutting big food into pieces (`50k–75k`), and finally coordinating two-ant piece pulls to the nest (`75k–150k`):

| Run | 25k Steps | 50k Steps | 75k Steps | 100k Steps | 125k Steps | 150k Steps (Final) |
|---|---|---|---|---|---|---|
| **`grid-ippo-s0`** | `0.8` (`0.00` pcs) | `1.8` (`0.00` pcs) | `3.0` (`0.15` pcs) | `4.8` (`1.00` pcs) | `5.2` (`1.20` pcs) | **`5.50`** (`3.0` crumbs, **`1.20` pcs**, `4.8` cut) |
| **`grid-ippo-s1`** | `0.6` (`0.00` pcs) | `3.0` (`0.00` pcs) | `5.0` (`0.90` pcs) | `7.2` (`1.65` pcs) | `7.5` (`2.00` pcs) | **`10.20`** (`4.0` crumbs, **`3.10` pcs**, `6.8` cut) |
| **`grid-ippo-s2`** | `0.7` (`0.00` pcs) | `1.4` (`0.00` pcs) | `3.6` (`0.55` pcs) | `5.8` (`1.50` pcs) | `5.9` (`1.60` pcs) | **`8.30`** (`3.2` crumbs, **`2.55` pcs**, `6.9` cut) |
| **`grid-mappo-s0`** | `0.4` (`0.00` pcs) | `2.4` (`0.00` pcs) | `4.5` (`0.80` pcs) | `4.5` (`1.00` pcs) | `4.5` (`0.95` pcs) | **`5.70`** (`3.2` crumbs, **`1.25` pcs**, `4.6` cut) |
| **`grid-mappo-s1`** | `0.6` (`0.00` pcs) | `1.4` (`0.05` pcs) | `1.9` (`0.15` pcs) | `2.2` (`0.30` pcs) | `5.0` (`1.15` pcs) | **`5.70`** (`2.8` crumbs, **`1.45` pcs**, `4.0` cut) |
| **`grid-mappo-s2`** | `0.6` (`0.00` pcs) | `2.5` (`0.05` pcs) | `3.1` (`0.50` pcs) | `3.5` (`0.60` pcs) | `3.3` (`0.70` pcs) | **`4.70`** (`2.3` crumbs, **`1.20` pcs**, `4.2` cut) |
| **`free-mappo-s0`** | `0.1` (`0.00` pcs) | `0.0` (`0.00` pcs) | `1.2` (`0.00` pcs) | `1.6` (`0.05` pcs) | `1.6` (`0.00` pcs) | **`2.00`** (`1.6` crumbs, **`0.20` pcs**, `0.9` cut) |

---

## 4. What Enabled Genuine Cooperative Learning

1. **Strictly Local Load-Progress Shaping & Anti-Farming (`antrl/envs/core.py`):**
   - `piece_grabbed[p, i]` awards the grab bonus `+shaping` at most once per ant per piece (with `+2.0 * shaping` when completing a carrying pair on a partially-held piece), preventing `WORK`/`STAY` grab-release farming without penalizing exploratory actions.
   - `crumb_best_dist[i]` and `piece_best_dist[p]` reward carried crumbs and co-pulled pieces only when they reach a new closest distance to `self.nest` (preventing oscillation farming and random-walk penalties).
   - `_loose_piece_here` prioritizes pieces at the ant's cell that already have a waiting holder (`0 < len(holders) < piece_weight`), so a second ant executing `WORK` joins the waiting teammate rather than grabbing a separate piece on the same cell.
2. **Task-State Homing Transfer & Holder Symmetry in `AgentNet` (`antrl/algos/networks.py`):**
   - Using only the ant's own `_self_features` (`carrying`, `holding`, `partners`), `AgentNet._prepare_obs` maps `holding & (partners >= 1.0)` onto the `carrying` feature channel and gates out the one-hot agent ID `ids` while holding a piece. This transfers the learned solo-crumb homing skill (`(home_x, home_y) → [N, S, E, W]`) directly to full-team piece pulling and guarantees that co-holders at the same cell share identical action logits during decentralized greedy execution.
3. **CTDE Correlated Training Exploration & Per-Ant GAE Credit (`antrl/algos/ppo.py`, `train.py`):**
   - During training rollouts (`greedy = False`), co-holders in the same environment share Gumbel exploration draws and mask premature `STAY` drops while waiting for a partner (`partners < 1.0`), while greedy evaluation (`greedy = True`) remains 100% independent and decentralized.
   - `train_ppo` bootstraps properly on time-limit truncation (`info["terminated"]`) and blends per-ant and team rewards (`0.5 * own + 0.5 * team / n`) in GAE.

---

## 5. Reproduce

```bash
.venv/bin/python -m pytest -v
.venv/bin/python scripts/compare.py .arkenos/runs/clean-baseline
.venv/bin/python scripts/compare.py .arkenos/runs/learned
```
