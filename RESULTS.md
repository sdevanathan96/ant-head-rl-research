# Results: cooperative transport across six MARL learners

**Summary.** Scores are sampled: each ant samples its action as it did in training (§1, §6.2). Greedy scores are given alongside.

- **Grid world, 1M steps.**
  - IPPO and MAPPO learn the full task from honest local observations: **11.85 / 12** and **11.61 / 12** food per episode over 5 seeds, with 3.94 and 3.82 of 4 pieces carried home by pairs of ants. MAPPO with one value output per ant scores **11.92** (§7).
  - With their default settings, only VDN among the value-based learners carries pieces (6.05 / 12, 1.81 pieces), and QMIX collapses to about 0.
  - Retuned (§8), VDN and QMIX reach **8.55** and **8.44 / 12** with 2.6–2.75 pieces. They still trail PPO, and their holders agree on 63–69% of co-holding steps against 93–98% for PPO.
  - IQL and IQL-team forage mostly crumbs (2.8–4.8 / 12). Tuned IQL moves an occasional piece (0.64 per episode) without seeing its partner.
- **Why QMIX collapsed** (§8.1). With the target network synced every 500 updates, the team value estimate overshoots early. QMIX's state-conditioned mixer absorbs the overshoot, its weights keep growing, and the estimate diverges. Syncing every 2,000 updates fixes it in both worlds.
- **Free world (continuous space, noisy headings).**
  - IPPO and MAPPO reach 9.59 and 9.43 / 12 at 1M steps, and **11.49** and **11.41** at 10M, close to their grid scores (§5).
  - Here a lone ant can drag a piece at half speed, so delivered pieces don't prove cooperation. With default settings PPO is clearly ahead of IQL, which cannot see its teammates (7.59). Tuned, IQL and VDN score 9.45 and 9.48, level with PPO.
  - A holder count shows that IPPO carries 74% of its delivered pieces in pairs, against 41–69% for VDN and IQL, tuned or not. With longer training, 81–85% are paired. Tuned IQL's absolute paired count (2.18) is close to IPPO's (2.29), so at 1M steps only the grid separates cooperative learners from independent ones.
- **How pairs coordinate on the grid.**
  - The two holders of a piece always see identical observations; only the ant id differs.
  - The shared policy learns to ignore the id while holding and to pull toward the nest, so the holders agree on 93–98% of steps.
  - Most stalled pulls are a lone holder pulling while it waits for a partner, not a disagreement.
  - Agreement rises with transport during training. Once transport is learned, sampling actions costs no agreement on the grid; in the free world it does (about 0.7 against 0.8).
- **Why the headline is sampled** (§6.2). Sampling scores each learner the way it acted in training. PPO's policy is random by design, and greedy play can deadlock: two holders that split see nearly the same thing next step and split again. Greedy scores understate PPO by 1.1–1.5 in the free world at 1M steps. The value-based learners, which act greedily 95% of the time anyway, change by at most 0.54.
- **MAPPO credit.**
  - One value output per ant gives the best and most consistent MAPPO on the grid (11.92 ± 0.04 over 5 seeds) and learns fastest, but trails the default in the free world (8.20 vs 9.43).
  - Training on the team reward alone is slower on the grid and much worse in the free world (4.76 vs 9.43).
- **Earlier results stay withdrawn.** The results that relied on a target oracle (`bfe4eb3`) or on hand-coded holder synchronisation (`e5295c0`) are withdrawn (§11).

## 1. Setup

- **World (`grid`).**
  - 10×10 grid, 4 ants, 4 crumbs (value 1), and 1 big food item that is cut into 4 pieces (`cut_work = 2`, value 2 each).
  - A piece moves only when **2 holders pull the same way** in the same step.
  - Episodes last 100 steps, and the maximum team food per episode is **12**. Reward shaping is on (`shaping = 0.1`).
- **World (`free`).**
  - A 10×10 continuous arena with noisy headings (±0.25 rad).
  - Each ant senses the nearest few ants, crumbs, pieces and big food within radius 3, plus its own load, the home vector, its distance home and the time step.
  - A piece moves by the vector sum of its holders' pulls divided by its weight (2). Two ants pulling the same way move it at full speed, one ant drags it at half speed, and opposite pulls cancel.
  - Food, rewards, episode length and shaping are the same as on the grid.
- **What an ant sees.**
  - On the grid: its own 5×5 window (`view_radius = 2`) plus `[carrying, holding, partners/piece_weight, home_x, home_y, t/max_steps]`, where `home` is the path-integrated vector to the nest.
  - Nothing in the observation points to food or names a role.
  - A scripted policy that walks along feature slots 3–4, where the withdrawn target oracle used to sit, scores 0 / 12.
- **Learners.** The six presets in `antrl/train.py`:
  - `iql`: one Q-network per ant, its own reward, blind to teammates.
  - `iql_team`: the same, but on the team reward.
  - `vdn` and `qmix`: a shared Q-network with ant id, the team reward, and a sum or monotonic mixer.
  - `ippo` and `mappo`: a shared policy with ant id. IPPO uses a per-ant critic V(o_i) and MAPPO a centralised critic V(s). Both blend each ant's own reward with the team's in the advantage (`0.5·own + 0.5·team/n`).
  - During training each ant picks its action independently (ε-greedy or sampling). Evaluation is decentralised too.
- **Protocol.**
  - About 1M environment steps per run (1,000,448 for PPO, 1,000,000 for the value-based learners).
  - An evaluation every ~100k steps on 50 held-out episodes (`seed = 10000..10049`), acting greedily: each ant takes its highest-scoring action. These give the learning curves (§3, §5) and the screen (§8.2). Layouts vary across evaluation episodes: the big food sits on 39 different cells in the 50 episodes.
  - **Headline scores are sampled.** Each final model is re-scored on the same 50 episodes with every ant sampling its own action as in training: PPO samples from π, and the value-based learners act ε-greedily at their final ε = 0.05 (`scripts/evaluate.py --sample`). Greedy scores are reported alongside. Until Nov 22 the headline was greedy; §6.2 explains the change.
  - Seeds 0–4 for IPPO, MAPPO and the MAPPO credit variants; seeds 0–2 for everything else, including the optional-switch runs.
  - The value-based learners use their default settings unless a row says **tuned**: `--set lr=2.5e-4` for IQL and IQL-team, `--set lr=2.5e-4 target_update=2000` for VDN and QMIX (§8). The defaults in `ValueConfig` are unchanged.
  - Values are mean ± SD (population) over seeds' final models.
  - Artifacts are written to `runs/fix-compare/<group>/<world>-<algo>-s<seed>/`, which is git-ignored.
  - The evaluation files of all 142 runs (`config.json`, `metrics.csv`, and where computed `eval.json`, `eval_sampled.json`, `coordination.json`, `coordination_sampled.json`) are committed in `results/fix-compare/` with the same layout, along with ten models: `honest/{grid,free}-{ippo,mappo}-s0/model.pt`, `long10m/free-{ippo,mappo}-s0/model.pt` and `tuned/{grid,free}-{vdn,qmix}-s0/model.pt`. §13 shows how to regenerate the rest.

## 2. Main results (1M steps)

All columns are sampled (§1) except the last, which gives team food with greedy actions. Pull efficiency = moving pulls / (moving + stalled pulls). On the grid a single puller can never move a piece, so every pull by a lone holder counts as stalled (§6).

### 2.1 Grid world

| Learner | Seeds | Team food (/12) | Per-seed team food | Crumbs (/4) | Pieces (/4) | Cut work (/8) | Pull eff. | Gini | Greedy team food |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `ippo` | 5 | **11.85 ± 0.13** | 11.92, 11.88, 11.94, 11.60, 11.90 | 3.98 | **3.94 ± 0.06** | 7.97 | 73% | 0.11 | 11.75 ± 0.08 |
| `mappo` | 5 | **11.61 ± 0.13** | 11.42, 11.50, 11.76, 11.64, 11.74 | 3.98 | **3.82 ± 0.06** | 7.92 | 69% | 0.14 | 11.42 ± 0.39 |
| `vdn`, tuned | 3 | 8.55 ± 0.15 | 8.62, 8.34, 8.70 | 3.06 | 2.75 ± 0.13 | 6.88 | 62% | 0.19 | 8.23 ± 0.14 |
| `qmix`, tuned | 3 | 8.44 ± 1.01 | 9.30, 7.02, 9.00 | 3.24 | 2.60 ± 0.44 | 6.81 | 47% | 0.21 | 8.44 ± 1.35 |
| `vdn` | 3 | 6.05 ± 1.50 | 6.34, 7.72, 4.08 | 2.43 | 1.81 ± 0.54 | 5.45 | 57% | 0.28 | 5.51 ± 1.50 |
| `iql`, tuned | 3 | 4.76 ± 0.43 | 4.26, 5.32, 4.70 | 3.48 | 0.64 ± 0.26 | 5.59 | 11% | 0.30 | 4.52 ± 0.29 |
| `iql` | 3 | 3.45 ± 0.17 | 3.22, 3.62, 3.50 | 3.45 | 0 | 3.73 | no piece moved | 0.38 | 3.29 ± 0.17 |
| `iql_team`, tuned | 3 | 3.21 ± 0.19 | 3.22, 3.44, 2.98 | 3.21 | 0 | 3.85 | 1% | 0.39 | 3.16 ± 0.06 |
| `iql_team` | 3 | 2.83 ± 0.20 | 2.74, 3.10, 2.64 | 2.83 | 0 | 1.56 | no piece moved | 0.44 | 2.73 ± 0.22 |
| `qmix` | 3 | 0.13 ± 0.19 | 0.40, 0.00, 0.00 | 0.13 | 0 | 0.55 | 4% | 0.06 | 0.08 ± 0.10 |
| `ippo` + all optional switches | 3 | 11.83 ± 0.07 | 11.90, 11.86, 11.74 | 3.97 | 3.93 ± 0.02 | 7.97 | 73% | 0.13 | 11.31 ± 0.32 |
| `mappo` + all optional switches | 3 | 11.79 ± 0.02 | 11.82, 11.78, 11.78 | 3.98 | 3.91 ± 0.01 | 7.89 | 77% | 0.13 | 11.10 ± 0.57 |

- **IPPO and MAPPO.** Every piece on the grid is moved by two pullers, so every delivered piece counts as paired. In seeds 0–2, every ant ends up doing roughly a quarter of the work: per-ant delivery shares are 0.236–0.265 for IPPO and 0.236–0.271 for MAPPO.
- **Optional switches.** "All optional switches" means `hold_on_stay=true feel_tug=true` and `agent_id=false` (§9.5). Sampled, they match the defaults (11.83 and 11.79 against 11.85 and 11.61). Greedy, they looked worse (11.31 and 11.10), because greedy play costs them more. They do not help at this budget.
- **"Tuned"** rows use a smaller learning rate, plus a 4× slower target sync for VDN and QMIX (§8). Every value-based learner improves with them.
- **VDN** carries pieces with its default settings, but its seeds vary widely (1.10–2.42 pieces) and scores swing between evaluations (§3). Tuned, it carries 2.75 pieces and its seeds agree within 0.4.
- **IQL and IQL-team** forage crumbs. With default settings they never move a piece. IQL on its own reward beats IQL on the team reward. Tuned IQL moves 0.64 pieces per episode, although its ants can't see each other and its holders pull the same way on only 12–14% of co-holding steps (§6).
- **QMIX** with default settings rises and then collapses. In the greedy evaluations during training, grid seeds 0 and 1 peaked at 3.86 (700k steps) and 3.40 (500k), then fell to about 0. Seed 2 never got above 0.16. The cause is the target-network schedule, not the task (§8.1). Tuned, QMIX scores about as well as tuned VDN (8.44 against 8.55), but its seeds spread more (7.02–9.30) and it is still improving steeply at 1M steps (§3).
- **PPO is still ahead on the grid**, by about 3 food and 1.2 pieces per episode over tuned VDN and QMIX. PPO was not tuned.

### 2.2 Free world

| Learner | Seeds | Team food (/12) | Per-seed team food | Crumbs (/4) | Pieces (/4) | **Paired pieces** | Cut work (/8) | Gini | Greedy team food |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `ippo` | 5 | **9.59 ± 0.45** | 10.22, 8.86, 9.46, 9.80, 9.62 | 3.42 | 3.09 ± 0.19 | **2.29 ± 0.21** | 7.33 | 0.20 | 8.49 ± 1.04 |
| `vdn`, tuned | 3 | 9.48 ± 0.27 | 9.86, 9.36, 9.22 | 2.65 | **3.41 ± 0.01** | 1.39 ± 0.21 | 7.50 | 0.23 | 9.63 ± 0.16 |
| `iql`, tuned | 3 | 9.45 ± 0.46 | 9.74, 8.80, 9.80 | 3.13 | 3.16 ± 0.18 | 2.18 ± 0.31 | 7.14 | 0.17 | 9.17 ± 0.59 |
| `mappo` | 5 | 9.43 ± 0.46 | 9.98, 9.82, 9.28, 8.68, 9.38 | 3.43 | 3.00 ± 0.23 | 1.96 ± 0.17 | 7.40 | 0.23 | 7.91 ± 0.78 |
| `iql` | 3 | 7.59 ± 0.60 | 7.98, 6.74, 8.06 | 2.74 | 2.43 ± 0.36 | 1.47 ± 0.21 | 6.15 | 0.25 | 7.82 ± 0.88 |
| `vdn` | 3 | 6.87 ± 1.18 | 6.58, 8.44, 5.60 | 0.98 | 2.95 ± 0.21 | 1.41 ± 0.21 | 7.17 | 0.29 | 6.84 ± 1.52 |
| `qmix`, tuned | 3 | 5.47 ± 2.51 | 3.56, 9.02, 3.82 | 3.19 | 1.14 ± 1.53 | 0.53 ± 0.74 | 3.27 | 0.25 | 5.35 ± 2.49 |
| `iql_team`, tuned | 3 | 5.43 ± 0.57 | 4.62, 5.88, 5.78 | 1.85 | 1.79 ± 0.27 | 0.83 ± 0.19 | 5.96 | 0.40 | 5.40 ± 0.73 |
| `iql_team` | 3 | 3.51 ± 0.53 | 3.26, 4.24, 3.02 | 1.80 | 0.85 ± 0.25 | 0.25 ± 0.04 | 3.59 | 0.47 | 3.47 ± 0.49 |
| `qmix` | 3 | 0.41 ± 0.58 | 1.24, 0.00, 0.00 | 0.41 | 0 | 0 | 0.03 | 0.16 | 0.37 ± 0.52 |

A paired piece is a delivered piece that 2 or more ants pulled together for at least half of its path (§4).

- **Team food can't rank cooperation here.** IQL is blind to its teammates and moves almost no pieces on the grid, yet tuned it scores 9.45, level with PPO. With default settings PPO is clearly ahead (9.43–9.59 against 7.59 for IQL), and the paired-piece column puts IPPO first. Greedy scores had default IQL level with MAPPO (§6.2).
- **Tuned value-based learners close the gap, by solo effort as much as pairing.** Tuned VDN scores 9.48 and tuned IQL 9.45, level with IPPO and MAPPO. But only 41% of VDN's and 69% of IQL's delivered pieces are paired, against 74% for IPPO and 65% for MAPPO. Tuned IQL's absolute paired count (2.18) is close to IPPO's (2.29). At this budget the free world doesn't separate cooperative from independent learners; the grid does (§2.1).
- **VDN specialises with default settings.** It delivers 2.95 pieces but only 0.98 of 4 crumbs, and fewer than half of its pieces are paired. Tuned, it collects 2.65 crumbs as well.
- **Tuned QMIX is stable but inconsistent here.** One seed carries pieces (9.02); the other two only forage crumbs (3.56 and 3.82).
- **Pull efficiency is 90–100% for every learner here,** since pulls stall only when they cancel. It is omitted for that reason.

## 3. Learning curves (mean over seeds 0–2)

| Learner | World | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ippo` | grid | 4.37 (0.4) | 8.27 (2.3) | 10.41 (3.3) | 10.77 (3.5) | 11.09 (3.6) | 11.29 (3.7) | 11.54 (3.8) | 11.52 (3.8) | 11.75 (3.9) | **11.77 (3.9)** |
| `ippo` + optional | grid | 3.49 (0.4) | 6.51 (1.8) | 8.20 (2.5) | 8.68 (2.7) | 9.49 (3.0) | 10.32 (3.3) | 10.36 (3.3) | 11.08 (3.7) | 11.49 (3.8) | 11.31 (3.7) |
| `mappo` | grid | 3.76 (0.4) | 7.26 (1.9) | 8.89 (2.7) | 10.13 (3.2) | 11.20 (3.6) | 11.19 (3.7) | 11.43 (3.7) | 11.43 (3.7) | 11.63 (3.8) | **11.48 (3.8)** |
| `mappo` + optional | grid | 2.69 (0.4) | 5.71 (1.6) | 8.08 (2.6) | 10.07 (3.3) | 10.09 (3.3) | 10.51 (3.5) | 10.51 (3.4) | 10.58 (3.5) | 10.78 (3.6) | 11.10 (3.7) |
| `vdn` | grid | 0.22 (0.0) | 0.85 (0.1) | 3.45 (0.7) | 4.22 (1.1) | 3.92 (1.0) | 6.61 (1.9) | 5.86 (1.7) | 5.68 (1.6) | 6.03 (1.9) | 5.51 (1.6) |
| `vdn`, tuned | grid | 1.66 (0.0) | 2.65 (0.3) | 6.07 (1.5) | 6.61 (1.9) | 7.44 (2.3) | 7.77 (2.4) | 8.56 (2.7) | 9.13 (3.0) | 8.55 (2.8) | 8.23 (2.7) |
| `qmix` | grid | 0.11 (0.0) | 0.03 (0.0) | 1.71 (0.0) | 2.19 (0.0) | 2.19 (0.0) | 1.91 (0.1) | 1.31 (0.0) | 1.19 (0.0) | 0.64 (0.0) | 0.08 (0.0) |
| `qmix`, tuned | grid | 0.00 (0.0) | 0.95 (0.0) | 2.71 (0.0) | 3.60 (0.4) | 3.79 (0.5) | 4.47 (0.7) | 5.03 (0.9) | 5.86 (1.5) | 8.11 (2.5) | 8.44 (2.7) |
| `ippo` | free | 1.18 (0.1) | 2.56 (0.4) | 3.09 (0.7) | 4.05 (1.0) | 5.65 (1.6) | 6.11 (1.8) | 7.15 (2.1) | 8.13 (2.5) | 8.42 (2.7) | **8.55 (2.7)** |
| `mappo` | free | 0.45 (0.0) | 0.63 (0.0) | 1.56 (0.1) | 2.61 (0.5) | 3.71 (0.9) | 4.60 (1.2) | 5.68 (1.6) | 6.77 (1.9) | 8.21 (2.6) | **7.85 (2.4)** |
| `vdn` | free | 0.28 (0.0) | 0.55 (0.1) | 1.48 (0.5) | 4.98 (2.1) | 5.98 (2.6) | 6.43 (2.8) | 6.75 (2.9) | 6.81 (2.9) | 6.59 (2.9) | 6.84 (2.9) |
| `vdn`, tuned | free | 2.21 (0.0) | 2.61 (0.1) | 4.95 (1.2) | 6.75 (2.2) | 7.37 (2.5) | 8.45 (3.0) | 8.91 (3.2) | 8.59 (3.0) | 8.84 (3.1) | 9.63 (3.5) |
| `iql`, tuned | free | 3.96 (0.4) | 6.86 (1.9) | 7.95 (2.4) | 7.88 (2.4) | 8.59 (2.7) | 8.39 (2.7) | 8.85 (2.9) | 8.64 (2.8) | 8.97 (3.0) | 9.17 (3.0) |
| `qmix` | free | 0.02 (0.0) | 0.00 (0.0) | 0.04 (0.0) | 0.00 (0.0) | 0.01 (0.0) | 0.01 (0.0) | 0.01 (0.0) | 0.01 (0.0) | 0.09 (0.0) | 0.37 (0.0) |
| `qmix`, tuned | free | 0.97 (0.0) | 1.17 (0.0) | 1.70 (0.0) | 2.13 (0.1) | 3.10 (0.4) | 4.03 (0.8) | 4.91 (1.0) | 5.35 (1.1) | 5.37 (1.1) | 5.35 (1.1) |

Each cell shows team food (/12), with pieces delivered (/4) in parentheses. These are the greedy evaluations logged during training (§1); the final sampled scores are in §2.

- **Order of learning.** Crumb foraging is learned first. On the grid, piece transport takes off between 100k and 300k steps for PPO; in the free world it starts later (300k–500k).
- **Exploration timing.** The value-based learners finish annealing ε at 100k steps and start moving pieces after about 300k, tuned or not.
- **Tuned value-based curves.** Tuned VDN climbs steadily, with no collapse. Tuned QMIX is slow until about 800k steps on the grid, then jumps (5.86 → 8.11 → 8.44) and hasn't levelled off by 1M. Tuned IQL in the free world learns fastest of all learners early on (6.86 at 200k steps).
- **Short runs mislead.** The protocol before `518d820` stopped at 150k steps, which is too short: at 150k, seeds of the same variant ranged from 0.05 to 2.35 pieces (§10).

## 4. Which pieces are carried in pairs (free world)

Every evaluation now records, for each step a piece moves, whether 1 ant or 2+ ants pulled it, and how far it moved. A delivered piece counts as **paired** when 2+ pullers moved it for at least half of its path (`pieces_paired`); `carry_team` and `carry_solo` are the path lengths. The 36 saved models from seeds 0–2 were re-scored greedily with `scripts/evaluate.py`, and every one reproduced its logged team food exactly. The table uses the sampled re-scoring (`--sample`).

| Learner | Pieces (/4) | Paired | Paired share | Path moved by 2+ pullers | Paired share, greedy |
|---|---:|---:|---:|---:|---:|
| `ippo` | 3.12 | 2.33 | **75%** | 68% | 73% |
| `mappo` | 3.13 | 2.01 | 64% | 63% | 66% |
| `iql` | 2.43 | 1.47 | 60% | 50% | 61% |
| `vdn` | 2.95 | 1.41 | 48% | 41% | 47% |
| `iql_team` | 0.85 | 0.25 | 29% | 28% | 32% |

Seeds 0–2, default settings, 50 held-out episodes each. The tuned learners are in §2.2.

- **IPPO pairs the largest share**, MAPPO only a little more than IQL, and none of them carries only in pairs.
- **IQL still pairs more than half of its pieces.** Its ants share no network and cannot see one another. Two holders on the same piece nonetheless receive identical observations and both tend to pull toward the nest (§6), so some pairing happens without any coordination.
- **On the grid,** a piece cannot move with a single puller, so every delivered piece is paired by construction.

## 5. Longer free-world training (10M steps)

| Learner | 1M | 2M | 3M | 4M | 5M | 6M | 7M | 8M | 9M | 10M | **Sampled at 10M** | Per-seed, sampled | Paired, sampled |
|---|---|---|---|---|---|---|---|---|---|---|---:|---|---:|
| `ippo` | 8.55 (2.7) | 9.03 (2.9) | 10.69 (3.6) | 10.35 (3.4) | 10.92 (3.6) | 10.74 (3.5) | 11.04 (3.6) | 10.67 (3.5) | 10.77 (3.5) | 10.91 ± 0.52 (3.6) | **11.49 ± 0.06 (3.8)** | 11.54, 11.52, 11.40 | 3.22 (85%) |
| `mappo` | 7.85 (2.4) | 9.57 (3.1) | 10.16 (3.3) | 10.49 (3.5) | 10.75 (3.6) | 11.07 (3.7) | 11.22 (3.7) | 11.37 (3.8) | 11.46 (3.8) | 11.31 ± 0.27 (3.8) | **11.41 ± 0.05 (3.8)** | 11.34, 11.44, 11.44 | 3.10 (81%) |

Seeds 0–2, team food (/12) with pieces delivered (/4) in parentheses. The 1M–10M columns are the greedy evaluations logged during training; the last three columns re-score the final models with sampled actions (§1). The runs are deterministic: over their first 3M steps they match the earlier 3M runs (`long/`) exactly, and at 1M the runs of §2.2.

A single greedy evaluation moves by a few tenths, so the plateaus are clearer as the mean of the evaluations after 8M steps: IPPO 10.98 (seeds 11.51, 11.01, 10.42), MAPPO 11.45 (seeds 11.48, 11.39, 11.49).

- **Sampled, the two finish level** (11.49 and 11.41), with their seeds within 0.15 of each other.
- **Greedy, MAPPO keeps improving until about 8M steps**, then levels off at about 11.4 of 12. The earlier reading that MAPPO was flat from 2.5M steps was wrong: it was a pause.
- **Greedy, IPPO levels off earlier, at about 11,** and its seeds stay apart (11.52, 10.96, 10.24 at 10M). Most of that gap is greedy play: seed 2 scores 10.24 greedy and 11.40 sampled.
- **Pairing grows with training.** At 10M, 81–85% of delivered pieces are paired, up from 65–74% at 1M.
- **The free world is slower to learn, not harder in the end.** At 10M steps IPPO and MAPPO (11.49 and 11.41) are close to their grid scores at 1M (11.85 and 11.61).

## 6. How the two holders coordinate

`scripts/coordination.py` replays each saved policy greedily on the 50 evaluation episodes. It records what both holders of a piece choose on each step, and what a lone holder does while it waits. Greedy replay shows each policy's preferred action; §6.1 gives agreement with sampled actions, which the headline scores use. All rows are means over seeds, with ranges across seeds in brackets.

| Learner | World | Co-holding steps / ep | Same observation | Both pull the same way | Pulls toward the nest | Agree when the nest is diagonal | Lone-holder steps / ep | Lone holder pulls |
|---|---|---:|---:|---|---|---|---:|---|
| `ippo` (5 seeds) | grid | 20.5 | 100% | 0.94 [0.89–0.99] | [0.97–1.00] | [0.87–0.99] | 11.6 | [0.73–1.00] |
| `mappo` (5) | grid | 20.3 | 100% | 0.93 [0.78–0.99] | [0.93–1.00] | [0.65–0.98] | 12.9 | [0.97–0.99] |
| `vdn` (3) | grid | 46.2 | 100% | 0.69 [0.64–0.74] | [0.57–0.62] | [0.63–0.82] | 16.2 | [0.52–0.88] |
| `vdn`, tuned (3) | grid | 46.0 | 100% | 0.71 [0.69–0.75] | [0.65–0.68] | [0.74–0.83] | 14.3 | [0.51–0.62] |
| `qmix`, tuned (3) | grid | 27.9 | 100% | 0.65 [0.60–0.68] | [0.77–0.80] | [0.59–0.68] | 22.5 | [0.85–0.92] |
| `iql`, tuned (3) | grid | 22.5 | 100% | 0.12 [0.10–0.14] | [0.57–0.71] | [0.10–0.15] | 23.5 | [0.72–0.86] |
| `ippo` (5) | free | 15.2 | 100% | 0.75 [0.66–0.82] | [0.76–1.00] | n/a | 11.2 | [0.97–1.00] |
| `mappo` (5) | free | 12.6 | 100% | 0.81 [0.74–0.89] | [0.86–0.99] | n/a | 13.6 | [0.90–1.00] |
| `vdn` (3) | free | 16.3 | 100% | 0.50 [0.47–0.53] | [0.72–0.87] | n/a | 31.4 | [0.93–0.98] |
| `vdn`, tuned (3) | free | 21.0 | 100% | 0.51 [0.46–0.59] | [0.75–0.91] | n/a | 35.0 | [0.93–0.97] |
| `iql` (3) | free | 17.3 | 100% | 0.31 [0.27–0.39] | [0.78–0.87] | n/a | 18.9 | [0.87–0.97] |
| `iql`, tuned (3) | free | 26.0 | 100% | 0.28 [0.24–0.31] | [0.74–0.82] | n/a | 16.5 | [0.90–0.94] |

What this shows:

1. **The holders see the same thing.** Both holders stand on the piece's cell, so their observations are identical on every co-holding step. A shared network can tell them apart only by the one-hot ant id it appends.
2. **IPPO and MAPPO learn to ignore the id while holding and to pull toward the nest.** They then agree on 93–94% of grid steps.
   - No simple fixed rule explains the axis choice. "Pull along the larger component of the home vector" matches only 51–83% of diagonal pulls.
   - The x-vs-y choice is similar across ant ids in most runs.
   - The exception is the weakest greedy grid run, MAPPO seed 3 (10.82 / 12 greedy, 11.64 sampled). Its x-share by ant id is 0.76, 0.65, 0.42 and 0.89: the id picks the axis, the holders agree on only 65% of diagonal steps, and the piece stalls more. This turns out to be a property of greedy play, not of the policy (§6.2).
3. **Most stalled pulls on the grid are waiting, not disagreement.** A holder whose partner has not arrived yet usually pulls anyway (73–100% of such steps), and on the grid that always stalls. For IPPO, about 10 of the ~11 stalled pulls per episode come from lone holders and about 1 from split pulls. The pull-efficiency column in §2.1 therefore understates how well co-holders agree.
4. **Separate networks or weaker policies agree less.**
   - VDN holders agree on 69% of grid steps and pull toward the nest only 57–62% of the time. Its pieces wander: holders spend 46 steps per episode co-holding, against 20 for PPO.
   - In the free world, IQL holders share no parameters and agree only 31% of the time.
   - In the free world split pulls are cheaper, since perpendicular pulls still move the piece at 0.71 of full speed. That is why agreement there is lower even for PPO.
5. **Tuning raised the value-based scores, not their holders' agreement.**
   - Tuned VDN and QMIX holders agree on 71% and 65% of grid co-holding steps, about as often as default VDN (69%), and split on 21% and 28%. Tuned VDN's extra pieces come from more cutting (6.53 of 8 cut work, against 4.99) and from pulls that point toward the nest more often (65–68%, against 57–62%), not from agreeing more often.
   - Tuned IQL's holders agree on 12% of grid co-holding steps and split on 69%. The few pieces it moves come from two independent policies happening to pull the same way.
   - Tuned QMIX in the free world is left out: two of its seeds never carry a piece.

### 6.1 Agreement during training

`python -m antrl.train --coord` repeats this analysis at every evaluation, on the same 50 held-out episodes: once greedily (`g_*` columns in `metrics.csv`), and once with each ant sampling its own action as it does in training (`s_*`). The random state is restored afterwards, so the run itself is unchanged. The 12 runs below (IPPO and MAPPO, seeds 0–2, both worlds) reproduce the §2 runs exactly at every evaluation, saved models included.

Share of co-holding steps on which both holders pull the same way, greedy / sampled, mean of seeds 0–2:

| Learner | World | 100k | 200k | 300k | 500k | 700k | 1M |
|---|---|---|---|---|---|---|---|
| `ippo` | grid | 0.22 / 0.31 | 0.67 / 0.67 | 0.74 / 0.83 | 0.78 / 0.90 | 0.87 / 0.92 | 0.93 / 0.91 |
| `mappo` | grid | 0.35 / 0.48 | 0.65 / 0.82 | 0.82 / 0.87 | 0.91 / 0.93 | 0.98 / 0.97 | 0.98 / 0.98 |
| `ippo` | free | – / 0.28 | 0.77 / 0.42 | 0.71 / 0.49 | 0.57 / 0.51 | 0.72 / 0.63 | 0.75 / 0.68 |
| `mappo` | free | – / 0.43 | – / 0.59 | – / 0.55 | 0.75 / 0.63 | 0.81 / 0.69 | 0.81 / 0.76 |

"–": fewer than 3 seeds' greedy policies co-held a piece for at least 1 step per episode.

- **On the grid, agreement rises with transport.** Sampled agreement goes from 0.31 to 0.83 (IPPO) and from 0.48 to 0.87 (MAPPO) over the first 300k steps, while delivered pieces rise from about 0.4 to 2.7–3.3. Early on the holders mostly fail to pull together at all: at 100k, 25% of IPPO's sampled co-holding steps are one holder pulling while the other only holds on, and 27% are neither pulling. By 1M both are 1–2%.
- **On the grid, sampling costs no agreement once transport is learned.** From 300k steps on, sampled agreement equals or exceeds greedy agreement. The caveat in the previous version of this section, that holders agree less during training, does not hold on the grid.
- **In the free world, sampling lowers agreement throughout.** At 1M steps the holders agree on 68% (IPPO) and 76% (MAPPO) of sampled co-holding steps, against 75% and 81% greedily, and pull different ways on 23–30%. The policies are also less settled there: policy entropy at 1M is 0.46–0.79, against 0.18–0.45 on the grid.

Final models, all seeds (`scripts/coordination.py --sample`; value-based learners act ε-greedily at their final ε = 0.05):

| Learner | World | Seeds | Same way, greedy | Same way, sampled |
|---|---|---:|---:|---:|
| `ippo` | grid | 5 | 0.94 | 0.93 |
| `mappo` | grid | 5 | 0.93 | 0.98 |
| `vdn` | grid | 3 | 0.69 | 0.64 |
| `ippo` | free | 5 | 0.75 | 0.68 |
| `mappo` | free | 5 | 0.81 | 0.73 |
| `vdn` | free | 3 | 0.50 | 0.49 |
| `iql` | free | 3 | 0.31 | 0.30 |
| `vdn`, tuned | grid | 3 | 0.71 | 0.69 |
| `qmix`, tuned | grid | 3 | 0.65 | 0.63 |
| `iql`, tuned | grid | 3 | 0.12 | 0.14 |
| `vdn`, tuned | free | 3 | 0.51 | 0.54 |
| `iql`, tuned | free | 3 | 0.28 | 0.30 |
| `ippo`, 3M steps | free | 3 | 0.78 | 0.73 |
| `mappo`, 3M steps | free | 3 | 0.61 | 0.69 |
| `ippo`, 10M steps | free | 3 | 0.75 | 0.75 |
| `mappo`, 10M steps | free | 3 | 0.79 | 0.79 |

### 6.2 Sampled versus greedy actions at evaluation

Since Nov 22 the headline scores in this report are sampled. `scripts/evaluate.py --sample` re-scores a saved model on the same 50 held-out episodes with each ant sampling its action as in training: PPO samples from π, and the value-based learners act ε-greedily at ε = 0.05. Greedy evaluation, where each ant takes its highest-scoring action, is what training logs at every evaluation (§3, §5, §8.2).

| Learner | World | Seeds | Team food, greedy | Team food, sampled | Paired pieces, greedy → sampled |
|---|---|---:|---:|---:|---|
| `ippo` | grid | 5 | 11.75 ± 0.08 | 11.85 ± 0.13 | 3.90 → 3.94 |
| `mappo` | grid | 5 | 11.42 ± 0.39 | 11.61 ± 0.13 | 3.77 → 3.82 |
| `vdn` | grid | 3 | 5.51 ± 1.50 | 6.05 ± 1.50 | 1.64 → 1.81 |
| `iql` | grid | 3 | 3.29 ± 0.17 | 3.45 ± 0.17 | 0 → 0 |
| `ippo` | free | 5 | 8.49 ± 1.04 | 9.59 ± 0.45 | 1.94 → 2.29 |
| `mappo` | free | 5 | 7.91 ± 0.78 | 9.43 ± 0.46 | 1.58 → 1.96 |
| `iql` | free | 3 | 7.82 ± 0.88 | 7.59 ± 0.60 | 1.55 → 1.47 |
| `vdn` | free | 3 | 6.84 ± 1.52 | 6.87 ± 1.18 | 1.37 → 1.41 |
| `vdn`, tuned | grid | 3 | 8.23 ± 0.14 | 8.55 ± 0.15 | 2.65 → 2.75 |
| `qmix`, tuned | grid | 3 | 8.44 ± 1.35 | 8.44 ± 1.01 | 2.66 → 2.60 |
| `iql`, tuned | grid | 3 | 4.52 ± 0.29 | 4.76 ± 0.43 | 0.58 → 0.64 |
| `vdn`, tuned | free | 3 | 9.63 ± 0.16 | 9.48 ± 0.27 | 1.59 → 1.39 |
| `iql`, tuned | free | 3 | 9.17 ± 0.59 | 9.45 ± 0.46 | 2.02 → 2.18 |
| `qmix`, tuned | free | 3 | 5.35 ± 2.49 | 5.47 ± 2.51 | 0.63 → 0.53 |
| `ippo`, 3M steps | free | 3 | 10.69 ± 0.49 | 10.86 ± 0.37 | 2.84 → 2.77 |
| `mappo`, 3M steps | free | 3 | 10.16 ± 0.60 | 10.27 ± 0.19 | 2.53 → 2.41 |
| `ippo`, 10M steps | free | 3 | 10.91 ± 0.52 | 11.49 ± 0.06 | 2.97 → 3.22 |
| `mappo`, 10M steps | free | 3 | 11.31 ± 0.27 | 11.41 ± 0.05 | 3.02 → 3.10 |
| `ippo` + all optional switches | grid | 3 | 11.31 ± 0.32 | 11.83 ± 0.07 | 3.73 → 3.93 |
| `mappo` + all optional switches | grid | 3 | 11.10 ± 0.57 | 11.79 ± 0.02 | 3.67 → 3.91 |

IQL-team and default QMIX barely move (grid 2.73 → 2.83 and 0.08 → 0.13; free 3.47 → 3.51 and 0.37 → 0.41), and neither does tuned IQL-team (grid 3.16 → 3.21, free 5.40 → 5.43).

- **Greedy evaluation underrates PPO where training hasn't settled.** In the free world at 1M steps, sampling raises IPPO from 8.49 to 9.59 and MAPPO from 7.91 to 9.43, and halves the spread across seeds. With more training the average gap shrinks to 0.1–0.2, but a single seed can still lose more to greedy play (IPPO seed 2 at 10M: 10.24 greedy, 11.40 sampled).
- **Greedy play can deadlock.** When the two holders split, the piece stays where it is, so on the next step they see nearly the same thing and a greedy policy tends to repeat the split. MAPPO seed 3 on the grid is the clearest case. Greedy, its holders agree on 78% of co-holding steps (65% when the nest is diagonal) and it scores 10.82. Sampled, they agree on 98%, each ant id's x-share is 0.70–0.76, and it scores 11.64. Sampling breaks the repeat.
- **The value-based learners change little,** tuned or not: by at most 0.54 (default VDN on the grid), and usually by 0.3 or less, since at ε = 0.05 they act greedily 95% of the time.
- **Effect on the comparisons.** Scored by sampling, PPO is clearly ahead of default IQL in the free world (9.43–9.59 against 7.59), where greedy scores had them nearly level. Tuned IQL and VDN reach the same level when sampled (9.45 and 9.48), so in the free world at 1M steps the two families are level either way. The grid ranking doesn't change. The optional switches, which looked slightly worse than the defaults greedily, match them when sampled.
- **Why sampled is the headline.** Sampling scores each learner the way it acted in training. PPO learns a random policy on purpose, and its training improves the return of that policy as it is sampled; the greedy policy is derived from it and was never trained directly. The value-based learners' policy is the greedy one, and at ε = 0.05 sampling changes them little. Greedy scores stay in every table, and training-time evaluations remain greedy (§12).

## 7. MAPPO credit variants

MAPPO's default credit gives each ant an advantage from its own blended return (`0.5·own_i + 0.5·team/n`), but its single critic V(s) is fitted to the mean of those returns. There are two opt-in alternatives:
- `--set credit=team`: standard MAPPO. Every ant gets the same advantage from the team reward, scaled by 1/n so the critic's loss scale matches.
- `--set critic_per_ant=true`: V(s) has one output per ant, each fitted to that ant's own blended return. This removes the mismatch.

| Variant | Grid team food | Grid pieces | Grid at 200k / 500k (greedy) | Free team food | Free pieces | Free paired | Greedy, grid / free |
|---|---:|---:|---|---:|---:|---:|---|
| blend, one V(s) (default) | 11.61 ± 0.13 | 3.82 | 6.70 / 11.13 | **9.43 ± 0.46** | **3.00** | **1.96** | 11.42 / 7.91 |
| team reward only | 11.26 ± 0.23 | 3.65 | 3.23 / 7.69 | 4.76 ± 1.49 | 1.43 | 0.80 | 11.10 / 3.26 |
| blend, one V(s) output per ant | **11.92 ± 0.04** | **3.96** | **8.09 / 11.55** | 8.20 ± 0.74 | 2.42 | 1.67 | 11.72 / 6.72 |

Seeds 0–4, 1M steps (seeds 3–4 of the two alternatives were added on Nov 21), sampled (§1). "Grid at 200k / 500k" is the greedy evaluation during training. Per-seed team food, sampled:
- default: grid 11.42, 11.50, 11.76, 11.64, 11.74; free 9.98, 9.82, 9.28, 8.68, 9.38.
- team reward only: grid 11.40, 10.94, 11.60, 11.18, 11.16; free 5.06, 5.42, 7.04, 3.30, 2.96.
- per-ant outputs: grid 11.88, 11.96, 11.88, 11.96, 11.92; free 8.94, 7.86, 8.04, 7.06, 9.08.

- **Each ant's own reward matters.** With the team reward alone, MAPPO learns the grid much more slowly (7.69 against 11.13 at 500k, greedy) and ends lower (11.26 against 11.61). In greedy replay its lone holders wait about twice as long for a partner (24.6 steps per episode, against 12.9 for the default), which is where its extra stalled pulls come from (24.6 per episode, against 15.5). In the free world it scores 4.76, about half the default, and its two new seeds are its weakest (3.30 and 2.96).
- **Per-ant value outputs help on the grid.** They give the best and most consistent MAPPO there: every seed scores at least 11.88 (11.40 greedy), and they learn fastest (8.09 at 200k against 6.70, greedy).
- **In the free world they trail the default.** With 5 seeds they score 8.20 against 9.43 (6.72 against 7.91 greedy). The seed ranges still overlap (7.06–9.08 against 8.68–9.98). One new seed scored 4.46 greedy and 7.06 sampled.
- **Holder agreement** at 1M (greedy replay) is similar across the variants on the grid (0.90–0.93). In the free world it is 0.69 (team reward only) and 0.71 (per-ant outputs), against 0.81 for the default.

## 8. Value-based learners: why QMIX collapsed, and a retune

Until Nov 21 the value-based learners ran only with their default settings: learning rate 5e-4, target networks synced every 500 updates (1,000 environment steps), and ε annealed to 0.05 over 100k steps. This section diagnoses QMIX's collapse and retunes IQL, VDN and QMIX. PPO was not retuned. The diagnosis (§8.1) and the screen (§8.2) use the greedy evaluations logged during training; the confirmation (§8.3) reports sampled scores like §2.

### 8.1 Why QMIX collapsed

Every value-based update now also logs the following (§9.4):
- the mean TD prediction and target (for VDN and QMIX, Q_tot)
- the gradient norm before clipping
- the gap between each ant's best and worst action value
- for QMIX, the mean size of the mixing weights its hypernetworks produce from the global state (|W1|, |W2|) and the state bias b2.

These are averaged over each evaluation window. QMIX was rerun for seeds 0–2 in both worlds, with an evaluation every 50k steps. The reruns match the §2 runs exactly at every shared evaluation.

| Run | Best score (step) | Final | Peak Q_tot (step) | Ants' own Q, range | \|W1\|, start → max | Max b2 |
|---|---|---:|---|---|---|---:|
| grid s0 | 3.86 (700k) | 0.22 | 40 (150k) | −5.3 to 1.2 | 0.17 → 1.65 | 9.8 |
| grid s1 | 3.48 (450k) | 0.02 | 40 (150k) | −9.5 to 1.1 | 0.18 → 3.27 | 23 |
| grid s2 | 0.16 (100k) | 0.00 | 2.3×10⁵ (950k) | 0.4 to 569 | 0.13 → 0.67 | 50 |
| free s0 | 1.10 (1M) | 1.10 | 3.0×10³ (350k) | −9.0 to 0.8 | 0.60 → 9.1 | 45 |
| free s1 | 0.04 (900k) | 0.00 | 5.2×10³ (600k) | 0.3 to 8.2 | 0.22 → 0.51 | 74 |
| free s2 | 0.10 (150k) | 0.00 | 4.0×10⁶ (600k) | −14 to 11 | 1.54 → 301 | 200 |

Q_tot estimates the discounted team return from replayed states. Tuned VDN and QMIX runs that score 8–9.5 on the grid end with Q_tot between 9.2 and 10.5 (§8.3). Default QMIX settles near 2.5 while its grid policy scores about 3.5.

- **Q_tot overshoots early.** On the grid it reaches about 40 at 150k steps, while the policy scores about 0. VDN on the same target schedule overshoots too: Q_tot reaches 37 at 200k steps in the `eps001` screen run below. VDN recovers, because its Q_tot is a plain sum of the ants' Q values.
- **QMIX's hypernetworks absorb the overshoot and keep growing.** |W1| and b2 rise through training, and on the grid the ants' own Q values drift negative. In the free world Q_tot diverges, to between 10³ and 4×10⁶, and the policy never gets past crumbs. In grid seed 2 the ants' Q values diverge as well.
- **The grid collapses follow jumps in the mixer.** Seed 1 scores 3.18 at 550k steps. By 700k, Q_tot has gone from 2.8 to 13, b2 from 6.4 to 20, and the score is 0.08.
- **The target schedule is the cause, not the learning rate.** Halving the learning rate alone still diverges in the free world: Q_tot reaches 1,032 at 800k steps and the score is 0.06. Syncing the target every 2,000 updates removes the overshoot in both worlds. In the grid screen runs, Q_tot then never exceeds its final value. In the free-world screen runs it peaks at 2.7–3.1, and |W1| stays below 0.8. Across all tuned QMIX runs, |W1| stays below 1.6.
- **A reading that fits the logs, not a proof:** the bootstrapped targets chase the estimate's own early overshoot. VDN's sum has a fixed gain of 1 per ant, but QMIX's mixer can raise its weights and bias to fit the inflated targets, which inflates the next targets in turn. A slower target gives the estimate time to settle between syncs.

### 8.2 Screen (seed 0)

One seed of each setting, 1M steps: on the grid for IQL, VDN and QMIX, and in the free world for QMIX, since it diverged worst there. Each cell gives greedy team food at 1M, with the mean of the last three evaluations (800k–1M) in brackets.

| Setting | IQL, grid | VDN, grid | QMIX, grid | QMIX, free |
|---|---|---|---|---|
| default (the §2 seed-0 runs) | 3.08 (3.21) | 5.44 (5.69) | 0.22 (1.67) | 1.10 (0.45) |
| `eps001`: `eps_end=0.01` | 3.40 (3.40) | 6.34 (6.02) | 3.28 (3.23) | – |
| `tgt2000`: `target_update=2000` | 3.20 (3.09) | 5.64 (6.19) | 6.10 (6.31) | 3.18 (3.03) |
| `eps001-tgt2000`: both | 3.46 (3.35) | **9.18** (7.26) | 6.44 (4.98) | 3.52 (3.35) |
| `lr25`: `lr=2.5e-4` | **4.16 (3.96)** | 7.88 (7.05) | 9.28 (8.99) | 0.06 (0.03) |
| `lr25-tgt2000`: both | – | 8.04 **(8.62)** | **9.52 (9.08)** | 3.52 **(3.57)** |

- `eps001` tests whether ε-random actions limit the value-based learners, since one random STAY drops a piece. On its own it helps little.
- `lr25-tgt2000` was added after `lr25` alone diverged in free-world QMIX. It wasn't run for IQL, whose estimates never overshoot: its mean prediction never exceeds 2.1 in any screen run.
- Settings were chosen by the mean of the last three grid evaluations, which is steadier than the final one. For QMIX the setting also had to stay stable in the free world. IQL and IQL-team get `lr=2.5e-4`; VDN and QMIX get `lr=2.5e-4 target_update=2000`. IQL-team wasn't screened and uses IQL's setting.

### 8.3 Confirmation: 3 seeds, both worlds

Seeds 0–2 with the chosen settings. Seed 0 is the screen run: on the grid for IQL, VDN and QMIX, and in the free world for QMIX. The other 20 runs are new.

| Learner | Setting | Grid team food, default → tuned | Grid pieces | Free team food, default → tuned | Free pieces, paired / delivered | Tuned, greedy, grid / free |
|---|---|---|---|---|---|---|
| `vdn` | `lr=2.5e-4 target_update=2000` | 6.05 ± 1.50 → **8.55 ± 0.15** | 1.81 → 2.75 | 6.87 ± 1.18 → **9.48 ± 0.27** | 1.41 / 2.95 → 1.39 / 3.41 | 8.23 / 9.63 |
| `qmix` | `lr=2.5e-4 target_update=2000` | 0.13 ± 0.19 → **8.44 ± 1.01** | 0 → 2.60 | 0.41 ± 0.58 → 5.47 ± 2.51 | 0 / 0 → 0.53 / 1.14 | 8.44 / 5.35 |
| `iql` | `lr=2.5e-4` | 3.45 ± 0.17 → 4.76 ± 0.43 | 0 → 0.64 | 7.59 ± 0.60 → 9.45 ± 0.46 | 1.47 / 2.43 → **2.18** / 3.16 | 4.52 / 9.17 |
| `iql_team` | `lr=2.5e-4` | 2.83 ± 0.20 → 3.21 ± 0.19 | 0 → 0 | 3.51 ± 0.53 → 5.43 ± 0.57 | 0.25 / 0.85 → 0.83 / 1.79 | 3.16 / 5.40 |

Sampled (§1), except the last column. Per-seed values are in §2. Learning curves are in §3, and holder agreement in §6.

- **Every value-based learner improves in both worlds.** VDN also becomes consistent: its seed spread shrinks from 1.50 to 0.15 on the grid and from 1.18 to 0.27 in the free world.
- **QMIX goes from collapse to about VDN's grid score** (8.44 against 8.55), but its seeds spread more (7.02–9.30), and its greedy curve is still rising steeply at 1M (§3). In the free world one seed carries pieces (9.02); the other two forage crumbs only (3.56, 3.82).
- **The grid ranking holds.**
  - PPO stays well ahead: 11.61–11.85 food with 3.8–3.9 pieces, against 8.4–8.6 with 2.6–2.75.
  - The holders of tuned VDN and QMIX agree on 69% and 63% of sampled co-holding steps, against 93–98% for PPO (§6.1).
  - IQL, blind to its teammates, now moves some pieces on the grid (0.64 per episode), but its holders agree on only 14% of co-holding steps.
- **In the free world, tuned value-based learners match PPO at 1M steps.** Tuned VDN (9.48) and IQL (9.45) are level with IPPO and MAPPO (9.59, 9.43); greedy, they scored above them. They pair a smaller share of their pieces: 41% for VDN and 69% for IQL, against 74% for IPPO and 65% for MAPPO.
- **Selection bias is small.** The settings were picked on seed 0, which is included above, using greedy scores. Seeds 1–2 alone give grid VDN 8.52, QMIX 8.01 and IQL 5.01 sampled (8.33, 7.90 and 4.70 greedy), and free QMIX 6.42 (6.27 greedy). In the free world only QMIX's seed 0 came from a screen.

## 9. What changed

### 9.1 Fixes from `518d820` (kept)

1. **Timeouts end the episode** (`antrl/train.py`, `train_ppo`).
   - `e5295c0` stored `done = terminated` and left timeouts out. GAE then bootstrapped through the timeout into the value of the *next* episode's reset state, because the observation had already been replaced by `env.reset()`.
   - Since `t / max_steps` is in every observation and in the global state, a timeout is a true terminal state of a finite-horizon task, and `done = terminated or truncated` is correct.
   - Until a team clears all the food, every episode ends by timeout, and transport happens late in an episode, so this bug hit transport hardest.
2. **Hand-coded holder synchronisation removed.** All of these were added in `e5295c0`:
   - The observation rewrite of a fully held piece as "carrying a crumb", and the blanked ant id while holding.
   - Shared Gumbel noise among holders and a STAY mask during PPO training. The mask also made the stored `log π(a)` inconsistent with the action actually sampled.
   - Shared random pulls in value-based exploration.

   Each ant now samples independently, and the network uses the observation as given.
3. **Kept: the `e5295c0` environment shaping changes.**
   - The grab bonus is paid at most once per ant per piece.
   - Carried crumbs and pieces earn shaping only when they reach a new closest distance to the nest.
   - A second ant pressing WORK on a shared cell joins the piece that already has a holder.

### 9.2 Nov 15 update (`3c4e2f3`)

1. **Value-based learners also treat a timeout as terminal** (`train_value`). They used to store `terminal = terminated`, which bootstrapped into the final observation (t / max_steps = 1), an observation the network is never trained on. Unlike PPO, they stored the true next observation, so the reset-state bug never applied to them. Test: `test_value_replay_treats_timeouts_as_terminal`.
2. **Holder metrics in every evaluation:** `pieces_paired`, `carry_team`, `carry_solo` (§4). Tests: `test_carry_split_counts_team_and_solo_pulls`, `test_pieces_paired_separates_pair_carries_from_solo_drags`.
3. **MAPPO credit options:** `credit` and `critic_per_ant` (§7).
   - The default update is unchanged bit for bit; this was checked against the previous code on random rollouts for both critics.
   - Tests: `test_team_credit_ignores_how_the_team_reward_is_split`, `test_per_ant_critic_heads_fit_each_ants_own_return`.
4. **Scripts:**
   - `scripts/evaluate.py` re-scores saved runs with the current metrics.
   - `scripts/coordination.py` produces the analysis in §6.
   - `scripts/compare.py` now reports paired pieces.

### 9.3 Nov 21 update

1. **Holder agreement during training** (`--coord`, §6.1). The analysis from `scripts/coordination.py` moved into `antrl/coordination.py` unchanged; the script still reproduces its earlier `coordination.json` files byte for byte. `python -m antrl.train --coord` runs it at every evaluation, greedy and sampled.
2. **Sampled evaluation** (§6.2): `evaluate(..., sample=True)` in `antrl/train.py`, `scripts/evaluate.py --sample` and `scripts/coordination.py --sample`. Greedy evaluation is unchanged; re-scoring reproduces the committed `eval.json` files byte for byte.
3. **Measuring never changes a run.** Sampling draws on torch's random generator and on the value-based learners' own, so both are restored afterwards. Tests:
   - `test_coord_logging_leaves_training_unchanged` trains the same tiny IPPO and IQL runs with and without `--coord` and requires identical models. It fails when the restore is removed.
   - `test_value_learner_sampling_restores_its_random_state` and `test_sampled_evaluation_restores_random_state_and_greedy_is_unchanged`.
   - 36 tests pass.
4. **New runs:** free-world IPPO and MAPPO to 10M steps (§5), seeds 3–4 of both MAPPO credit variants (§7), and IPPO and MAPPO seeds 0–2 with `--coord` (§6.1).

### 9.4 Nov 22 update

1. **Value-based diagnostics.**
   - `ValueLearner.update()` now also returns `pred_mean`, `target_mean`, `grad_norm` (before clipping) and `q_gap`. For QMIX it adds `mix_w1`, `mix_w2` and `mix_b2`, from `QMixer.diagnostics`.
   - `train_value` logs the mean of each statistic over the evaluation window. The `loss` and `q_mean` columns of earlier runs held the last update's value only.
   - Training is unchanged. Tiny QMIX and IQL runs on the old and new code give identical models, and the six QMIX reruns match the §2 runs at every shared evaluation.
   - Tests: `test_value_update_reports_diagnostics` and `test_qmix_training_logs_window_averaged_diagnostics`. 38 tests pass.
2. **No default changed.** The tuned settings (§8) are passed with `--set`, so every earlier run still reproduces with the defaults.
3. **New runs:** the six QMIX diagnostic reruns (`qmix-diag/`), 18 single-seed screen runs (`screen/<setting>/`), and 20 new tuned runs. With the four seed-0 screen runs reused, that makes 24 runs in `tuned/`.
4. **Sampled scores are the headline.** The summary, §2, §4, §5, §7 and §8.3, and the README, now report sampled scores, with greedy scores alongside. Training-time evaluations, the learning curves and the screen stay greedy.
   - The six optional-switch runs (`all/`) were re-scored greedily and sampled. Greedy reproduces their logged scores exactly.
   - `scripts/compare.py --sample` builds its table from `eval_sampled.json`. Test: `test_compare_reads_greedy_metrics_and_sampled_evals`. 39 tests pass.

### 9.5 Optional switches (all off by default)

| Switch | Effect |
|---|---|
| `--env hold_on_stay=true` | STAY keeps hold of a piece; a 7th action, DROP, lets go |
| `--env feel_tug=true` | 2 extra features: the summed pull directions of the *other* holders of the ant's piece last step (local load sensing) |
| `--set agent_id=false` | the shared network gets no one-hot ant id |
| `--set credit=team` | PPO advantage from the team reward only (§7) |
| `--set critic_per_ant=true` | MAPPO critic with one output per ant (§7) |

## 10. Ablations at 150k steps (from `518d820`; 3 seeds, 20 held-out episodes)

These are short runs. At this budget, seeds of one variant spread widely, so treat differences under about 1 piece as noise.

**10a. How much of `e5295c0`'s result came from hand-coded syncing.** Each part was switched off in a patched copy of `e5295c0`; the timeout bug is present in all rows.

| Variant | IPPO team | IPPO pieces | MAPPO team | MAPPO pieces |
|---|---:|---:|---:|---:|
| `e5295c0` as committed | 8.00 | 2.28 | 5.37 | 1.30 |
| shared exploration noise and STAY mask off | 6.88 | 1.73 | 5.00 | 1.15 |
| observation rewrite and id blanking off | 7.47 | 1.95 | 4.18 | 0.37 |
| both off | 4.33 | 0.23 | 3.97 | 0.43 |

**10b. Fix ladder.** Each row starts from "both off" above and adds one change.

| Variant | IPPO team | IPPO pieces | MAPPO team | MAPPO pieces |
|---|---:|---:|---:|---:|
| honest, with the timeout bug | 4.33 | 0.23 | 3.97 | 0.43 |
| **+ timeout fix (= the default since `518d820`)** | **6.20** | **1.30** | **5.83** | **1.20** |
| + `hold_on_stay` | 5.23 | 0.80 | 5.03 | 1.15 |
| + `hold_on_stay`, no agent id | 5.78 | 1.32 | 4.92 | 1.03 |
| + `hold_on_stay`, `feel_tug` | 6.08 | 1.25 | 5.28 | 0.87 |
| + all three | 3.73 | 0.65 | 5.65 | 1.42 |

The timeout fix alone puts honest MAPPO above `e5295c0`'s hand-coded MAPPO at the same budget (5.83 vs 5.37).

## 11. Corrections to earlier results

- **`bfe4eb3` ("Improved / ANT-3", 12.00 / 12 for QMIX, IPPO and MAPPO): withdrawn.** That commit's `_target_pos` assigned roles by ant index (ants 0–1 carry, the rest collect crumbs) and put a one-step direction to each ant's target into its observation (`step_dir`). A scripted policy that just follows that direction, with no learning, scores 12/12 in 100 of 100 episodes in both worlds.
- **`e5295c0` ("Genuine Learned", IPPO 8.00 / MAPPO 5.37 at 150k): superseded.** The oracle was gone, but most of the transport came from hand-coded holder synchronisation (§10a), and the timeout bug was present.
- **Original-code baseline (`db2195c`, 150k steps), still valid as the starting point:** grid IPPO 2.52 / 12 with 0 pieces, grid MAPPO 3.42 / 12 with 0 pieces, free MAPPO 0.10 / 12 (1 seed).

## 12. Limitations and open items

- **Free world.**
  - Both PPO learners level off by about 8M steps (§5). The optional switches and the credit variants were run at 1M steps only.
  - The paired-piece measure is a threshold: at least half the path moved by 2+ pullers.
- **Value-based learners.**
  - The tuned settings (§8) come from a one-seed screen of five settings at 1M steps, mostly on the grid. Batch size, replay size, ε schedule length and network size were not varied.
  - PPO was not tuned, so §2 compares tuned value-based learners with PPO at its defaults.
  - Tuned QMIX is still improving at 1M steps, and in the free world only one of its three seeds carries pieces. Its ranking against VDN is not settled.
  - Lowering the final ε to 0.01, to test whether random actions drop pieces, helped little on its own (§8.2). Stale partner behaviour in replay, the other suspected limit, is untested.
- **Coordination.**
  - Agreement during training (§6.1) was measured for IPPO and MAPPO, seeds 0–2, at each evaluation on held-out episodes, not in the training rollouts themselves.
  - What makes a policy ignore the ant id while holding is open. MAPPO seed 3's id-dependent axis choice shows up only in greedy play (§6.2).
- **Evaluation.**
  - Headline scores are sampled, from the final model of each run. Training logs only greedy evaluations, so the learning curves (§3, §5), the QMIX diagnosis and the screen (§8.1–§8.2), including the choice of tuned settings, use greedy scores. In the free world at 1M steps those understate PPO by 1.1–1.5 (§6.2).
  - The coordination analysis (§6) replays greedy policies; sampled agreement is in §6.1.
- **Seeds.**
  - 5 seeds for IPPO, MAPPO and the MAPPO credit variants at 1M steps; 3 for everything else.
  - Final numbers come from one checkpoint per seed.
  - In the free world, MAPPO with per-ant value outputs trails the default by about 1.2 with 5 seeds, but the seed ranges overlap.
- **No equal-budget comparison** with `e5295c0` at 1M steps.

## 13. Reproduce

```bash
.venv/bin/python -m pytest -q                                   # 39 tests
.venv/bin/python scripts/compare.py --sample results/fix-compare   # the headline (sampled) tables, from the committed files
.venv/bin/python scripts/compare.py results/fix-compare            # greedy: each run's last training-time evaluation

# main results: 1M steps; ~4–6 min per PPO run, ~16–26 min per value-based run (one core each)
for w in grid free; do
  for a in ippo mappo; do for s in 0 1 2 3 4; do
    .venv/bin/python -m antrl.train --world $w --algo $a --seed $s --steps 1000000 \
      --eval-every 100000 --eval-episodes 50 --env shaping=0.1 --out runs/fix-compare/honest
  done; done
  for a in iql iql_team vdn qmix; do for s in 0 1 2; do
    .venv/bin/python -m antrl.train --world $w --algo $a --seed $s --steps 1000000 \
      --eval-every 100000 --eval-episodes 50 --env shaping=0.1 --out runs/fix-compare/honest
  done; done
done

# with every optional switch on (§2.1)
.venv/bin/python -m antrl.train --world grid --algo ippo --seed 0 --steps 1000000 \
  --eval-every 100000 --eval-episodes 50 --env shaping=0.1 hold_on_stay=true feel_tug=true \
  --set agent_id=false --out runs/fix-compare/all

# 3M- and 10M-step free world (§5): the honest loop for ippo and mappo, seeds 0-2, with
#   --steps 3000000 --out runs/fix-compare/long  or  --steps 10000000 --out runs/fix-compare/long10m
# MAPPO credit variants (§7), seeds 0-4: add --set credit=team (out: runs/fix-compare/credit-team)
#                                         or --set critic_per_ant=true (out: runs/fix-compare/critic-per-ant)
# agreement during training (§6.1): the honest loop for ippo and mappo, seeds 0-2, with
#   --coord --out runs/fix-compare/coord
# QMIX diagnosis (§8.1): the honest loop for qmix, seeds 0-2, with --eval-every 50000 --out runs/fix-compare/qmix-diag
# screen (§8.2): seed 0, e.g. --set lr=2.5e-4 target_update=2000 --out runs/fix-compare/screen/lr25-tgt2000
#   (eps001: eps_end=0.01; tgt2000: target_update=2000; lr25: lr=2.5e-4; free-world QMIX with --eval-every 50000)
# tuned value-based learners (§8.3): the honest loop for the value-based learners, seeds 0-2, with
#   iql, iql_team: --set lr=2.5e-4    vdn, qmix: --set lr=2.5e-4 target_update=2000    --out runs/fix-compare/tuned

.venv/bin/python scripts/evaluate.py runs/fix-compare/honest/*-s[012]  # adds pieces_paired to older runs
.venv/bin/python scripts/coordination.py runs/fix-compare/honest/grid-ippo-s*
.venv/bin/python scripts/coordination.py --sample runs/fix-compare/honest/grid-ippo-s*   # §6.1 final models
.venv/bin/python scripts/evaluate.py --sample runs/fix-compare/{honest,all,long,long10m,credit-team,critic-per-ant,tuned}/*   # headline scores (§2, §6.2)
.venv/bin/python scripts/coordination.py runs/fix-compare/tuned/*                        # §6 tuned rows (and --sample)
.venv/bin/python scripts/compare.py runs/fix-compare
```

§10a used one-off patches of `e5295c0` that are not part of the repo. In §10b, the "+ `hold_on_stay`" and "+ `hold_on_stay`, `feel_tug`" rows map directly onto the switches. The two no-agent-id rows zeroed the id input rather than removing it, so they will not reproduce bit for bit.
