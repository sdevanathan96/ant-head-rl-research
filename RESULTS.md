# Results: cooperative transport across six MARL learners

**Summary.**

- **Grid world, 1M steps.**
  - IPPO and MAPPO learn the full task from honest local observations: **11.75 / 12** and **11.42 / 12** food per episode over 5 seeds, with 3.90 and 3.77 of 4 pieces carried home by pairs of ants.
  - Among the value-based learners, only VDN carries pieces: 5.51 / 12 with 1.64 pieces.
  - IQL and IQL-team forage crumbs only (about 3 / 12).
  - QMIX trains unstably and ends near 0.
- **Free world (continuous space, noisy headings).**
  - IPPO and MAPPO reach 8.49 and 7.91 / 12 at 1M steps, and **10.69 and 10.16** at 3M.
  - Here a lone ant can drag a piece at half speed, so delivered pieces don't prove cooperation. IQL, which cannot see its teammates, scores 7.82.
  - A new holder count shows that IPPO carries 73% of its delivered pieces in pairs, against 47–61% for VDN and IQL.
- **How pairs coordinate on the grid.**
  - The two holders of a piece always see identical observations; only the ant id differs.
  - The shared policy learns to ignore the id while holding and to pull toward the nest, so the holders agree on 93–94% of steps.
  - Most stalled pulls are a lone holder pulling while it waits for a partner, not a disagreement.
- **MAPPO credit.**
  - One value output per ant gives the best MAPPO on the grid (11.75 ± 0.11) and learns faster.
  - Training on the team reward alone is slower on the grid and much worse in the free world (4.14 vs 7.85).
- **Earlier results stay withdrawn.** The results that relied on a target oracle (`1166e33`) or on hand-coded holder synchronisation (`e7cfa42`) are withdrawn (§10).

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
  - During training each ant picks its action independently (ε-greedy or sampling). Evaluation is greedy and decentralised.
- **Protocol.**
  - About 1M environment steps per run (1,000,448 for PPO, 1,000,000 for the value-based learners).
  - An evaluation every ~100k steps on 50 greedy held-out episodes (`seed = 10000..10049`). Layouts vary across evaluation episodes: the big food sits on 39 different cells in the 50 episodes.
  - Seeds 0–4 for IPPO and MAPPO; seeds 0–2 for everything else, including variants.
  - Values are mean ± SD (population) over seeds' final evaluations.
  - Artifacts are written to `runs/fix-compare/<group>/<world>-<algo>-s<seed>/`. That folder is git-ignored; §12 shows how to regenerate it.

## 2. Main results (1M steps)

Pull efficiency = moving pulls / (moving + stalled pulls). On the grid a single puller can never move a piece, so every pull by a lone holder counts as stalled (§6).

### 2.1 Grid world

| Learner | Seeds | Team food (/12) | Per-seed team food | Crumbs (/4) | Pieces (/4) | Cut work (/8) | Pull eff. | Gini |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `ippo` | 5 | **11.75 ± 0.08** | 11.88, 11.64, 11.78, 11.68, 11.78 | 3.96 | **3.90 ± 0.04** | 7.92 | 77% | 0.12 |
| `mappo` | 5 | **11.42 ± 0.39** | 11.70, 11.60, 11.14, 10.82, 11.86 | 3.88 | **3.77 ± 0.14** | 7.74 | 71% | 0.15 |
| `vdn` | 3 | 5.51 ± 1.50 | 5.44, 7.38, 3.72 | 2.23 | 1.64 ± 0.52 | 4.99 | 61% | 0.28 |
| `iql` | 3 | 3.29 ± 0.17 | 3.08, 3.30, 3.50 | 3.29 | 0 | 3.46 | no piece moved | 0.42 |
| `iql_team` | 3 | 2.73 ± 0.22 | 2.74, 3.00, 2.46 | 2.73 | 0 | 0.97 | no piece moved | 0.47 |
| `qmix` | 3 | 0.08 ± 0.10 | 0.22, 0.02, 0.00 | 0.08 | 0 | 0.47 | 7% | 0.06 |
| `ippo` + all optional switches | 3 | 11.31 ± 0.32 | 10.86, 11.44, 11.62 | 3.84 | 3.73 ± 0.11 | 7.70 | 80% | 0.14 |
| `mappo` + all optional switches | 3 | 11.10 ± 0.57 | 10.60, 10.80, 11.90 | 3.77 | 3.67 ± 0.21 | 7.56 | 80% | 0.16 |

- **IPPO and MAPPO.** Every piece on the grid is moved by two pullers, so every delivered piece counts as paired. In seeds 0–2, every ant ends up doing roughly a quarter of the work: per-ant delivery shares are 0.244–0.258 for IPPO and 0.228–0.266 for MAPPO.
- **Optional switches.** "All optional switches" means `hold_on_stay=true feel_tug=true` and `agent_id=false` (§8.3). They do not help at this budget.
- **VDN** carries pieces, but its seeds vary widely (1.08–2.34 pieces) and scores swing between evaluations (§3).
- **IQL and IQL-team** learn to forage crumbs but never move a piece. IQL on its own reward beats IQL on the team reward.
- **QMIX** rises and then collapses. In grid seeds 0 and 1 it peaked at 3.86 (700k steps) and 3.40 (500k), then fell to about 0. Seed 2 never got above 0.16. In seed 0 the TD loss climbed from about 0.001 to 0.14 before the collapse. This looks like a training-stability problem with the default settings, not a measure of what QMIX can do on this task; it has not been diagnosed.

### 2.2 Free world

| Learner | Seeds | Team food (/12) | Per-seed team food | Crumbs (/4) | Pieces (/4) | **Paired pieces** | Cut work (/8) | Gini |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `ippo` | 5 | **8.49 ± 1.04** | 10.24, 7.94, 7.46, 9.10, 7.72 | 3.16 | 2.66 ± 0.47 | **1.94 ± 0.40** | 6.31 | 0.21 |
| `mappo` | 5 | 7.91 ± 0.78 | 8.08, 7.22, 8.24, 6.92, 9.10 | 3.02 | 2.44 ± 0.41 | 1.58 ± 0.40 | 5.94 | 0.28 |
| `iql` | 3 | 7.82 ± 0.88 | 8.14, 6.62, 8.70 | 2.77 | 2.53 ± 0.40 | 1.55 ± 0.32 | 6.25 | 0.26 |
| `vdn` | 3 | 6.84 ± 1.52 | 6.84, 8.70, 4.98 | 1.00 | 2.92 ± 0.39 | 1.37 ± 0.29 | 7.19 | 0.27 |
| `iql_team` | 3 | 3.47 ± 0.49 | 3.38, 4.12, 2.92 | 1.61 | 0.93 ± 0.25 | 0.30 ± 0.03 | 3.67 | 0.45 |
| `qmix` | 3 | 0.37 ± 0.52 | 1.10, 0.00, 0.00 | 0.37 | 0 | 0 | 0.03 | 0.15 |

A paired piece is a delivered piece that 2 or more ants pulled together for at least half of its path (§4).

- **Team food can't rank cooperation here.** IQL is blind to its teammates and has 0 pieces on the grid, yet it matches MAPPO on team food. The paired-piece column, and the grid, are the cooperation measures.
- **VDN specialises.** It delivers the most pieces (2.92) but only 1.00 of 4 crumbs, and fewer than half of its pieces are paired.
- **Pull efficiency is near 100% for every learner here,** since pulls stall only when they cancel. It is omitted for that reason.

## 3. Learning curves (mean over seeds 0–2)

| Learner | World | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ippo` | grid | 4.37 (0.4) | 8.27 (2.3) | 10.41 (3.3) | 10.77 (3.5) | 11.09 (3.6) | 11.29 (3.7) | 11.54 (3.8) | 11.52 (3.8) | 11.75 (3.9) | **11.77 (3.9)** |
| `ippo` + optional | grid | 3.49 (0.4) | 6.51 (1.8) | 8.20 (2.5) | 8.68 (2.7) | 9.49 (3.0) | 10.32 (3.3) | 10.36 (3.3) | 11.08 (3.7) | 11.49 (3.8) | 11.31 (3.7) |
| `mappo` | grid | 3.76 (0.4) | 7.26 (1.9) | 8.89 (2.7) | 10.13 (3.2) | 11.20 (3.6) | 11.19 (3.7) | 11.43 (3.7) | 11.43 (3.7) | 11.63 (3.8) | **11.48 (3.8)** |
| `mappo` + optional | grid | 2.69 (0.4) | 5.71 (1.6) | 8.08 (2.6) | 10.07 (3.3) | 10.09 (3.3) | 10.51 (3.5) | 10.51 (3.4) | 10.58 (3.5) | 10.78 (3.6) | 11.10 (3.7) |
| `vdn` | grid | 0.22 (0.0) | 0.85 (0.1) | 3.45 (0.7) | 4.22 (1.1) | 3.92 (1.0) | 6.61 (1.9) | 5.86 (1.7) | 5.68 (1.6) | 6.03 (1.9) | 5.51 (1.6) |
| `qmix` | grid | 0.11 (0.0) | 0.03 (0.0) | 1.71 (0.0) | 2.19 (0.0) | 2.19 (0.0) | 1.91 (0.1) | 1.31 (0.0) | 1.19 (0.0) | 0.64 (0.0) | 0.08 (0.0) |
| `ippo` | free | 1.18 (0.1) | 2.56 (0.4) | 3.09 (0.7) | 4.05 (1.0) | 5.65 (1.6) | 6.11 (1.8) | 7.15 (2.1) | 8.13 (2.5) | 8.42 (2.7) | **8.55 (2.7)** |
| `mappo` | free | 0.45 (0.0) | 0.63 (0.0) | 1.56 (0.1) | 2.61 (0.5) | 3.71 (0.9) | 4.60 (1.2) | 5.68 (1.6) | 6.77 (1.9) | 8.21 (2.6) | **7.85 (2.4)** |
| `vdn` | free | 0.28 (0.0) | 0.55 (0.1) | 1.48 (0.5) | 4.98 (2.1) | 5.98 (2.6) | 6.43 (2.8) | 6.75 (2.9) | 6.81 (2.9) | 6.59 (2.9) | 6.84 (2.9) |
| `qmix` | free | 0.02 (0.0) | 0.00 (0.0) | 0.04 (0.0) | 0.00 (0.0) | 0.01 (0.0) | 0.01 (0.0) | 0.01 (0.0) | 0.01 (0.0) | 0.09 (0.0) | 0.37 (0.0) |

Each cell shows team food (/12), with pieces delivered (/4) in parentheses.

- **Order of learning.** Crumb foraging is learned first. On the grid, piece transport takes off between 100k and 300k steps for PPO; in the free world it starts later (300k–500k).
- **Exploration timing.** The value-based learners finish annealing ε at 100k steps and start moving pieces after about 300k.
- **Short runs mislead.** The protocol before `e478b7b` stopped at 150k steps, which is too short: at 150k, seeds of the same variant ranged from 0.05 to 2.35 pieces (§9).

## 4. Which pieces are carried in pairs (free world)

Every evaluation now records, for each step a piece moves, whether 1 ant or 2+ ants pulled it, and how far it moved. A delivered piece counts as **paired** when 2+ pullers moved it for at least half of its path (`pieces_paired`); `carry_team` and `carry_solo` are the path lengths. The 36 saved models from seeds 0–2 were re-scored with `scripts/evaluate.py`, and every one reproduced its logged team food exactly.

| Learner | Pieces (/4) | Paired | Paired share | Path moved by 2+ pullers |
|---|---:|---:|---:|---:|
| `ippo` | 2.74 | 2.01 | **73%** | 72% |
| `mappo` | 2.42 | 1.59 | 66% | 63% |
| `iql` | 2.53 | 1.55 | 61% | 52% |
| `vdn` | 2.92 | 1.37 | 47% | 42% |
| `iql_team` | 0.93 | 0.30 | 32% | 28% |

Seeds 0–2, 50 held-out episodes each.

- **IPPO and MAPPO cooperate more** than the value-based learners, but none of them carries only in pairs.
- **IQL still pairs more than half of its pieces.** Its ants share no network and cannot see one another. Two holders on the same piece nonetheless receive identical observations and both tend to pull toward the nest (§6), so some pairing happens without any coordination.
- **On the grid,** a piece cannot move with a single puller, so every delivered piece is paired by construction.

## 5. Longer free-world training (3M steps)

| Learner | 500k | 1M | 1.5M | 2M | 2.5M | 3M | Per-seed at 3M | Paired at 3M |
|---|---|---|---|---|---|---|---|---:|
| `ippo` | 5.65 (1.6) | 8.55 (2.7) | 8.79 (2.8) | 9.03 (2.9) | 10.07 (3.3) | **10.69 ± 0.49 (3.6)** | 11.36, 10.22, 10.50 | 2.84 (80%) |
| `mappo` | 3.71 (0.9) | 7.85 (2.4) | 8.88 (2.8) | 9.57 (3.1) | 10.14 (3.3) | **10.16 ± 0.60 (3.3)** | 10.90, 9.42, 10.16 | 2.53 (76%) |

Seeds 0–2. The runs are deterministic: at 1M steps they match the 1M runs of §2.2 exactly.

- **IPPO** is still rising slowly at 3M steps.
- **MAPPO** has been flat at about 10.2 since 2.5M steps.
- **Pairing grows with training:** the share of paired pieces rises from 73% to 80% for IPPO, and from 66% to 76% for MAPPO.

## 6. How the two holders coordinate

`scripts/coordination.py` replays each saved policy greedily on the 50 evaluation episodes. It records what both holders of a piece choose on each step, and what a lone holder does while it waits. All rows are means over seeds, with ranges across seeds in brackets.

| Learner | World | Co-holding steps / ep | Same observation | Both pull the same way | Pulls toward the nest | Agree when the nest is diagonal | Lone-holder steps / ep | Lone holder pulls |
|---|---|---:|---:|---|---|---|---:|---|
| `ippo` (5 seeds) | grid | 20.5 | 100% | 0.94 [0.89–0.99] | [0.97–1.00] | [0.87–0.99] | 11.6 | [0.73–1.00] |
| `mappo` (5) | grid | 20.3 | 100% | 0.93 [0.78–0.99] | [0.93–1.00] | [0.65–0.98] | 12.9 | [0.97–0.99] |
| `vdn` (3) | grid | 46.2 | 100% | 0.69 [0.64–0.74] | [0.57–0.62] | [0.63–0.82] | 16.2 | [0.52–0.88] |
| `ippo` (5) | free | 15.2 | 100% | 0.75 [0.66–0.82] | [0.76–1.00] | n/a | 11.2 | [0.97–1.00] |
| `mappo` (5) | free | 12.6 | 100% | 0.81 [0.74–0.89] | [0.86–0.99] | n/a | 13.6 | [0.90–1.00] |
| `vdn` (3) | free | 16.3 | 100% | 0.50 [0.47–0.53] | [0.72–0.87] | n/a | 31.4 | [0.93–0.98] |
| `iql` (3) | free | 17.3 | 100% | 0.31 [0.27–0.39] | [0.78–0.87] | n/a | 18.9 | [0.87–0.97] |

What this shows:

1. **The holders see the same thing.** Both holders stand on the piece's cell, so their observations are identical on every co-holding step. A shared network can tell them apart only by the one-hot ant id it appends.
2. **IPPO and MAPPO learn to ignore the id while holding and to pull toward the nest.** They then agree on 93–94% of grid steps.
   - No simple fixed rule explains the axis choice. "Pull along the larger component of the home vector" matches only 51–83% of diagonal pulls.
   - The x-vs-y choice is similar across ant ids in most runs.
   - The exception is the weakest grid run, MAPPO seed 3 (10.82 / 12). Its x-share by ant id is 0.76, 0.65, 0.42 and 0.89: the id picks the axis, the holders agree on only 65% of diagonal steps, and the piece stalls more.
3. **Most stalled pulls on the grid are waiting, not disagreement.** A holder whose partner has not arrived yet usually pulls anyway (73–100% of such steps), and on the grid that always stalls. For IPPO, about 10 of the ~11 stalled pulls per episode come from lone holders and about 1 from split pulls. The pull-efficiency column in §2.1 therefore understates how well co-holders agree.
4. **Separate networks or weaker policies agree less.**
   - VDN holders agree on 69% of grid steps and pull toward the nest only 57–62% of the time. Its pieces wander: holders spend 46 steps per episode co-holding, against 20 for PPO.
   - In the free world, IQL holders share no parameters and agree only 31% of the time.
   - In the free world split pulls are cheaper, since perpendicular pulls still move the piece at 0.71 of full speed. That is why agreement there is lower even for PPO.

**Caveat:** this is greedy evaluation. During training each ant samples independently, so the holders agree less often than this, and agreement during training was not measured.

## 7. MAPPO credit variants

MAPPO's default credit gives each ant an advantage from its own blended return (`0.5·own_i + 0.5·team/n`), but its single critic V(s) is fitted to the mean of those returns. There are two opt-in alternatives:
- `--set credit=team`: standard MAPPO. Every ant gets the same advantage from the team reward, scaled by 1/n so the critic's loss scale matches.
- `--set critic_per_ant=true`: V(s) has one output per ant, each fitted to that ant's own blended return. This removes the mismatch.

| Variant | Grid team food | Grid pieces | Grid at 200k / 500k | Free team food | Free pieces | Free paired |
|---|---:|---:|---|---:|---:|---:|
| blend, one V(s) (default) | 11.48 ± 0.24 | 3.79 | 7.26 / 11.20 | **7.85 ± 0.45** | 2.42 | 1.59 |
| team reward only | 11.33 ± 0.23 | 3.69 | 3.23 / 6.56 | 4.14 ± 0.53 | 1.22 | 0.75 |
| blend, one V(s) output per ant | **11.75 ± 0.11** | **3.89** | **9.03 / 11.48** | 7.46 ± 0.54 | 2.09 | 1.41 |

Seeds 0–2, 1M steps. Per-seed team food:
- team reward only: grid 11.48, 11.00, 11.50; free 4.32, 3.42, 4.68.
- per-ant outputs: grid 11.62, 11.88, 11.74; free 7.82, 6.70, 7.86.

- **Each ant's own reward matters.** With the team reward alone, MAPPO learns the grid much more slowly (6.56 vs 11.20 at 500k), though it catches up by 1M. In the free world it ends at about half the default's score.
- **Per-ant value outputs help on the grid.** They learn fastest there and finish level with IPPO (11.75). In the free world they show no gain at 1M steps (7.46 vs 7.85, within the seed spread).

## 8. What changed

### 8.1 Fixes from `e478b7b` (kept)

1. **Timeouts end the episode** (`antrl/train.py`, `train_ppo`).
   - `e7cfa42` stored `done = terminated` and left timeouts out. GAE then bootstrapped through the timeout into the value of the *next* episode's reset state, because the observation had already been replaced by `env.reset()`.
   - Since `t / max_steps` is in every observation and in the global state, a timeout is a true terminal state of a finite-horizon task, and `done = terminated or truncated` is correct.
   - Until a team clears all the food, every episode ends by timeout, and transport happens late in an episode, so this bug hit transport hardest.
2. **Hand-coded holder synchronisation removed.** All of these were added in `e7cfa42`:
   - The observation rewrite of a fully held piece as "carrying a crumb", and the blanked ant id while holding.
   - Shared Gumbel noise among holders and a STAY mask during PPO training. The mask also made the stored `log π(a)` inconsistent with the action actually sampled.
   - Shared random pulls in value-based exploration.

   Each ant now samples independently, and the network uses the observation as given.
3. **Kept: the `e7cfa42` environment shaping changes.**
   - The grab bonus is paid at most once per ant per piece.
   - Carried crumbs and pieces earn shaping only when they reach a new closest distance to the nest.
   - A second ant pressing WORK on a shared cell joins the piece that already has a holder.

### 8.2 This update

1. **Value-based learners also treat a timeout as terminal** (`train_value`). They used to store `terminal = terminated`, which bootstrapped into the final observation (t / max_steps = 1), an observation the network is never trained on. Unlike PPO, they stored the true next observation, so the reset-state bug never applied to them. Test: `test_value_replay_treats_timeouts_as_terminal`.
2. **Holder metrics in every evaluation:** `pieces_paired`, `carry_team`, `carry_solo` (§4). Tests: `test_carry_split_counts_team_and_solo_pulls`, `test_pieces_paired_separates_pair_carries_from_solo_drags`.
3. **MAPPO credit options:** `credit` and `critic_per_ant` (§7).
   - The default update is unchanged bit for bit; this was checked against the previous code on random rollouts for both critics.
   - Tests: `test_team_credit_ignores_how_the_team_reward_is_split`, `test_per_ant_critic_heads_fit_each_ants_own_return`.
4. **Scripts:**
   - `scripts/evaluate.py` re-scores saved runs with the current metrics.
   - `scripts/coordination.py` produces the analysis in §6.
   - `scripts/compare.py` now reports paired pieces.

### 8.3 Optional switches (all off by default)

| Switch | Effect |
|---|---|
| `--env hold_on_stay=true` | STAY keeps hold of a piece; a 7th action, DROP, lets go |
| `--env feel_tug=true` | 2 extra features: the summed pull directions of the *other* holders of the ant's piece last step (local load sensing) |
| `--set agent_id=false` | the shared network gets no one-hot ant id |
| `--set credit=team` | PPO advantage from the team reward only (§7) |
| `--set critic_per_ant=true` | MAPPO critic with one output per ant (§7) |

## 9. Ablations at 150k steps (from `e478b7b`; 3 seeds, 20 held-out episodes)

These are short runs. At this budget, seeds of one variant spread widely, so treat differences under about 1 piece as noise.

**9a. How much of `e7cfa42`'s result came from hand-coded syncing.** Each part was switched off in a patched copy of `e7cfa42`; the timeout bug is present in all rows.

| Variant | IPPO team | IPPO pieces | MAPPO team | MAPPO pieces |
|---|---:|---:|---:|---:|
| `e7cfa42` as committed | 8.00 | 2.28 | 5.37 | 1.30 |
| shared exploration noise and STAY mask off | 6.88 | 1.73 | 5.00 | 1.15 |
| observation rewrite and id blanking off | 7.47 | 1.95 | 4.18 | 0.37 |
| both off | 4.33 | 0.23 | 3.97 | 0.43 |

**9b. Fix ladder.** Each row starts from "both off" above and adds one change.

| Variant | IPPO team | IPPO pieces | MAPPO team | MAPPO pieces |
|---|---:|---:|---:|---:|
| honest, with the timeout bug | 4.33 | 0.23 | 3.97 | 0.43 |
| **+ timeout fix (= the default since `e478b7b`)** | **6.20** | **1.30** | **5.83** | **1.20** |
| + `hold_on_stay` | 5.23 | 0.80 | 5.03 | 1.15 |
| + `hold_on_stay`, no agent id | 5.78 | 1.32 | 4.92 | 1.03 |
| + `hold_on_stay`, `feel_tug` | 6.08 | 1.25 | 5.28 | 0.87 |
| + all three | 3.73 | 0.65 | 5.65 | 1.42 |

The timeout fix alone puts honest MAPPO above `e7cfa42`'s hand-coded MAPPO at the same budget (5.83 vs 5.37).

## 10. Corrections to earlier results

- **`1166e33` ("Improved / ANT-3", 12.00 / 12 for QMIX, IPPO and MAPPO): withdrawn.** That commit's `_target_pos` assigned roles by ant index (ants 0–1 carry, the rest collect crumbs) and put a one-step direction to each ant's target into its observation (`step_dir`). A scripted policy that just follows that direction, with no learning, scores 12/12 in 100 of 100 episodes in both worlds.
- **`e7cfa42` ("Genuine Learned", IPPO 8.00 / MAPPO 5.37 at 150k): superseded.** The oracle was gone, but most of the transport came from hand-coded holder synchronisation (§9a), and the timeout bug was present.
- **Original-code baseline (`a216048`, 150k steps), still valid as the starting point:** grid IPPO 2.52 / 12 with 0 pieces, grid MAPPO 3.42 / 12 with 0 pieces, free MAPPO 0.10 / 12 (1 seed).

## 11. Limitations and open items

- **Free world.**
  - Training has not converged for IPPO at 3M steps.
  - The paired-piece measure is a threshold: at least half the path moved by 2+ pullers.
  - The optional switches and the credit variants were run at 1M steps only.
- **Value-based learners.**
  - They use the default settings: replay of 100k transitions, ε annealed to 0.05 and kept there, and target sync every 500 updates. Nothing was tuned.
  - The QMIX collapse in both worlds has not been diagnosed.
  - Possible reasons the value-based learners lag PPO include persistent ε-random actions (one random STAY drops a piece) and stale partner behaviour in replay. Neither has been tested.
- **Coordination.**
  - The analysis in §6 is greedy evaluation.
  - Agreement during training, where each ant samples independently, was not measured.
  - What makes a policy ignore the ant id while holding, and why MAPPO seed 3 did not, is open.
- **Seeds.**
  - 5 seeds for IPPO and MAPPO at 1M steps; 3 for everything else.
  - Final numbers come from one checkpoint per seed.
  - The credit variants' free-world differences between the blend and per-ant outputs are within the seed spread.
- **No equal-budget comparison** with `e7cfa42` at 1M steps.

## 12. Reproduce

```bash
.venv/bin/python -m pytest -q                                   # 33 tests

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

# 3M-step free world (§5): the honest loop with --steps 3000000 --out runs/fix-compare/long
# MAPPO credit variants (§7): add --set credit=team (out: runs/fix-compare/credit-team)
#                            or --set critic_per_ant=true (out: runs/fix-compare/critic-per-ant)

.venv/bin/python scripts/evaluate.py runs/fix-compare/honest/*-s[012]  # adds pieces_paired to older runs
.venv/bin/python scripts/coordination.py runs/fix-compare/honest/grid-ippo-s*
.venv/bin/python scripts/compare.py runs/fix-compare
```

§9a used one-off patches of `e7cfa42` that are not part of the repo. In §9b, the "+ `hold_on_stay`" and "+ `hold_on_stay`, `feel_tug`" rows map directly onto the switches. The two no-agent-id rows zeroed the id input rather than removing it, so they will not reproduce bit for bit.
