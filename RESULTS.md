# Results: cooperative transport with IPPO and MAPPO

**Summary.**

- With honest observations, independent action sampling and no hand-coded coordination, IPPO and MAPPO learn the full grid-world task within 1M environment steps. IPPO delivers **11.77 / 12** food per episode (3.91 of 4 pieces carried home by pairs of ants) and MAPPO **11.48 / 12** (3.79 pieces), over 3 seeds and 50 held-out episodes each.
- Most of the gain over the previous commit comes from fixing one bug: the PPO rollout treated timeouts as unfinished episodes.
- Earlier results that relied on a target oracle (`1166e33`) or on hand-coded holder synchronisation (`e7cfa42`) are withdrawn; see §6.

## 1. Setup

- **World (`grid`).** 10×10 grid, 4 ants, 4 crumbs (value 1) and 1 big food item that is cut into 4 pieces (`cut_work = 2`, value 2 each). A piece moves only when **2 holders pull the same way** in the same step. Episodes last 100 steps and the maximum team food per episode is **12**. Reward shaping is on (`shaping = 0.1`).
- **What an ant sees.** Its own 5×5 window (`view_radius = 2`) plus `[carrying, holding, partners/piece_weight, home_x, home_y, t/max_steps]`, where `home` is the path-integrated vector to the nest. Nothing in the observation points to food or names a role. A scripted policy that walks along feature slots 3–4 (where the withdrawn target oracle used to sit) scores 0/12.
- **Learners.** The `ippo` and `mappo` presets: one shared policy with a one-hot ant id, the team reward, and an advantage that blends own and team reward (`0.5·own + 0.5·team/n`). IPPO uses a per-ant critic V(o_i); MAPPO uses a centralised critic V(s). During training each ant samples its action independently. Evaluation is greedy and decentralised.
- **Protocol.** 1,000,448 environment steps per run, with an evaluation every ~100k steps on 50 greedy held-out episodes (`seed = 10000..10049`). 3 training seeds per learner. Layouts vary across evaluation episodes: the big food sits on 39 different cells in the 50 episodes. Artifacts are written to `runs/fix-compare/{honest,all}/grid-<algo>-s<seed>/` (git-ignored; see §8 to regenerate).

## 2. Main result (1M steps)

Values are mean ± SD over the 3 seeds' final evaluations. Pull efficiency = moving pulls / (moving + stalled pulls).

| Learner | Variant | Team food (/12) | Per-seed team food | Crumbs (/4) | Pieces (/4) | Cut work (/8) | Pull eff. | Gini |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `ippo` | **fixes only (default)** | **11.77 ± 0.10** | 11.88, 11.64, 11.78 | 3.95 | **3.91 ± 0.05** | 7.93 | 78.0% | 0.13 |
| `ippo` | + all optional switches | 11.31 ± 0.32 | 10.86, 11.44, 11.62 | 3.84 | 3.73 ± 0.11 | 7.70 | 79.9% | 0.14 |
| `mappo` | **fixes only (default)** | **11.48 ± 0.24** | 11.70, 11.60, 11.14 | 3.89 | **3.79 ± 0.08** | 7.77 | 74.0% | 0.14 |
| `mappo` | + all optional switches | 11.10 ± 0.57 | 10.60, 10.80, 11.90 | 3.77 | 3.67 ± 0.21 | 7.56 | 79.6% | 0.16 |

"All optional switches" means `hold_on_stay=true feel_tug=true` and `agent_id=false` (§4.2). They do not help at this budget, and they are off by default.

Every ant ends up doing roughly a quarter of the work: per-ant delivery shares are 0.244–0.258 for IPPO and 0.228–0.266 for MAPPO.

## 3. Learning curves (mean over 3 seeds)

| Learner | Variant | 100k | 200k | 300k | 400k | 500k | 600k | 700k | 800k | 900k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ippo` | fixes only | 4.37 (0.4) | 8.27 (2.3) | 10.41 (3.3) | 10.77 (3.5) | 11.09 (3.6) | 11.29 (3.7) | 11.54 (3.8) | 11.52 (3.8) | 11.75 (3.9) | **11.77 (3.9)** |
| `ippo` | + optional | 3.49 (0.4) | 6.51 (1.8) | 8.20 (2.5) | 8.68 (2.7) | 9.49 (3.0) | 10.32 (3.3) | 10.36 (3.3) | 11.08 (3.7) | 11.49 (3.8) | 11.31 (3.7) |
| `mappo` | fixes only | 3.76 (0.4) | 7.26 (1.9) | 8.89 (2.7) | 10.13 (3.2) | 11.20 (3.6) | 11.19 (3.7) | 11.43 (3.7) | 11.43 (3.7) | 11.63 (3.8) | **11.48 (3.8)** |
| `mappo` | + optional | 2.69 (0.4) | 5.71 (1.6) | 8.08 (2.6) | 10.07 (3.3) | 10.09 (3.3) | 10.51 (3.5) | 10.51 (3.4) | 10.58 (3.5) | 10.78 (3.6) | 11.10 (3.7) |

Cells show team food (/12), with pieces delivered (/4) in parentheses.

Crumb foraging is learned first. Piece transport takes off between 100k and 300k steps. The previous protocol stopped at 150k, which is too short: at 150k, seeds of the same variant ranged from 0.05 to 2.35 pieces (§5).

## 4. What changed

### 4.1 Fixes in this commit

1. **Timeouts end the episode** (`antrl/train.py`, `train_ppo`). `e7cfa42` stored `done = terminated` and left timeouts out. That made GAE bootstrap through the timeout into the value of the *next* episode's reset state, because the observation had already been replaced by `env.reset()`. Since `t / max_steps` is in every observation and in the global state, a timeout is a true terminal state of a finite-horizon task, so `done = terminated or truncated` is correct. Until a team clears all the food, every episode ends by timeout, and transport happens late in an episode, so this bug hit transport hardest. A regression test covers it (`test_ppo_rollout_treats_timeouts_as_terminal`).
2. **Hand-coded holder synchronisation removed.** All of these were added in `e7cfa42`:
   - `AgentNet._prepare_obs` rewrote a fully held piece to look like "carrying a crumb", and the ant id was blanked while holding. Together these guaranteed that two holders of the same piece produce identical logits, so synchronised pulling at evaluation was built in rather than learned.
   - PPO training shared one Gumbel noise draw among all holders in an environment and masked STAY (let go) for a waiting holder. The mask also made the stored `log π(a)` inconsistent with the action actually sampled.
   - Value-based exploration gave every holder the same random pull.
   - These were detected through observation size (`obs_dim >= 60` / `>= 20`).

   Each ant now samples independently and the network uses the observation as given. Regression tests: `test_ppo_samples_each_ant_independently_even_while_holding`, `test_agent_id_switch`.
3. **Kept: the `e7cfa42` environment shaping changes.** These are honest environment rules:
   - The grab bonus is paid at most once per ant per piece, so WORK/STAY cannot farm it.
   - Carried crumbs and pieces earn shaping only when they reach a new closest distance to the nest.
   - A second ant pressing WORK on a shared cell joins the piece that already has a holder.

### 4.2 Optional switches (all off by default)

| Switch | Effect |
|---|---|
| `--env hold_on_stay=true` | STAY keeps hold of a piece; a 7th action, DROP, lets go |
| `--env feel_tug=true` | 2 extra features: the summed pull directions of the *other* holders of the ant's piece last step (local load sensing) |
| `--set agent_id=false` | the shared network gets no one-hot ant id |

## 5. Ablations at 150k steps (3 seeds, 20 held-out episodes)

These are short runs. At this budget, seeds of one variant spread widely, so treat differences under about 1 piece as noise.

**5a. How much of `e7cfa42`'s result came from hand-coded syncing.** Each part was switched off in a patched copy of `e7cfa42`; the timeout bug is present in all rows.

| Variant | IPPO team | IPPO pieces | MAPPO team | MAPPO pieces |
|---|---:|---:|---:|---:|
| `e7cfa42` as committed | 8.00 | 2.28 | 5.37 | 1.30 |
| shared exploration noise and STAY mask off | 6.88 | 1.73 | 5.00 | 1.15 |
| observation rewrite and id blanking off | 7.47 | 1.95 | 4.18 | 0.37 |
| both off | 4.33 | 0.23 | 3.97 | 0.43 |

**5b. Fix ladder.** Each row starts from "both off" above and adds one change.

| Variant | IPPO team | IPPO pieces | MAPPO team | MAPPO pieces |
|---|---:|---:|---:|---:|
| honest, with the timeout bug | 4.33 | 0.23 | 3.97 | 0.43 |
| **+ timeout fix (= this commit's default)** | **6.20** | **1.30** | **5.83** | **1.20** |
| + `hold_on_stay` | 5.23 | 0.80 | 5.03 | 1.15 |
| + `hold_on_stay`, no agent id | 5.78 | 1.32 | 4.92 | 1.03 |
| + `hold_on_stay`, `feel_tug` | 6.08 | 1.25 | 5.28 | 0.87 |
| + all three | 3.73 | 0.65 | 5.65 | 1.42 |

The timeout fix alone puts honest MAPPO above `e7cfa42`'s hand-coded MAPPO at the same budget (5.83 vs 5.37). The repo code reproduces the "+ timeout fix" row exactly: IPPO s0 gives 7.45 team food and 2.05 pieces, MAPPO s0 gives 4.95 and 0.80.

## 6. Corrections to earlier results

- **`1166e33` ("Improved / ANT-3", 12.00 / 12 for QMIX, IPPO, MAPPO): withdrawn.** That commit's `_target_pos` assigned roles by ant index (ants 0–1 carry, the rest collect crumbs) and put a one-step direction to each ant's target into its observation (`step_dir`). A scripted policy that just follows that direction, with no learning, scores 12/12 in 100 of 100 episodes in both worlds.
- **`e7cfa42` ("Genuine Learned", IPPO 8.00 / MAPPO 5.37 at 150k): superseded.** The oracle was gone, but most of the transport came from hand-coded holder synchronisation (§5a), and the timeout bug was present.
- **Original-code baseline (`a216048`, 150k steps, from `e7cfa42`'s runs), still valid as the starting point:** grid IPPO 2.52 / 12 with 0 pieces, grid MAPPO 3.42 / 12 with 0 pieces, free MAPPO 0.10 / 12 (1 seed).

## 7. Limitations and open items

- **Grid world only.** The free world has not been rerun with these fixes.
- **Value-based learners** (IQL, IQL-team, VDN, QMIX) have not been rerun with these fixes.
- **MAPPO critic mismatch.** MAPPO's single critic V(s) is trained on the mean of per-ant blended returns while each ant's advantage uses its own return. Standard MAPPO (team reward only) or per-ant value heads have not been tried.
- **No equal-budget comparison** with `e7cfa42` at 1M steps, so whether its hand-coded syncing would also reach about 12 at that budget is unknown.
- **Coordination mechanism.** About a quarter of pulls still stall, so coordination is learned and imperfect. One plausible mechanism is a convention: both holders see the same home vector and learn to pull along it. This has not been verified.
- **Seeds.** 3 seeds per configuration; final numbers come from one checkpoint per seed.

## 8. Reproduce

```bash
.venv/bin/python -m pytest -q                                   # 28 tests

# main result: fixes only (defaults), 3 seeds × {ippo, mappo}; each run takes ~6 min on one core
for a in ippo mappo; do for s in 0 1 2; do
  .venv/bin/python -m antrl.train --world grid --algo $a --seed $s --steps 1000000 \
    --eval-every 100000 --eval-episodes 50 --env shaping=0.1 --out runs/fix-compare/honest
done; done

# with every optional switch on
.venv/bin/python -m antrl.train --world grid --algo ippo --seed 0 --steps 1000000 \
  --eval-every 100000 --eval-episodes 50 --env shaping=0.1 hold_on_stay=true feel_tug=true \
  --set agent_id=false --out runs/fix-compare/all

.venv/bin/python scripts/compare.py runs/fix-compare/honest
.venv/bin/python scripts/compare.py runs/fix-compare/all
```

§5a used one-off patches of `e7cfa42` that are not part of this commit. §5b's "+ `hold_on_stay`" and "+ `hold_on_stay`, `feel_tug`" rows map directly onto the switches. The two no-agent-id rows zeroed the id input rather than removing it, so they will not reproduce bit-for-bit.
