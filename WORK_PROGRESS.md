# Work Progress — Ant Foraging

Dated build record for `projects/learning-environments/ant-foraging/`. Newest entries are at the bottom. The entries for Nov 1–15 were written on Nov 20 from the commit history, when the project moved into this repo.

## 2026-11-01 — Sudhanva Devanathan

- **Work done:**
  - Built the grid world and the free (continuous) world. Both share four jobs: find food, bring it home, cut big food into pieces, and carry pieces together. On the grid a piece moves only when two holders pull the same way; in the free world the pulls are summed as forces.
  - Built six learners: IQL (the stage-1 baseline), IQL on the team reward, VDN, QMIX, IPPO and MAPPO, with per-ant or shared networks.
  - Added the training CLI with presets, held-out evaluation (team food, per-ant share, Gini, stalled vs moving pulls), CSV logs and checkpoints, and tests for the mechanics and algorithms. (`600355b`)
  - Ran first single-seed results. (`db2195c`)
- **Results / findings:**
  - Every learner learned to forage crumbs, but none carried a piece. MAPPO did best (3.50 / 12) and learned to cut, but its holders never pulled in sync.
  - A later commit the same day reported 12 / 12 for QMIX, IPPO and MAPPO (`bfe4eb3`). **Withdrawn:** a target oracle in the observation gave each ant a role and a step direction.
- **Blockers / next steps:** transport. The ideas were a carrying curriculum, progress shaping, recurrent agents and more seeds.

## 2026-11-08 — Sudhanva Devanathan

- **Work done:**
  - Removed the target oracle and the other shortcuts, so observations are honest again. Fixed reward farming in the shaping: the grab bonus is paid once per ant per piece, and progress is credited only at a new closest distance to the nest. (`e5295c0`)
  - Found and fixed the main blocker: PPO bootstrapped through timeouts into the value of the next episode's reset state. The time step is observed, so a timeout now ends the episode. (`518d820`)
  - Removed the hand-coded holder synchronisation added in `e5295c0` (shared noise, a STAY mask, the observation rewrite). Every ant now samples its own action from its own observation.
  - Added opt-in switches, all off by default: `hold_on_stay`, `feel_tug` and `agent_id`. 28 tests pass.
  - Updated the README (`151a140`) and added free-world results (`800141b`).
- **Results / findings:**
  - Grid, 1M steps, 3 seeds, 50 held-out episodes: IPPO 11.77 / 12 (3.91 / 4 pieces), MAPPO 11.48 / 12 (3.79). The two holders learn to pull together from honest local observations.
  - Free world, 1M steps, 3 seeds: IPPO 8.55 / 12, MAPPO 7.85 / 12, both still improving.
  - A lone ant can drag a piece at half speed in the free world, so delivered pieces there don't prove cooperation.
  - The `e5295c0` results relied on the hand-coded synchronisation and are superseded.
- **Blockers / next steps:** run the value-based learners with the same care, find a way to count pieces that were really carried by two ants, and add seeds.

## 2026-11-15 — Sudhanva Devanathan

- **Work done:** (`3c4e2f3`)
  - Value-based learners now treat a timeout as terminal too, as PPO does.
  - New evaluation metrics: `pieces_paired` (delivered pieces pulled by 2+ ants for at least half their path) and the `carry_team` / `carry_solo` path lengths.
  - Opt-in PPO credit options: `credit=team` (the team reward only) and `critic_per_ant` (MAPPO's V(s) with one output per ant). The default update is unchanged bit for bit.
  - `scripts/evaluate.py` re-scores saved runs; `scripts/coordination.py` records what the two holders of a piece do at each step.
  - Trained all six learners at 1M steps in both worlds, IPPO and MAPPO with 5 seeds, free-world IPPO and MAPPO to 3M steps, and both credit options with 3 seeds. 33 tests pass.
- **Results / findings:**
  - Grid: IPPO 11.75 ± 0.08 and MAPPO 11.42 ± 0.39 of 12; VDN 5.51; IQL 3.29 and IQL-team 2.73 (crumbs only); QMIX collapses to 0.08.
  - Free world: IQL, whose ants ignore each other, scores 7.82, close to MAPPO's 7.91, so team food doesn't measure cooperation there. The paired count does: 73% of IPPO's delivered pieces are carried in pairs, against 61% for IQL and 47% for VDN.
  - At 3M steps in the free world, IPPO reaches 10.69 (80% paired) and is still improving; MAPPO reaches 10.16 and stopped improving around 2.5M.
  - Coordination: the two holders always see identical observations. The shared policy learns to ignore the ant id and pull toward the nest, so the holders agree on 93–94% of grid steps. Most stalled pulls come from a lone holder waiting for a partner.
  - Credit: with the team reward alone, MAPPO's free-world score halves (7.85 → 4.14). One value output per ant learns the grid fastest (9.03 at 200k steps, 11.75 at 1M).
- **Blockers / next steps:** diagnose QMIX's collapse, measure agreement during training, train the free world past 3M steps, add seeds for the credit options, and try recurrent agents.

## 2026-11-20 — Sudhanva Devanathan

- **Work done:**
  - Moved the project from `sdevanathan96/ant-head-rl-research` into this folder, with its full history: every file moved under `projects/learning-environments/ant-foraging/`, and messages, authors and dates unchanged.
  - Added the contributor, project brief, problem statement and end goal sections to the README.
  - Committed the evaluation files behind every table in `results/fix-compare/`, plus four headline models.
- **Results / findings:**
  - The imported commits got new IDs, because their file paths changed. Older commit messages still cite the original IDs:

    | Original | In this repo | Date | Commit |
    |---|---|---|---|
    | `a3196e4` | `600355b` | Nov 1 | Add ant foraging worlds and multi-agent RL learners |
    | `a216048` | `db2195c` | Nov 1 | Add first training results |
    | `1166e33` | `bfe4eb3` | Nov 1 | Improve cooperative piece transport (withdrawn) |
    | `e7cfa42` | `e5295c0` | Nov 8 | Remove hardcoded task oracles |
    | `e478b7b` | `518d820` | Nov 8 | Fix PPO timeouts, drop hand-coded holder sync |
    | `283efa7` | `151a140` | Nov 8 | Update README |
    | `8f634e1` | `800141b` | Nov 8 | Add free-world results at 1M steps |
    | `83b3e8b` | `3c4e2f3` | Nov 15 | All six learners at 1M steps, paired-carry metric, coordination analysis, credit options |

  - The 33 tests pass from this folder.
- **Blockers / next steps:** the folder path still has to be registered on the Iacon board.

## 2026-11-21 — Sudhanva Devanathan

- **Work done:**
  - Trained free-world IPPO and MAPPO to 10M steps (3 seeds each).
  - Added seeds 3–4 for both MAPPO credit variants, in both worlds (8 runs).
  - Added `--coord` to `antrl.train`: every evaluation also records what the two holders of a piece do, greedily and with each ant sampling its action as in training. The analysis moved from `scripts/coordination.py` into `antrl/coordination.py` unchanged; the script still reproduces its earlier output byte for byte.
  - Added sampled evaluation: `evaluate(..., sample=True)`, `scripts/evaluate.py --sample` and `scripts/coordination.py --sample`. Greedy evaluation is unchanged.
  - Retrained IPPO and MAPPO, seeds 0–2, both worlds, with `--coord` (12 runs). Re-scored every saved model with sampled actions.
  - Sampling draws on random generators that training also uses, so both are restored afterwards. A new test trains the same tiny runs with and without `--coord` and requires identical models; it fails if the restore is removed. 36 tests pass.
  - Added this week's evaluation files and the 10M seed-0 models in `results/fix-compare/` (94 runs in total).
- **Results / findings:**
  - **Free world, 10M steps:** MAPPO keeps improving until about 8M and levels off at about 11.4 of 12 (11.31 ± 0.27 at 10M; 11.45 averaged over the evaluations after 8M). IPPO levels off earlier, at about 11 (10.91 ± 0.52). Last week's reading that MAPPO was flat from 2.5M was a pause. 80–83% of delivered pieces are paired. The first 3M steps match last week's runs exactly.
  - **Credit variants, 5 seeds:** the per-ant value outputs remain the best MAPPO on the grid (11.72 ± 0.19) but now trail the default in the free world (6.72 against 7.91). Team reward only: 11.10 on the grid, 3.26 in the free world.
  - **Agreement during training:** on the grid it rises with transport (IPPO 0.31 → 0.83 over the first 300k steps). Once transport is learned, sampling costs no agreement on the grid. In the free world it does: 0.68 sampled against 0.75 greedy for IPPO at 1M. The 12 `--coord` runs reproduce last week's runs exactly, saved models included.
  - **Greedy evaluation understates PPO in the free world before training settles:** sampled, IPPO and MAPPO score 9.59 and 9.43 at 1M (greedy 8.49 and 7.91), clearly ahead of IQL (7.59). The value-based learners don't change. Greedy play can deadlock when two holders split. MAPPO grid seed 3, last week's weakest run, scores 10.82 greedy and 11.64 sampled.
- **Blockers / next steps:** QMIX diagnosis and value-based tuning are next. Decide whether greedy or sampled scores should be the main protocol. PR #24 was merged the same evening; this work continues on `ant-head-research`, updated from `main`.

## 2026-11-22 — Sudhanva Devanathan

- **Work done:**
  - Value-based updates now log the TD prediction and target, the gradient norm, the action-value gap, and, for QMIX, the size of its mixing weights and state bias, averaged over each evaluation window. Training is unchanged: tiny runs on the old and new code give identical models. Two new tests; 38 tests pass.
  - Reran QMIX, seeds 0–2, both worlds, with the new logging (6 runs). They match last week's runs at every shared evaluation.
  - Screened five settings on seed 0: lower final ε, 4× slower target sync, half the learning rate, and two combinations. The screen covered IQL, VDN and QMIX on the grid, and QMIX in the free world (18 runs).
  - Confirmed the best settings on seeds 0–2 in both worlds for IQL, IQL-team, VDN and QMIX (24 runs, 20 new). Re-scored them with sampled actions and ran the holder analysis.
  - Added the evaluation files of all 48 runs and four tuned models in `results/fix-compare/` (142 runs in total). No default setting changed.
- **Results / findings:**
  - **Why QMIX collapsed:** with the target network synced every 500 updates, the team value estimate overshoots early (about 40 on the grid at 150k steps, against about 10 for a good policy). VDN recovers, because its estimate is a plain sum of the ants' Q values. QMIX's mixer absorbs the overshoot: its weights and state bias keep growing, the estimate diverges in the free world (up to 4×10⁶), and the grid policies collapse when the mixer jumps. Halving the learning rate alone doesn't help in the free world; syncing every 2,000 updates fixes both worlds.
  - **Tuned settings:** `lr=2.5e-4 target_update=2000` for VDN and QMIX, `lr=2.5e-4` for IQL and IQL-team.
  - **Grid:** QMIX 0.08 → 8.44 ± 1.35 and VDN 5.51 → 8.23 ± 0.14, each carrying about 2.7 pieces. IQL 3.29 → 4.52 (0.58 pieces), IQL-team 2.73 → 3.16. PPO still leads (11.4–11.75, 3.8–3.9 pieces). The tuned VDN and QMIX holders agree on 65–71% of co-holding steps, against 93–94% for PPO.
  - **Free world:** VDN 6.84 → 9.63 ± 0.16, IQL 7.82 → 9.17, IQL-team 3.47 → 5.40, QMIX 0.37 → 5.35 (only one of three seeds carries pieces). Tuned VDN and IQL match PPO's 1M scores, greedy or sampled. Tuned IQL's paired-piece count (2.02) equals IPPO's, so at 1M steps the free world doesn't separate cooperative learners from independent ones; the grid does.
  - PPO was not tuned, so the comparison is now tuned value-based learners against PPO's defaults.
  - Grid value runs slowed to about 140 minutes each while ten ran at once and the Mac's swap filled. The runs are deterministic, so I restarted the five least advanced with at most four grid runs at a time; results are unaffected.
- **Blockers / next steps:** tune PPO with the same care; train tuned QMIX longer, with more seeds; find out why value-based holders agree less; make the free world a sharper test of cooperation.
