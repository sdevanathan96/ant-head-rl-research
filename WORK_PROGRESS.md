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
