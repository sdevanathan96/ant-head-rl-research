# First results (preliminary)

**Setup.** Grid world 10×10, 4 ants, 4 crumbs (value 1) and one big food item
(4 pieces × value 2, `cut_work = 2`, `piece_weight = 2`), 100-step episodes,
view radius 2, light shaping (`shaping = 0.1`). One seed per configuration, no
hyperparameter tuning. Each number is the mean over 20 greedy evaluation episodes
on held-out seeds at the end of training. Maximum team food per episode is 12. A
random policy delivers about 0.1.

| Run | Steps | Team food | Crumbs | Pieces | Cut work | Stalled pulls | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|
| `iql` (stage 1: own reward, blind) | 150k | 1.80 | 1.80 | 0 | 1.20 | 0.85 | 0.50 |
| `iql_team` (shared reward) | 150k | 0.50 | 0.50 | 0 | 0.80 | 0.00 | 0.38 |
| `vdn` | 150k | 0.50 | 0.50 | 0 | 0.60 | 0.00 | 0.20 |
| `qmix` | 150k | 0.40 | 0.40 | 0 | 0.35 | 0.65 | 0.20 |
| `ippo` | 150k | 2.10 | 2.10 | 0 | 4.35 | 10.55 | 0.25 |
| `mappo` | 150k | **3.50** | 3.50 | 0 | 5.70 | 5.25 | 0.33 |
| `mappo` (long) | 2M | 2.80 | 2.80 | 0 | 7.70 | 0.00 | 0.40 |
| `mappo`, free world | 1M | 0.60 | 0.60 | 0 | 6.90 | 0.00 | 0.31 |

## What this shows

1. **Every learner learned to find crumbs and bring them home; none learned to carry
   a piece.** Zero pieces were delivered in every run, including 2M steps of MAPPO.
   The cooperative job is the hard part, as the research predicted for sparse,
   coordination-dependent rewards.
2. **MAPPO learned to cut but not to carry.** It cut about 3.9 pieces free per episode
   (7.7 units of cut work). After 2M steps it no longer pulled at all (0 stalled pulls),
   so the pieces it cut were simply left behind. Earlier in training (150k), MAPPO and
   IPPO did grab pieces and pull them, but the pulls stalled. They never pulled the
   same way at the same time.
3. **MAPPO was the strongest crumb forager** (3.5 of 4 crumbs at 150k). Training it
   15× longer did not help (2.8), so more steps alone won't unlock the cooperative job.
4. **The team-reward value learners (`iql_team`, VDN, QMIX) did worse than own-reward
   IQL at 150k steps.** With one seed and a short budget this is not conclusive. A
   single shared reward is a noisier signal for each ant than its own (the
   credit-assignment problem), and these runs were still early in their exploration
   schedule (ε anneals over 100k steps).
5. **The free-rider metric works.** Stage-1 IQL had the highest inequality
   (Gini 0.50): one ant's share was 0 and another's 0.48.
6. **The free world is harder.** MAPPO reached 0.6 after 1M steps.

## Next steps

- **Curriculum for carrying:** start with `piece_weight = 1`, pieces near the nest and
  no crumbs, then raise the weight to 2 and move the food out.
- **Shaping that rewards progress:** potential-based shaping on a held piece's distance
  to the nest, so pulling in the right direction pays before delivery.
- **Recurrent agents (GRU)** for the partial views, so holders can track what their partner did.
- **Multiple seeds and longer runs** for the value-based learners before comparing methods.

## Reproduce

```bash
for a in iql iql_team vdn qmix ippo mappo; do
  python -m antrl.train --world grid --algo $a --steps 150000 --eval-every 25000 --env shaping=0.1
done
python -m antrl.train --world grid --algo mappo --steps 2000000 --eval-every 200000 --env shaping=0.1
python -m antrl.train --world free --algo mappo --steps 1000000 --eval-every 100000 --env shaping=0.1
python scripts/compare.py runs/
```
