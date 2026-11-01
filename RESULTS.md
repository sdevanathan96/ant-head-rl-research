# Empirical Results: Multi-Agent Collaboration & Food Splitting

## 1. Experimental Setup

- **Worlds:**
  - `grid`: 10×10 discrete grid, 4 ants, 4 crumbs (value `1.0` each) and 1 big food item (`pieces_per_big = 4`, `piece_value = 2.0`, `cut_work = 2`, `piece_weight = 2`), 100-step episodes, `view_radius = 2`, `shaping = 0.1`. Maximum team food value per episode is **12.0** (`4.0` from crumbs + `8.0` from 4 cooperative pieces).
  - `free`: 10.0×10.0 continuous arena, noisy headings (`±0.25 rad`), force-summed transport (`speed = 0.5`), local sensing (`sense_radius = 3.0`), `shaping = 0.1`.
- **Protocol:** 3 seeds (`s0`, `s1`, `s2`) per algorithm × 20 greedy held-out evaluation episodes per seed (`seed = 10000..10019`, 60 evaluation episodes per row). All run artifacts (`config.json`, `metrics.csv`, `model.pt`, `eval_details.json`) are saved in `.arkenos/runs/`.

---

## 2. Baseline (`C0`) vs. Improved (`ANT-3`) Collaboration Performance

| Condition | World | Learner | Seeds | Team Food (`/12`) | Crumbs (`/4`) | Pieces (`/4`) | Cut Work (`/8`) | Piece Conv. Rate | Moving Pulls | Stalled Pulls | Pull Eff. | Cond. Delivery Gini | Work Gini |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (`C0`)** | `grid` | `iql` | 3 | `2.08 ± 0.18` | `2.08` | `0.00 ± 0.00` | `1.18` | `0.0%` | `0.00` | `0.43` | `0.0%` | `0.51` | `0.45` |
| **Baseline (`C0`)** | `grid` | `iql_team` | 3 | `0.50 ± 0.25` | `0.50` | `0.00 ± 0.00` | `0.27` | `0.0%` | `0.00` | `1.43` | `0.0%` | `0.64` | `0.24` |
| **Baseline (`C0`)** | `grid` | `vdn` | 3 | `0.70 ± 0.25` | `0.70` | `0.00 ± 0.00` | `0.53` | `0.0%` | `0.00` | `1.63` | `0.0%` | `0.69` | `0.39` |
| **Baseline (`C0`)** | `grid` | `qmix` | 3 | `1.18 ± 0.58` | `1.18` | `0.00 ± 0.00` | `0.42` | `0.0%` | `0.00` | `2.20` | `0.0%` | `0.56` | `0.34` |
| **Baseline (`C0`)** | `grid` | `ippo` | 3 | `2.25 ± 0.19` | `2.25` | `0.00 ± 0.00` | `2.13` | `0.0%` | `0.00` | `4.12` | `0.0%` | `0.50` | `0.47` |
| **Baseline (`C0`)** | `grid` | `mappo` | 3 | `1.77 ± 0.34` | `1.77` | `0.00 ± 0.00` | `1.62` | `0.0%` | `0.07` | `3.08` | `1.2%` | `0.43` | `0.37` |
| **Improved (`ANT-3`)** | `grid` | `iql` | 3 | **`7.28 ± 1.98`** | `3.85` | **`1.72 ± 0.98`** | `5.60` | **`56.6%`** | `19.87` | `89.32` | `18.9%` | `0.28` | `0.22` |
| **Improved (`ANT-3`)** | `grid` | `iql_team` | 3 | **`3.77 ± 0.15`** | `3.70` | **`0.03 ± 0.05`** | `1.67` | **`2.8%`** | `0.73` | `18.22` | `2.2%` | `0.47` | `0.45` |
| **Improved (`ANT-3`)** | `grid` | `vdn` | 3 | **`11.90 ± 0.08`** | `4.00` | **`3.95 ± 0.04`** | `8.00` | **`98.8%`** | `38.63` | `11.27` | `79.8%` | `0.05` | `0.11` |
| **Improved (`ANT-3`)** | `grid` | `qmix` | 3 | **`12.00 ± 0.00`** | `4.00` | **`4.00 ± 0.00`** | `8.00` | **`100.0%`** | `36.40` | `0.42` | `98.9%` | `0.03` | `0.09` |
| **Improved (`ANT-3`)** | `grid` | `ippo` | 3 | **`12.00 ± 0.00`** | `4.00` | **`4.00 ± 0.00`** | `8.00` | **`100.0%`** | `36.40` | `0.03` | `99.9%` | `0.02` | `0.09` |
| **Improved (`ANT-3`)** | `grid` | `mappo` | 3 | **`12.00 ± 0.00`** | `4.00` | **`4.00 ± 0.00`** | `8.00` | **`100.0%`** | `36.40` | `0.03` | `99.9%` | `0.03` | `0.09` |
| **Improved (`ANT-3`)** | `free` | `mappo` | 3 | **`12.00 ± 0.00`** | `4.00` | **`4.00 ± 0.00`** | `8.00` | **`100.0%`** | `54.27` | `0.00` | `100.0%` | `0.06` | `0.18` |

---

## 3. Food-Splitting & Credit-Distribution Analysis

Food splitting in `antrl` operates along two coupled dimensions:
1. **Physical Food Splitting (`cut_work`, `per_ant_cut`, and Piece Conversion):**
   - Splitting the big food item into 4 transportable pieces requires `8.0` units of `WORK`.
   - Under `ANT-3`, Ants 0 and 1 specialize primarily in cooperative big-food cutting and heavy piece transport (`per_ant_cut ≈ [3.13, 3.10, 1.07, 0.70]` on Grid `mappo`, and `[3.90, 4.10, 0.00, 0.00]` on Free `mappo`), while Ants 2 and 3 clear the solo crumbs first and then assist with remaining cuts and carries.
   - Cut-to-delivery conversion (`pieces / floor(cut_work / 2)`) rises from **`0.0%`** in all baseline learners to **`100.0%`** in `qmix`, `ippo`, and `mappo`, **`98.8%`** in `vdn`, and **`56.6%`** in independent `iql`.
2. **Value & Credit Splitting Across Ants (`per_ant_value`, `per_ant_share`, `gini`, `work_gini`):**
   - Upon piece delivery, `piece_value = 2.0` is split evenly among holders (`1.0` each).
   - In the improved cooperative learners (`vdn`, `qmix`, `ippo`, `mappo`), each ant delivers roughly `3.00 / 12.00` of total colony food value (`per_ant_value ≈ [2.97, 3.03, 3.17, 2.83]`), driving delivery inequality down from `gini = 0.43–0.69` to **`0.02–0.05`** (`work_gini = 0.09–0.11`).
   - Comparing `iql` (`7.28 ± 1.98` team value, `1.72` pieces) vs. `iql_team` (`3.77 ± 0.15` team value, `0.03` pieces) vs. `vdn`/`qmix` (`11.90–12.00` team value, `3.95–4.00` pieces) isolates the role of **value factorisation**: naive shared team reward without a mixer (`iql_team`) suffers severe credit dilution across the 4 separate ant networks, whereas `vdn` and `qmix` decompose team value and add per-ant credit targets to achieve near-optimal coordination.

---

## 4. Reproduce

```bash
.venv/bin/python -m pytest -v
.venv/bin/python scripts/compare.py .arkenos/runs/
.venv/bin/python .arkenos/lib/analyze_ant4.py
```
