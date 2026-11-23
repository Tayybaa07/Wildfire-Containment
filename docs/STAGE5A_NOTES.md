# Stage 5a: Robustness Experiments (wrong model and wind shift)

**Author:** Zainab Parveen · **Stage:** 5a

## 1. Question
How much does each method lose when the model it plans with is wrong?
The fire always follows the **true** environment. The agent plans with a **believed** one.

## 2. Design
Every agent plans with the environment object it is handed (`agent.select_action(state, env, rng)`),
so a wrong model is just `planner_env != world_env`. No existing file was changed.

- `wildfire/robustness.py`: `run_mismatch_episode`, `run_mismatch_scenario`, `paired_delta`
  (same `TrialResult` / `ScenarioSummary` records as Stage 4).
- `experiments/robustness.py`: the experiment runner (CLI).
- `tests/test_robustness.py`: 12 tests.

Rules: believed, true and shifted environments must share grid size and `k`. Vegetation, wind,
spread probabilities and lambda may differ. Seeds follow the Stage 4 protocol (common random
numbers): same seed gives the same ignition cell and environment stream for every agent.

Note: the example in `STAGE4_NOTES.md` (section 4) builds the true environment only, so it does not
make the agent plan with a different model. Use `run_mismatch_scenario` instead.

## 3. Conditions
Each scenario group (`3x3_mixed_E`, `5x5_mixed_E`) is run under:

| Condition | Meaning | Reference for the paired change |
|---|---|---|
| `correct` | believed = true (wind E) | none |
| `belief_wind_N` | agent thinks wind blows N | `correct` |
| `belief_wind_W` | agent thinks wind blows W (opposite) | `correct` |
| `belief_wind_none` | agent thinks there is no wind | `correct` |
| `belief_spread_x0.7` | agent underestimates every spread probability | `correct` |
| `belief_spread_x1.3` | agent overestimates every spread probability | `correct` |
| `shift_aware` | true wind shifts E to N (turn 1 on 3x3, turn 2 on 5x5); the agent is told | none |
| `shift_unaware` | same world, the agent keeps believing wind E | `shift_aware` |

Agents: NoOp, B0 Random, B1 NearestFront, B2 Greedy, MCTS (N=150 on 3x3, N=100 on 5x5, nearest
rollouts) and the exact solver (3x3 only; it is not tractable on 5x5).

## 4. How to run (from the project root)
```
python experiments/robustness.py            # full run, about 40 seconds
python experiments/robustness.py --quick    # tiny smoke test
python experiments/robustness.py --seeds-small 300 --seeds-large 100 --shift-turn 3
```

## 5. Output (results/stage5a/, not committed)
- `stage5a_robustness_raw.csv/.json` and `stage5a_robustness_summary.csv/.json`: same columns as
  Stage 4. `scenario_id` is `<group>__<condition>`, for example `3x3_mixed_E__belief_wind_W`.
  `peak_memory_mb` is NaN and the optimality-gap columns are empty (not measured here).
- `stage5a_conditions.csv`: what each `scenario_id` means (true wind, believed wind, believed
  `p_veg`, shift target and turn, whether the agent knows about the shift, reference scenario).
- `stage5a_degradation.csv`: `scenario_id, reference_id, agent, n_paired, delta_mean, delta_ci`.
  The change in return is paired by seed. Negative means worse than the reference.

## 6. Results (default run: 3x3 with 100 seeds, 5x5 with 40 seeds)
Mean return per episode (higher is better):

| Condition | NoOp | Random | Nearest | Greedy | MCTS | Exact |
|---|---|---|---|---|---|---|
| 3x3 correct | -4.30 | -3.09 | -1.83 | -1.56 | -1.45 | -1.42 |
| 3x3 wind N | -4.30 | -3.09 | -1.83 | -1.81 | -1.81 | -1.75 |
| 3x3 wind W | -4.30 | -3.09 | -1.83 | -2.15 | -1.91 | -1.98 |
| 3x3 no wind | -4.30 | -3.09 | -1.83 | -1.88 | -1.76 | -1.63 |
| 3x3 spread x0.7 | -4.30 | -3.09 | -1.83 | -1.54 | -1.73 | -1.42 |
| 3x3 spread x1.3 | -4.30 | -3.09 | -1.83 | -1.56 | -1.58 | -1.42 |
| 3x3 shift, aware | -4.50 | -3.05 | -1.83 | -1.63 | -1.49 | -1.43 |
| 3x3 shift, unaware | -4.50 | -3.05 | -1.83 | -1.69 | -1.53 | -1.47 |
| 5x5 correct | -7.88 | -5.83 | -2.21 | -1.74 | -3.46 | n/a |
| 5x5 wind N | -7.88 | -5.83 | -2.21 | -2.44 | -3.84 | n/a |
| 5x5 wind W | -7.88 | -5.83 | -2.21 | -3.24 | -5.57 | n/a |
| 5x5 no wind | -7.88 | -5.83 | -2.21 | -2.12 | -4.38 | n/a |
| 5x5 spread x0.7 | -7.88 | -5.83 | -2.21 | -1.74 | -3.98 | n/a |
| 5x5 spread x1.3 | -7.88 | -5.83 | -2.21 | -1.74 | -2.64 | n/a |
| 5x5 shift, aware | -8.07 | -5.67 | -2.17 | -1.82 | -3.27 | n/a |
| 5x5 shift, unaware | -8.07 | -5.67 | -2.17 | -1.80 | -3.60 | n/a |

Paired change vs `correct` (mean +/- 95% CI):

| Condition | 3x3 Greedy | 3x3 Exact | 3x3 MCTS | 5x5 Greedy | 5x5 MCTS |
|---|---|---|---|---|---|
| wind N | -0.26 +/- 0.20 | -0.33 +/- 0.14 | -0.37 +/- 0.16 | -0.70 +/- 0.46 | -0.38 +/- 0.98 |
| wind W | -0.59 +/- 0.22 | -0.57 +/- 0.17 | -0.46 +/- 0.17 | -1.50 +/- 0.66 | -2.10 +/- 1.11 |
| no wind | -0.32 +/- 0.20 | -0.21 +/- 0.10 | -0.31 +/- 0.14 | -0.38 +/- 0.34 | -0.91 +/- 1.30 |
| spread x0.7 | +0.01 +/- 0.01 | +0.00 +/- 0.00 | -0.28 +/- 0.16 | +0.01 +/- 0.08 | -0.52 +/- 0.58 |
| spread x1.3 | +0.00 +/- 0.00 | +0.00 +/- 0.00 | -0.13 +/- 0.11 | +0.00 +/- 0.00 | +0.82 +/- 0.75 |
| shift, unaware vs aware | -0.06 +/- 0.06 | -0.03 +/- 0.04 | -0.05 +/- 0.06 | +0.02 +/- 0.12 | -0.33 +/- 0.28 |

## 7. What the run shows
1. **Model-free agents cannot be hurt by a wrong belief.** NoOp, Random and Nearest give identical
   results in every belief condition (a unit test checks this). Only agents that use the model
   (Greedy, MCTS, Exact) change.
2. **A wrong wind direction is the most damaging error**, and the opposite direction (W) is the worst.
   On 3x3 the exact solver loses 0.57 +/- 0.17 per episode when it believes W. On 5x5 greedy loses
   1.50 +/- 0.66 and ends at -3.24, clearly worse than the model-free Nearest rule (-2.21).
3. **Spread-probability errors barely change Greedy or the exact policy** in this setup: scaling every
   probability by 0.7 or 1.3 left their decisions almost unchanged. MCTS reacts more, but its 5x5
   intervals are wide, so those differences are not conclusive.
4. **The wind shift mattered little here.** Being unaware costs 0.03 to 0.06 on 3x3, which is within noise.
   Most fires are short, and few cells are affected after the shift.
5. **MCTS (N=100) is worse than Greedy on 5x5 even with the correct model** (-3.46 vs -1.74),
   consistent with the Stage 4 scalability results.

## 8. Limitations
- One vegetation map (`mixed`, seed 0), one true wind (E), `k = 1`, `lambda = 0.2`.
- 5x5 uses 40 seeds and one MCTS budget, so MCTS differences there have wide intervals.
- The exact solver is only available on 3x3.
- Fires are short, so a shift at turn 1 or 2 changes little; later or stronger shifts were not tested.
- "Robustness" here means loss in return when the belief is wrong; the solver is never re-solved
  during an episode except in the `shift_aware` control.
