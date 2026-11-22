# Wildfire Containment: Experiment Runner, Parameter Sweeps & Scalability (Stage 4 Notes)

**Author:** Tayyba Bashir (CMS ID: 541682)  
**Stage:** 4 (Experiment Runner, Rollout-Budget Sweeps, Scalability)  
**Status:** Completed & Validated

---

## 1. Overview & Experimental Methodology

Stage 4 establishes the reproducible experimental infrastructure for the entire project. It connects Zainab's environment/baselines (Stage 1), Tayyba's exact solver (Stage 2), and Basma's MCTS agent (Stage 3) into an automated evaluation pipeline.

### Common Random Numbers (CRN) Protocol
To ensure rigorous statistical comparisons, every agent evaluated in a given scenario receives the **exact same sequence of integer seeds**:
```python
env_seq, agent_seq = np.random.SeedSequence(seed).spawn(2)
env_rng = np.random.default_rng(env_seq)
agent_rng = np.random.default_rng(agent_seq)
```
- Initial state ignition is identical for every agent on trial $i$.
- Stochastic fire spread transitions receive the identical random stream across agents.
- Agent choices (e.g. random tie-breaking or MCTS rollouts) receive an independent agent stream that does not contaminate environment stochasticity.

### Metrics Logged
For each individual episode:
- **Decision Quality:** Total return ($\sum r_t$), cells saved, cells burnt, firebreaks placed, steps to containment.
- **Optimality Gap:** Exact Bellman suboptimality gap $\text{Gap}(s, a) = V^*(s) - Q^*(s, a) \ge 0.0$ computed at every turn for grids where $R \times C \le 12$ ($3\times 3$ and $3\times 4$), and gracefully marked `None` on larger grids where exact solving is intractable.
- **Temporal Cost:** Wall-clock decision latency in milliseconds per decision (`ms/decision`), total episode duration (`ms`).
- **Spatial Cost:** MCTS tree nodes per decision (`tree_nodes_per_decision`), peak process memory in megabytes via `tracemalloc`.

---

## 2. Experimental Suites

The master runner script ([`run_stage4.py`](file:///Users/hammadbashir/Downloads/Wildfire-Containment%20/run_stage4.py)) executes three structured experimental suites:

### Suite A: Small-Grid Benchmark & Optimality Gap ($3\times 3$ and $3\times 4$)
Evaluates all agents against the theoretical upper bound (`ExactAgent` from Stage 2) across 6 configurations (uniform grass vs. mixed vegetation, no wind vs. East wind) over 300 fixed seeds per agent.

#### Empirical Results (Averaged across 300 trials per scenario)
| Scenario | Agent | Return (95% CI) | Cells Saved | Time/Dec (ms) | Gap to Opt (95% CI) |
|---|---|---|---|---|---|
| **$3\times 3$ grass (None)** | NoOp | -4.94 +/- 0.32 | 3.06 +/- 0.32 | 0.00 | 0.8888 +/- 0.0177 |
| | B0-Random | -2.86 +/- 0.20 | 5.67 +/- 0.18 | 0.02 | 0.5191 +/- 0.0329 |
| | B1-NearestFront | -1.39 +/- 0.12 | 7.01 +/- 0.10 | 0.02 | 0.0000 +/- 0.0000 |
| | B2-Greedy | -1.60 +/- 0.16 | 6.83 +/- 0.13 | 0.02 | 0.0576 +/- 0.0102 |
| | MCTS (N=150) | -1.47 +/- 0.13 | 6.94 +/- 0.11 | 7.25 | 0.0372 +/- 0.0112 |
| | **Exact** | **-1.39 +/- 0.12** | **7.01 +/- 0.10** | **0.00** | **0.0000 +/- 0.0000** |
| **$3\times 3$ mixed (East)** | NoOp | -5.55 +/- 0.29 | 2.45 +/- 0.29 | 0.00 | 0.9765 +/- 0.0196 |
| | B0-Random | -3.17 +/- 0.20 | 5.39 +/- 0.18 | 0.02 | 0.6019 +/- 0.0327 |
| | B1-NearestFront | -1.50 +/- 0.12 | 6.92 +/- 0.10 | 0.02 | 0.1234 +/- 0.0069 |
| | B2-Greedy | -1.39 +/- 0.12 | 7.01 +/- 0.10 | 0.02 | 0.0000 +/- 0.0000 |
| | MCTS (N=150) | -1.48 +/- 0.12 | 6.93 +/- 0.10 | 7.86 | 0.0848 +/- 0.0134 |
| | **Exact** | **-1.39 +/- 0.12** | **7.01 +/- 0.10** | **0.00** | **0.0000 +/- 0.0000** |
| **$3\times 4$ mixed (None)** | NoOp | -4.59 +/- 0.40 | 6.41 +/- 0.40 | 0.00 | 0.8972 +/- 0.0146 |
| | B0-Random | -3.07 +/- 0.25 | 8.50 +/- 0.22 | 0.02 | 0.6315 +/- 0.0336 |
| | B1-NearestFront | -1.26 +/- 0.17 | 10.11 +/- 0.14 | 0.02 | 0.0248 +/- 0.0055 |
| | B2-Greedy | -1.19 +/- 0.15 | 10.17 +/- 0.12 | 0.03 | 0.0094 +/- 0.0036 |
| | MCTS (N=150) | -1.38 +/- 0.15 | 10.01 +/- 0.13 | 9.26 | 0.0931 +/- 0.0149 |
| | **Exact** | **-1.17 +/- 0.14** | **10.19 +/- 0.12** | **0.00** | **0.0000 +/- 0.0000** |
| **$3\times 4$ mixed (East)** | NoOp | -6.93 +/- 0.38 | 4.07 +/- 0.38 | 0.00 | 1.1472 +/- 0.0271 |
| | B0-Random | -4.09 +/- 0.27 | 7.59 +/- 0.24 | 0.02 | 0.7985 +/- 0.0392 |
| | B1-NearestFront | -2.05 +/- 0.21 | 9.46 +/- 0.17 | 0.02 | 0.3179 +/- 0.0084 |
| | B2-Greedy | -1.39 +/- 0.12 | 10.01 +/- 0.10 | 0.03 | 0.0000 +/- 0.0000 |
| | MCTS (N=150) | -1.56 +/- 0.14 | 9.86 +/- 0.12 | 10.44 | 0.0730 +/- 0.0169 |
| | **Exact** | **-1.39 +/- 0.12** | **10.01 +/- 0.10** | **0.00** | **0.0000 +/- 0.0000** |

---

### Suite B: MCTS Rollout-Budget & Policy Sweeps
Sweeps rollout budgets $N \in \{50, 100, 500, 1000, 5000\}$ and compares rollout policies (`random` vs. `nearest` / heuristic) on $3\times 3$ and $5\times 5$ grids.

#### Empirical Results on $3\times 3$ Mixed Vegetation (East Wind, $k=1$)
| Agent | Return (95% CI) | Cells Saved | Time/Dec (ms) | Gap to Opt (95% CI) |
|---|---|---|---|---|
| Exact (Optimal) | -1.23 +/- 0.33 | 7.15 +/- 0.27 | 0.00 | 0.0000 +/- 0.0000 |
| B2-Greedy | -1.23 +/- 0.33 | 7.15 +/- 0.27 | 0.02 | 0.0000 +/- 0.0000 |
| MCTS (N=50, random) | -1.63 +/- 0.38 | 6.80 +/- 0.32 | 2.55 | 0.2505 +/- 0.0747 |
| MCTS (N=50, nearest) | -1.48 +/- 0.37 | 6.92 +/- 0.31 | 2.44 | 0.1706 +/- 0.0632 |
| MCTS (N=100, random) | -1.48 +/- 0.37 | 6.92 +/- 0.31 | 5.64 | 0.1655 +/- 0.0673 |
| MCTS (N=100, nearest) | -1.33 +/- 0.35 | 7.05 +/- 0.30 | 4.98 | 0.0856 +/- 0.0410 |
| MCTS (N=500, random) | -1.40 +/- 0.35 | 7.00 +/- 0.29 | 28.07 | 0.0778 +/- 0.0346 |
| MCTS (N=500, nearest) | -1.31 +/- 0.33 | 7.08 +/- 0.28 | 25.92 | 0.0500 +/- 0.0237 |
| MCTS (N=1000, nearest) | -1.28 +/- 0.35 | 7.10 +/- 0.29 | 53.12 | 0.0725 +/- 0.0237 |
| MCTS (N=5000, nearest) | **-1.25 +/- 0.34** | **7.12 +/- 0.28** | **252.28** | **0.0614 +/- 0.0236** |

#### Key Takeaways for Suite B:
1. **Convergence toward Optimality:** As budget grows from $N=50$ to $N=5000$, MCTS return monotonically converges from $-1.63$ to $-1.25$, cutting the optimality gap by over $75\%$.
2. **Heuristic Rollouts Win:** At every budget level, `nearest` (B1 heuristic) rollouts outperform `random` rollouts in decision quality while running slightly faster (heuristic terminates rollouts in fewer random turns).
3. **Linear Time Complexity:** Decision latency scales linearly with $N$ ($\approx 2.5\text{ms}$ at $N=50$, $\approx 25\text{ms}$ at $N=500$, $\approx 252\text{ms}$ at $N=5000$).

---

### Suite C: Large-Grid Scalability Runs ($5\times 5$, $6\times 6$, $8\times 8$, $10\times 10$)
Scales all agents up to $10\times 10$ ($100$ cells) with both $k=1$ and $k=2$ firebreaks per turn.

#### Empirical Results ($N=30$ seeds per scenario)
| Grid | $k$ | Agent | Return (95% CI) | Cells Saved (95% CI) | Time/Dec (ms) | Peak Mem (MB) |
|---|---|---|---|---|---|---|
| $5\times 5$ | 1 | NoOp | -12.03 +/- 2.52 | 11.97 +/- 2.52 | 0.00 | 0.00 |
| | 1 | B2-Greedy | -1.97 +/- 0.69 | 22.43 +/- 0.60 | 0.05 | 0.01 |
| | 1 | MCTS (N=100) | -2.32 +/- 0.73 | 22.13 +/- 0.65 | 15.97 | 0.40 |
| $5\times 5$ | 2 | B2-Greedy | -0.40 +/- 0.00 | 24.00 +/- 0.00 | 0.45 | 0.01 |
| | 2 | MCTS (N=100) | -3.70 +/- 1.06 | 21.37 +/- 0.86 | 10.97 | 0.38 |
| $6\times 6$ | 1 | NoOp | -13.53 +/- 2.96 | 21.47 +/- 2.96 | 0.00 | 0.00 |
| | 1 | B2-Greedy | -1.28 +/- 0.47 | 34.10 +/- 0.39 | 0.08 | 0.01 |
| | 1 | MCTS (N=100) | -3.35 +/- 1.26 | 32.23 +/- 1.14 | 22.36 | 0.68 |
| $8\times 8$ | 1 | NoOp | -12.10 +/- 4.50 | 50.90 +/- 4.50 | 0.00 | 0.00 |
| | 1 | B2-Greedy | -1.28 +/- 0.47 | 62.10 +/- 0.39 | 0.13 | 0.01 |
| | 1 | MCTS (N=100) | -8.10 +/- 3.15 | 55.87 +/- 2.88 | 41.86 | 1.15 |
| $10\times 10$ | 1 | NoOp | -18.83 +/- 7.70 | 80.17 +/- 7.70 | 0.00 | 0.00 |
| | 1 | B0-Random | -15.06 +/- 6.14 | 85.40 +/- 5.73 | 0.02 | 0.01 |
| | 1 | B1-Nearest | -1.04 +/- 0.39 | 98.30 +/- 0.33 | 0.15 | 0.01 |
| | 1 | B2-Greedy | -1.04 +/- 0.39 | 98.30 +/- 0.33 | 0.20 | 0.01 |
| | 1 | MCTS (N=100) | -9.83 +/- 3.99 | 90.27 +/- 3.69 | 62.26 | 1.84 |
| $10\times 10$ | 2 | NoOp | -18.83 +/- 7.70 | 80.17 +/- 7.70 | 0.00 | 0.00 |
| | 2 | B1-Nearest | -0.40 +/- 0.00 | 99.00 +/- 0.00 | 0.15 | 0.01 |
| | 2 | B2-Greedy | -0.40 +/- 0.00 | 99.00 +/- 0.00 | 8.91 | 0.01 |
| | 2 | MCTS (N=100) | -8.21 +/- 3.62 | 92.67 +/- 3.05 | 101.28 | 2.10 |

---

## 3. Data Schema & File Structure for Stages 5 & 6

All outputs are saved automatically to [`results/stage4/`](file:///Users/hammadbashir/Downloads/Wildfire-Containment%20/results/stage4/):

### Raw Trial Datasets
- `results/stage4/suite_a_small_grid_raw.csv` and `.json`
- `results/stage4/suite_b_budget_sweep_raw.csv` and `.json`
- `results/stage4/suite_c_scalability_raw.csv` and `.json`

**Raw Columns:**
```
scenario_id,agent,seed,return,saved,burnt,firebreaks,steps,time_total_ms,time_per_decision_ms,tree_nodes_per_decision,optimality_gap,max_optimality_gap
```

### Aggregated Scenario Summaries
- `results/stage4/suite_a_small_grid_summary.csv` and `.json`
- `results/stage4/suite_b_budget_sweep_summary.csv` and `.json`
- `results/stage4/suite_c_scalability_summary.csv` and `.json`

**Summary Columns:**
```
scenario_id,agent,n_episodes,return_mean,return_ci,saved_mean,saved_ci,burnt_mean,burnt_ci,firebreaks_mean,firebreaks_ci,time_per_decision_ms_mean,time_per_decision_ms_ci,tree_nodes_mean,tree_nodes_ci,optimality_gap_mean,optimality_gap_ci,peak_memory_mb
```

---

## 4. Instructions for Team Handover

### For Zainab (Stage 5a — Robustness Experiments)
To run wind perturbation or mid-episode shift experiments:
```python
from wildfire import make_config, WildfireEnv, run_scenario, AGENTS, MCTSAgent

# Create perturbed environment
cfg_true = make_config(5, 5, wind="N")
env = WildfireEnv(cfg_true)

# Agent planned on wrong wind "E"
agent_mcts = MCTSAgent(n_rollouts=100)

trials, summaries = run_scenario(
    scenario_id="robustness_wrong_wind",
    env=env,
    agents={"MCTS": agent_mcts},
    seeds=range(100),
    compute_gaps=False
)
```

### For Basma (Stage 5b — Failure Cases & Stage 6a — Plots)
To load data directly for matplotlib / seaborn plotting:
```python
import pandas as pd

# Load budget sweep for optimality gap vs budget plot
df_sweep = pd.read_csv("results/stage4/suite_b_budget_sweep_summary.csv")

# Filter 3x3 scenario
df_3x3 = df_sweep[df_sweep["scenario_id"] == "3x3_budget_sweep"]

# Load scalability data for scaling curve plots
df_scale = pd.read_csv("results/stage4/suite_c_scalability_summary.csv")
```
