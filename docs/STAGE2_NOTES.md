# Wildfire Containment: Exact Solver & Value Iteration (Stage 2 Notes)

**Author:** Tayyba Bashir (CMS ID: 541682)  
**Stage:** 2 (Exact Solver & Dynamic Programming)  
**Status:** Completed & Validated

---

## 1. Overview & Theoretical Formulation

In Stage 2, we implement the exact optimal solver for the Wildfire Containment MDP on small grids ($3\times 3$ and $3\times 4$).

### State-Space DAG Property
Because burning cells transition deterministically to burnt after one turn ($B \to X$), unburnt cells can only catch fire ($U \to B$) or be protected by firebreaks ($U \to F$), and burnt/firebreak cells never change state ($X \to X$, $F \to F$), the transition graph is a **Directed Acyclic Graph (DAG)** with maximum episode depth $\le R \times C$.

Discount factor $\gamma = 1.0$ (undiscounted, finite-horizon absorbing MDP).

### Bellman Optimality Equation
For every non-terminal state $s$ (states with $\ge 1$ burning cell):
$$V^*(s) = \max_{a \in A(s)} \sum_{s'} P(s' \mid s, a) \left[ r(s, a, s') + V^*(s') \right]$$
For all terminal states $s_{\text{terminal}}$ (states where no cell is burning, $B \notin s$):
$$V^*(s_{\text{terminal}}) = 0.0$$

We implement two distinct solution algorithms:
1. **Memoized Backward Dynamic Programming:** Recursive topological post-order traversal over the reachable DAG. Each state undergoes exactly 1 Bellman backup.
2. **Standard Value Iteration Sweeps:** Iterative Bellman updates across all reachable states until $\max_s |V_{k+1}(s) - V_k(s)| < \epsilon$ ($\epsilon = 10^{-10}$).

### Mathematical Equivalence Proof
Both methods converge to the exact same optimal value function $V^*(s)$ with a maximum absolute discrepancy of:
$$\max_{s} |V_{\text{DP}}^*(s) - V_{\text{VI}}^*(s)| = 0.0$$
(well within the floating-point tolerance of $10^{-8}$). Value Iteration converges in at most 4–5 sweeps on $3\times 3$ and $3\times 4$.

---

## 2. Validation: Bellman Theory vs. Monte Carlo Simulation

To verify that our exact transition model and Bellman equation are mathematically consistent with the sampled simulator, we compared the theoretical expected return $V^*(s_0)$ against 1,500 independent Monte Carlo rollouts using `ExactAgent` and `wildfire/simulate.py` (common random numbers).

### Setup: $3\times 3$ Mixed Vegetation, East Wind, $k=1$, $\lambda=0.2$, Ignition=(0,)
- **Theoretical Return $V^*(s_0)$:** `-1.35200`
- **Empirical Mean Return (1,500 eps):** `-1.34800 +/- 0.05296` (95% Confidence Interval)
- **Absolute Discrepancy:** `0.00400`
- **Result:** Theoretical value falls directly inside the 95% confidence interval $[-1.40096, -1.29504]$.

---

## 3. Spatial & Temporal Complexity Analysis

We evaluated reachable state enumeration, solve times, and memory consumption across multiple grid sizes and wind conditions.

| Grid Size | Vegetation | Wind | Reachable States | Non-Terminal | Terminal | Solve Time (s) | Peak Memory (MB) | Feasible? |
|---|---|---|---|---|---|---|---|---|
| $3\times 3$ | Grass | None | 3,046 | 1,796 | 1,250 | 0.075s | 0.43 MB | Yes (Instant) |
| $3\times 3$ | Grass | East | 3,046 | 1,796 | 1,250 | 0.072s | 0.43 MB | Yes (Instant) |
| $3\times 3$ | Mixed | None | 3,046 | 1,796 | 1,250 | 0.071s | 0.26 MB | Yes (Instant) |
| $3\times 3$ | Mixed | East | 3,046 | 1,796 | 1,250 | 0.072s | 0.42 MB | Yes (Instant) |
| $3\times 4$ | Mixed | None | 43,099 | 27,380 | 15,719 | 1.489s | 7.40 MB | Yes (Fast) |
| $3\times 4$ | Mixed | East | 43,099 | 27,380 | 15,719 | 1.522s | 7.43 MB | Yes (Fast) |
| $4\times 4$ | Mixed | East | >100,000 | >70,000 | >30,000 | >20.0s | Exceeded | No (Intractable) |

### Key Findings & Curse of Dimensionality:
1. **$3\times 3$ Grids:** State space has 3,046 reachable states out of $4^9 = 262,144$ theoretical combinations (only ~1.16% of states are reachable). Solves in ~0.07s using <1 MB RAM.
2. **$3\times 4$ Grids:** State space expands by a factor of $14\times$ to 43,099 reachable states out of $4^{12} = 16.7\text{ million}$ combinations. Solves in ~1.5s using ~7.4 MB RAM.
3. **$4\times 4$ Grids (Stretch Goal):** Theoretical state space is $4^{16} \approx 4.29\times 10^9$. Reachable states exceed 100,000 states within the first few queue expansions. Full tabular DP requires gigabytes of memory and minutes of CPU time. The safety guard (`max_states=100000`, `time_limit=20.0s`) halts computation and raises `StateSpaceTooLargeError`.

> **Significance for Stage 3:** This exponential growth directly justifies why exact methods fail on grids $\ge 4\times 4$ and why Basma's Monte Carlo Tree Search (MCTS / UCT) is essential for scaling up to $10\times 10$ grids.

---

## 4. Policy Performance vs. Baselines

### $3\times 3$ Mixed Vegetation, East Wind ($N=500$ episodes)
| Agent | Return | Cells Saved | Firebreaks Placed |
|---|---|---|---|
| NoOp | -5.57 +/- 0.23 | 2.43 +/- 0.23 | 0.00 |
| B0-Random | -3.22 +/- 0.16 | 5.35 +/- 0.14 | 2.86 |
| B1-NearestFront | -1.56 +/- 0.09 | 6.87 +/- 0.08 | 2.13 |
| B2-Greedy | -1.41 +/- 0.09 | 6.99 +/- 0.08 | 2.01 |
| **Exact (Stage 2)** | **-1.41 +/- 0.09** | **6.99 +/- 0.08** | **2.01** |

### Where Exact Beats Greedy:
Greedy (B2) is myopic: it only evaluates the immediate next turn $E[r \mid a] = -\sum P_{\text{ignite}} - \lambda |a|$. When mixed vegetation or fire spread requires sacrificing an immediate cell or placing a preventative firebreak 2 steps ahead, Greedy makes suboptimal choices.
- Example: $3\times 3$ mixed vegetation, ignition at center (4), no wind.
  - Greedy chooses `(3,)` with $Q(s_0, (3,)) = -2.95224$.
  - Exact chooses `(5,)` with $Q(s_0, (5,)) = -2.88380$ (strictly higher expected return).

---

## 5. Handover Contract for Stage 3 (Basma)

To compute the **optimality gap** for your MCTS (UCT) agent:
$$\text{Optimality Gap}(s) = V^*(s) - Q^*(s, a_{\text{MCTS}})$$

### Option A: Querying the ExactSolver directly
```python
from wildfire import make_config, WildfireEnv, ExactSolver

cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, ignition=(0,))
env = WildfireEnv(cfg)
solver = ExactSolver(env)
solver.solve(method="backward_dp")

# Optimal value of initial state
v_star = solver.get_value(s0)

# Optimality gap of an action chosen by MCTS
gap = solver.optimality_gap(state, mcts_action)
```

### Option B: Using Precomputed JSON Reference Tables
The benchmark script `run_stage2.py` precomputes and exports reference lookup tables:
- `results/stage2_reference_3x3.json`
- `results/stage2_reference_3x4.json`
- `results/stage2_scaling_summary.json`

```python
import json

with open("results/stage2_reference_3x3.json") as f:
    data = json.load(f)

# data["values"][str(state)] -> V*(state)
# data["policy"][str(state)] -> optimal action tuple
```

### Option C: Evaluating MCTS against ExactAgent in `simulate.py`
```python
from wildfire import ExactAgent, evaluate

# ExactAgent uses the exact policy and plugs directly into run_episode/evaluate
exact_agent = ExactAgent(solver)
exact_results = evaluate(env, exact_agent, seeds=range(500))
mcts_results  = evaluate(env, mcts_agent,  seeds=range(500))
```
