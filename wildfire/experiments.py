"""Experiment Runner and Benchmarking Suite (Stage 4).

Provides reproducible experiment execution with:
- Common Random Numbers (CRN) across agents and seeds
- Decision-level timing and latency metrics (ms/decision)
- MCTS tree node tracking and memory measurement (tracemalloc)
- Automatic optimality gap tracking relative to ExactSolver on small grids
- Structured CSV and JSON artifact generation for Stage 5 (robustness/failure cases)
  and Stage 6 (statistics and plotting).
"""
from __future__ import annotations

import csv
import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents import Agent
from .config import Config, make_config
from .env import Action, State, WildfireEnv
from .exact_solver import ExactSolver, StateSpaceTooLargeError
from .simulate import mean_ci


@dataclass
class TrialResult:
    """Raw record for a single episode trial."""
    scenario_id: str
    agent: str
    seed: int
    return_: float
    saved: int
    burnt: int
    firebreaks: int
    steps: int
    time_total_ms: float
    time_per_decision_ms: float
    tree_nodes_per_decision: Optional[float] = None
    optimality_gap: Optional[float] = None
    max_optimality_gap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["return"] = d.pop("return_")
        return d


@dataclass
class ScenarioSummary:
    """Aggregated summary statistics for one (scenario, agent) pair."""
    scenario_id: str
    agent: str
    n_episodes: int
    return_mean: float
    return_ci: float
    saved_mean: float
    saved_ci: float
    burnt_mean: float
    burnt_ci: float
    firebreaks_mean: float
    firebreaks_ci: float
    time_per_decision_ms_mean: float
    time_per_decision_ms_ci: float
    tree_nodes_mean: Optional[float]
    tree_nodes_ci: Optional[float]
    optimality_gap_mean: Optional[float]
    optimality_gap_ci: Optional[float]
    peak_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSuiteResult:
    """Full container holding raw trials and summaries for a benchmark suite."""
    name: str
    trials: List[TrialResult] = field(default_factory=list)
    summaries: List[ScenarioSummary] = field(default_factory=list)

    def save(self, output_dir: str) -> Tuple[str, str, str, str]:
        """Save raw and summary tables as both CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)
        raw_csv = os.path.join(output_dir, f"{self.name}_raw.csv")
        raw_json = os.path.join(output_dir, f"{self.name}_raw.json")
        sum_csv = os.path.join(output_dir, f"{self.name}_summary.csv")
        sum_json = os.path.join(output_dir, f"{self.name}_summary.json")

        # 1. Raw CSV
        trial_dicts = [t.to_dict() for t in self.trials]
        if trial_dicts:
            keys = list(trial_dicts[0].keys())
            with open(raw_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(trial_dicts)

        # 2. Raw JSON
        with open(raw_json, "w") as f:
            json.dump(trial_dicts, f, indent=2)

        # 3. Summary CSV
        sum_dicts = [s.to_dict() for s in self.summaries]
        if sum_dicts:
            keys = list(sum_dicts[0].keys())
            with open(sum_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(sum_dicts)

        # 4. Summary JSON
        with open(sum_json, "w") as f:
            json.dump(sum_dicts, f, indent=2)

        return raw_csv, raw_json, sum_csv, sum_json


def run_monitored_episode(
    env: WildfireEnv,
    agent: Agent,
    seed: int,
    scenario_id: str = "default",
    solver: Optional[ExactSolver] = None,
    compute_gaps: bool = True,
) -> TrialResult:
    """Execute a single episode with precise timing, node counts, and optimality gaps.

    Ensures Common Random Numbers by using SeedSequence(seed).spawn(2) identical
    to wildfire.simulate.run_episode.
    """
    env_seq, agent_seq = np.random.SeedSequence(seed).spawn(2)
    env_rng = np.random.default_rng(env_seq)
    agent_rng = np.random.default_rng(agent_seq)

    state = env.reset(seed=seed)
    total_reward = 0.0
    firebreaks = 0
    steps = 0
    decision_times_ms: List[float] = []
    tree_nodes: List[int] = []
    step_gaps: List[float] = []

    t_start = time.perf_counter()

    for _ in range(env.horizon + 1):
        if env.is_terminal(state):
            break

        # Time the individual decision
        d_start = time.perf_counter()
        action = agent.select_action(state, env, agent_rng)
        d_elapsed_ms = (time.perf_counter() - d_start) * 1000.0
        decision_times_ms.append(d_elapsed_ms)

        # Record tree nodes if the agent tracks it (e.g. MCTSAgent)
        if hasattr(agent, "last_tree_nodes") and agent.last_tree_nodes is not None:
            tree_nodes.append(int(agent.last_tree_nodes))

        # Record optimality gap if exact solver is provided
        if compute_gaps and solver is not None:
            gap = solver.optimality_gap(state, action)
            step_gaps.append(float(gap))

        state, reward, _ = env.step(state, action, env_rng)
        total_reward += reward
        firebreaks += len(action)
        steps += 1
    else:
        raise RuntimeError("Episode exceeded the horizon.")

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    mean_decision_ms = float(np.mean(decision_times_ms)) if decision_times_ms else 0.0
    mean_tree_nodes = float(np.mean(tree_nodes)) if tree_nodes else None
    mean_gap = float(np.mean(step_gaps)) if step_gaps else None
    max_gap = float(np.max(step_gaps)) if step_gaps else None

    burnt = env.count(state, env.cfg.vegetation[0][0] if False else 2)  # X code = 2
    # Count burnt directly by searching for X
    burnt = sum(1 for v in state if v == 2)
    saved = env.n - burnt

    return TrialResult(
        scenario_id=scenario_id,
        agent=agent.name,
        seed=seed,
        return_=float(total_reward),
        saved=int(saved),
        burnt=int(burnt),
        firebreaks=int(firebreaks),
        steps=int(steps),
        time_total_ms=float(total_time_ms),
        time_per_decision_ms=mean_decision_ms,
        tree_nodes_per_decision=mean_tree_nodes,
        optimality_gap=mean_gap,
        max_optimality_gap=max_gap,
    )


def run_scenario(
    scenario_id: str,
    env: WildfireEnv,
    agents: Dict[str, Agent],
    seeds: Sequence[int],
    compute_gaps: bool = True,
    solver: Optional[ExactSolver] = None,
) -> Tuple[List[TrialResult], List[ScenarioSummary]]:
    """Run all agents across the provided seeds on a single scenario.

    Monitors peak memory with tracemalloc and computes paired summaries.
    """
    # For small grids (R*C <= 12), prepare ExactSolver once if gap computation is enabled
    can_solve_exact = (env.n <= 12)
    active_solver: Optional[ExactSolver] = solver
    if compute_gaps and can_solve_exact and active_solver is None:
        try:
            active_solver = ExactSolver(env)
            # If ignition is fixed, pre-solve it; if 3x3, pre-solve all single-cell ignitions
            if env.cfg.ignition is not None:
                active_solver.solve_backward_dp(initial_state=env.reset(ignition=env.cfg.ignition), max_states=80000, time_limit=15.0)
            elif env.n <= 9:
                for c in range(env.n):
                    active_solver.solve_backward_dp(initial_state=env.reset(ignition=(c,)), max_states=80000, time_limit=15.0)
            else:
                active_solver.solve_backward_dp(initial_state=env.reset(), max_states=80000, time_limit=15.0)
        except (StateSpaceTooLargeError, Exception):
            active_solver = None

    # Connect active_solver to any ExactAgent instance to avoid re-solving
    for agent in agents.values():
        if hasattr(agent, "solver") and getattr(agent, "solver") is None and active_solver is not None:
            setattr(agent, "solver", active_solver)

    trials: List[TrialResult] = []
    summaries: List[ScenarioSummary] = []

    for agent_key, agent in agents.items():
        tracemalloc.start()
        agent_trials: List[TrialResult] = []

        for seed in seeds:
            t = run_monitored_episode(
                env=env,
                agent=agent,
                seed=seed,
                scenario_id=scenario_id,
                solver=active_solver,
                compute_gaps=compute_gaps and (active_solver is not None),
            )
            agent_trials.append(t)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_mem / (1024.0 * 1024.0)

        trials.extend(agent_trials)

        # Aggregate metrics
        rets = [t.return_ for t in agent_trials]
        savs = [t.saved for t in agent_trials]
        brns = [t.burnt for t in agent_trials]
        fbrs = [t.firebreaks for t in agent_trials]
        tpds = [t.time_per_decision_ms for t in agent_trials]

        nodes = [t.tree_nodes_per_decision for t in agent_trials if t.tree_nodes_per_decision is not None]
        gaps = [t.optimality_gap for t in agent_trials if t.optimality_gap is not None]

        r_m, r_ci = mean_ci(rets)
        s_m, s_ci = mean_ci(savs)
        b_m, b_ci = mean_ci(brns)
        f_m, f_ci = mean_ci(fbrs)
        t_m, t_ci = mean_ci(tpds)

        n_m, n_ci = mean_ci(nodes) if nodes else (None, None)
        g_m, g_ci = mean_ci(gaps) if gaps else (None, None)

        summaries.append(
            ScenarioSummary(
                scenario_id=scenario_id,
                agent=agent.name,
                n_episodes=len(agent_trials),
                return_mean=float(r_m),
                return_ci=float(r_ci),
                saved_mean=float(s_m),
                saved_ci=float(s_ci),
                burnt_mean=float(b_m),
                burnt_ci=float(b_ci),
                firebreaks_mean=float(f_m),
                firebreaks_ci=float(f_ci),
                time_per_decision_ms_mean=float(t_m),
                time_per_decision_ms_ci=float(t_ci),
                tree_nodes_mean=float(n_m) if n_m is not None else None,
                tree_nodes_ci=float(n_ci) if n_ci is not None else None,
                optimality_gap_mean=float(g_m) if g_m is not None else None,
                optimality_gap_ci=float(g_ci) if g_ci is not None else None,
                peak_memory_mb=float(peak_mb),
            )
        )

    return trials, summaries


def run_suite(
    name: str,
    scenarios: Sequence[Tuple[str, WildfireEnv, Dict[str, Agent]]],
    seeds: Sequence[int],
    compute_gaps: bool = True,
    verbose: bool = True,
) -> ExperimentSuiteResult:
    """Execute an entire suite across multiple scenarios and return ExperimentSuiteResult."""
    suite_res = ExperimentSuiteResult(name=name)

    if verbose:
        print(f"\n{'='*75}\nRUNNING EXPERIMENT SUITE: {name.upper()} ({len(scenarios)} scenarios, {len(seeds)} seeds/agent)\n{'='*75}")

    for scenario_id, env, agents in scenarios:
        if verbose:
            print(f"\nScenario: {scenario_id} ({env.R}x{env.C}, wind={env.cfg.wind}, k={env.cfg.k})")
            print(f"{'Agent':<24} | {'Return':>14} | {'Saved':>12} | {'Time/Dec (ms)':>15} | {'Gap to Opt':>12}")
            print("-" * 85)

        s_trials, s_summaries = run_scenario(
            scenario_id=scenario_id,
            env=env,
            agents=agents,
            seeds=seeds,
            compute_gaps=compute_gaps,
        )
        suite_res.trials.extend(s_trials)
        suite_res.summaries.extend(s_summaries)

        if verbose:
            for s in s_summaries:
                gap_str = (
                    f"{s.optimality_gap_mean:.4f} +/-{s.optimality_gap_ci:.4f}"
                    if s.optimality_gap_mean is not None
                    else "N/A"
                )
                print(
                    f"{s.agent:<24} | "
                    f"{s.return_mean:>6.2f} +/-{s.return_ci:<4.2f} | "
                    f"{s.saved_mean:>5.2f} +/-{s.saved_ci:<3.2f} | "
                    f"{s.time_per_decision_ms_mean:>8.2f} +/-{s.time_per_decision_ms_ci:<4.2f} | "
                    f"{gap_str:>12}"
                )

    return suite_res
