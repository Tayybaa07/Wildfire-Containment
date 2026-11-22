"""Stage 4 Master Experiment Runner & Benchmark Suite (Tayyba).

Executes:
1. Suite A: Small-Grid Benchmark & Optimality Gap (3x3 and 3x4)
2. Suite B: MCTS Rollout-Budget & Rollout Policy Sweeps (3x3 and 5x5)
3. Suite C: Large-Grid Scalability Runs (5x5, 6x6, 8x8, 10x10 with k=1 and k=2)

Exports raw trial CSV/JSON and aggregated summary CSV/JSON to `results/stage4/`
for consumption by Stage 5 (robustness & failure cases) and Stage 6 (statistics & plots).
"""
from __future__ import annotations

import argparse
import os
import sys
# Ensure repository root is on sys.path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
from typing import Dict, List, Tuple

from wildfire import (
    Config,
    ExactAgent,
    ExactSolver,
    GreedyAgent,
    MCTSAgent,
    NearestFrontAgent,
    NoOpAgent,
    RandomAgent,
    WildfireEnv,
    make_config,
    run_suite,
)


def get_standard_agents(env: WildfireEnv, include_exact: bool = True, mcts_rollouts: int = 200) -> Dict[str, Any]:
    """Create a dictionary of standard comparison agents."""
    agents = {
        "NoOp": NoOpAgent(),
        "B0-Random": RandomAgent(),
        "B1-NearestFront": NearestFrontAgent(),
        "B2-Greedy": GreedyAgent(),
        "MCTS": MCTSAgent(n_rollouts=mcts_rollouts, rollout_policy="nearest", name=f"MCTS(N={mcts_rollouts})"),
    }
    if include_exact and env.n <= 12:
        agents["Exact"] = ExactAgent()
    return agents


def build_suite_a(quick: bool = False, custom_seeds: int = None) -> Tuple[str, List[Tuple[str, WildfireEnv, Dict[str, Any]]], range]:
    """Suite A: Small-grid benchmark & optimality gap across 3x3 and 3x4."""
    n_seeds = custom_seeds if custom_seeds is not None else (25 if quick else 300)
    seeds = range(n_seeds)

    scenarios = []
    # 3x3 variants
    for veg in ("grass", "mixed"):
        for wind in (None, "E"):
            s_id = f"3x3_{veg}_wind_{wind or 'None'}"
            cfg = make_config(3, 3, vegetation=veg, wind=wind, k=1, lam=0.2, ignition=(0,))
            env = WildfireEnv(cfg)
            scenarios.append((s_id, env, get_standard_agents(env, include_exact=True, mcts_rollouts=150)))

    # 3x4 variants
    for wind in (None, "E"):
        s_id = f"3x4_mixed_wind_{wind or 'None'}"
        cfg = make_config(3, 4, vegetation="mixed", wind=wind, k=1, lam=0.2, ignition=(0,))
        env = WildfireEnv(cfg)
        scenarios.append((s_id, env, get_standard_agents(env, include_exact=True, mcts_rollouts=150)))

    return "suite_a_small_grid", scenarios, seeds


def build_suite_b(quick: bool = False, custom_seeds: int = None) -> Tuple[str, List[Tuple[str, WildfireEnv, Dict[str, Any]]], range]:
    """Suite B: MCTS Rollout-Budget & Policy Sweeps (3x3 and 5x5)."""
    n_seeds = custom_seeds if custom_seeds is not None else (20 if quick else 40)
    seeds = range(n_seeds)

    budgets = [20, 50, 100] if quick else [50, 100, 500, 1000, 5000]
    policies = ["random", "nearest"]

    scenarios = []

    # Scenario 1: 3x3 mixed wind E (where exact gap is known)
    cfg_3 = make_config(3, 3, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env_3 = WildfireEnv(cfg_3)
    agents_3 = {
        "B2-Greedy": GreedyAgent(),
        "Exact": ExactAgent(),
    }
    for b in budgets:
        for p in policies:
            name = f"MCTS(N={b},{p})"
            agents_3[name] = MCTSAgent(n_rollouts=b, rollout_policy=p, name=name)
    scenarios.append(("3x3_budget_sweep", env_3, agents_3))

    # Scenario 2: 5x5 mixed wind E (proposal reference benchmark)
    cfg_5 = make_config(5, 5, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env_5 = WildfireEnv(cfg_5)
    agents_5 = {
        "B1-NearestFront": NearestFrontAgent(),
        "B2-Greedy": GreedyAgent(),
    }
    for b in budgets:
        for p in policies:
            name = f"MCTS(N={b},{p})"
            agents_5[name] = MCTSAgent(n_rollouts=b, rollout_policy=p, name=name)
    scenarios.append(("5x5_budget_sweep", env_5, agents_5))

    return "suite_b_budget_sweep", scenarios, seeds


def build_suite_c(quick: bool = False, custom_seeds: int = None) -> Tuple[str, List[Tuple[str, WildfireEnv, Dict[str, Any]]], range]:
    """Suite C: Large-Grid Scalability Runs (5x5, 6x6, 8x8, 10x10 with k=1 and k=2)."""
    n_seeds = custom_seeds if custom_seeds is not None else (15 if quick else 30)
    seeds = range(n_seeds)

    grid_sizes = [5, 6, 8, 10]
    k_values = [1, 2]

    scenarios = []
    for sz in grid_sizes:
        for k in k_values:
            s_id = f"{sz}x{sz}_mixed_wind_E_k{k}"
            cfg = make_config(sz, sz, vegetation="mixed", wind="E", k=k, lam=0.2, ignition=(0,))
            env = WildfireEnv(cfg)
            agents = {
                "NoOp": NoOpAgent(),
                "B0-Random": RandomAgent(),
                "B1-NearestFront": NearestFrontAgent(),
                "B2-Greedy": GreedyAgent(),
                "MCTS": MCTSAgent(n_rollouts=100, rollout_policy="nearest", name="MCTS(N=100)"),
            }
            scenarios.append((s_id, env, agents))

    return "suite_c_scalability", scenarios, seeds


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Experiment Runner (Wildfire Containment)")
    parser.add_argument("--quick", action="store_true", help="Run in fast smoke-test mode with fewer seeds and smaller sweeps")
    parser.add_argument("--output-dir", type=str, default="results/stage4", help="Directory where CSV/JSON outputs will be saved")
    parser.add_argument("--suites", type=str, default="all", help="Suites to execute: 'all', or comma-separated 'a,b,c'")
    parser.add_argument("--seeds", type=int, default=None, help="Explicitly override number of seeds per scenario")
    args = parser.parse_args()

    selected_suites = [s.strip().lower() for s in args.suites.split(",")] if args.suites != "all" else ["a", "b", "c"]

    print("=" * 80)
    print("STAGE 4: EXPERIMENT RUNNER & BENCHMARKING SUITES (TAYYBA)")
    print(f"Mode       : {'QUICK SMOKE TEST' if args.quick else 'FULL BENCHMARK'}")
    print(f"Output Dir : {args.output_dir}")
    print(f"Suites     : {selected_suites}")
    print("=" * 80)

    t_all_start = time.time()
    generated_files = []

    # 1. Suite A
    if "a" in selected_suites or "all" in selected_suites:
        name_a, scenarios_a, seeds_a = build_suite_a(quick=args.quick, custom_seeds=args.seeds)
        res_a = run_suite(name=name_a, scenarios=scenarios_a, seeds=seeds_a, compute_gaps=True, verbose=True)
        files_a = res_a.save(args.output_dir)
        generated_files.extend(files_a)

    # 2. Suite B
    if "b" in selected_suites or "all" in selected_suites:
        name_b, scenarios_b, seeds_b = build_suite_b(quick=args.quick, custom_seeds=args.seeds)
        # In Suite B, gaps are tracked on 3x3 where solver exists
        res_b = run_suite(name=name_b, scenarios=scenarios_b, seeds=seeds_b, compute_gaps=True, verbose=True)
        files_b = res_b.save(args.output_dir)
        generated_files.extend(files_b)

    # 3. Suite C
    if "c" in selected_suites or "all" in selected_suites:
        name_c, scenarios_c, seeds_c = build_suite_c(quick=args.quick, custom_seeds=args.seeds)
        # In Suite C, exact gaps are safely skipped (grids >= 5x5)
        res_c = run_suite(name=name_c, scenarios=scenarios_c, seeds=seeds_c, compute_gaps=False, verbose=True)
        files_c = res_c.save(args.output_dir)
        generated_files.extend(files_c)

    elapsed_total = time.time() - t_all_start
    print("\n" + "=" * 80)
    print(f"STAGE 4 RUN COMPLETED IN {elapsed_total:.2f}s")
    print(f"Saved {len(generated_files)} structured files to {args.output_dir}:")
    for fpath in generated_files:
        size_kb = os.path.getsize(fpath) / 1024.0
        print(f"  - {fpath} ({size_kb:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
