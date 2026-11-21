"""Stage 2 Validation & Benchmarking Script (Tayyba).

Executes:
1. Mathematical validation / Simulation cross-check:
   Compares theoretical V*(s0) against 1,500+ Monte Carlo simulated episodes
   of ExactAgent using wildfire/simulate.py (with 95% confidence intervals).
2. Grid scaling benchmark:
   Measures reachable state counts, solve time, peak memory, and return vs baselines across:
   - 3x3 grids (grass vs mixed, no wind vs E wind)
   - 3x4 grids
   - 4x4 grids (guarded attempt reporting state-space explosion and feasibility limits)
3. Handover Artifact Generation for Stage 3 (Basma):
   Exports precomputed optimal reference tables to `results/` for calculating MCTS optimality gaps.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from wildfire import (
    Config,
    WildfireEnv,
    make_config,
    run_episode,
    evaluate,
    mean_ci,
    NoOpAgent,
    RandomAgent,
    NearestFrontAgent,
    GreedyAgent,
    ExactSolver,
    ExactAgent,
    StateSpaceTooLargeError,
)


def run_cross_check(episodes: int = 1500) -> Dict[str, Any]:
    print("=" * 70)
    print("STEP 1: THEORETICAL BELLMAN vs MONTE CARLO SIMULATION CROSS-CHECK")
    print("=" * 70)

    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)

    print(f"Solving 3x3 mixed grid (wind=E, ignition=0, k=1, lam=0.2)...")
    res = solver.solve(method="both", epsilon=1e-10)
    v_theoretical = res.v_s0

    print(f"  Exact V*(s0)             : {v_theoretical:.5f}")
    print(f"  Reachable states         : {res.reachable_count} ({res.non_terminal_count} non-terminal, {res.terminal_count} terminal)")
    print(f"  VI sweeps to converge    : {res.sweeps}")
    print(f"  Max |V_DP - V_VI|        : {res.max_dp_vi_diff:.1e}")

    print(f"\nSimulating {episodes} episodes using ExactAgent (common random numbers)...")
    agent = ExactAgent(solver)
    seeds = range(episodes)
    sim_results = evaluate(env, agent, seeds)

    returns = [x["return"] for x in sim_results]
    saved = [x["saved"] for x in sim_results]
    fb = [x["firebreaks"] for x in sim_results]

    ret_mean, ret_ci = mean_ci(returns)
    saved_mean, saved_ci = mean_ci(saved)
    fb_mean = float(np.mean(fb))

    diff = abs(ret_mean - v_theoretical)
    within_ci = diff <= ret_ci

    print(f"  Theoretical Return V*(s0): {v_theoretical:.5f}")
    print(f"  Empirical Mean Return    : {ret_mean:.5f} +/- {ret_ci:.5f} (95% CI)")
    print(f"  Absolute Discrepancy     : {diff:.5f}")
    print(f"  Inside 95% Confidence Int: {within_ci}")
    print(f"  Average Cells Saved      : {saved_mean:.2f} +/- {saved_ci:.2f}")
    print(f"  Average Firebreaks Placed: {fb_mean:.2f}")

    if within_ci:
        print("\n>>> VALIDATION SUCCESS: Theoretical V*(s0) matches Monte Carlo empirical return within 95% CI.")
    else:
        print("\n>>> WARNING: Theoretical V*(s0) outside 95% CI (rare statistical fluctuation or model mismatch).")

    return {
        "v_theoretical": v_theoretical,
        "empirical_mean": ret_mean,
        "empirical_ci": ret_ci,
        "discrepancy": diff,
        "within_ci": within_ci,
        "episodes": episodes,
        "saved_mean": saved_mean,
        "fb_mean": fb_mean,
    }


def run_scaling_benchmarks() -> List[Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("STEP 2: GRID SCALING BENCHMARKS & SPATIAL/TEMPORAL COMPLEXITY")
    print("=" * 70)

    scenarios = [
        ("3x3 grass (wind=None)", 3, 3, "grass", None, 1, (0,)),
        ("3x3 grass (wind=E)", 3, 3, "grass", "E", 1, (0,)),
        ("3x3 mixed (wind=None)", 3, 3, "mixed", None, 1, (0,)),
        ("3x3 mixed (wind=E)", 3, 3, "mixed", "E", 1, (0,)),
        ("3x4 mixed (wind=None)", 3, 4, "mixed", None, 1, (0,)),
        ("3x4 mixed (wind=E)", 3, 4, "mixed", "E", 1, (0,)),
        ("4x4 mixed (wind=E) [Guarded]", 4, 4, "mixed", "E", 1, (0,)),
    ]

    records = []

    print(f"{'Scenario':<28} | {'States':<8} | {'Solve(s)':<8} | {'Peak Mem':<9} | {'V*(s0)':<8} | {'Feasible?'}")
    print("-" * 75)

    for name, r, c, veg, wind, k, ign in scenarios:
        cfg = make_config(r, c, vegetation=veg, wind=wind, k=k, lam=0.2, ignition=ign)
        env = WildfireEnv(cfg)
        solver = ExactSolver(env)

        # 4x4 has a safety limit of 100,000 states to prevent memory exhaustion
        max_states = 100000 if (r * c >= 16) else 150000
        time_limit = 20.0

        try:
            res = solver.solve(method="backward_dp", max_states=max_states, time_limit=time_limit)
            rec = {
                "scenario": name,
                "rows": r,
                "cols": c,
                "veg": veg,
                "wind": wind,
                "states": res.reachable_count,
                "non_terminal": res.non_terminal_count,
                "terminal": res.terminal_count,
                "solve_time": round(res.solve_time, 3),
                "peak_mem_mb": round(res.peak_memory_mb, 2),
                "v_s0": round(res.v_s0, 4),
                "feasible": True,
            }
            records.append(rec)
            print(f"{name:<28} | {res.reachable_count:<8} | {res.solve_time:<8.3f} | {res.peak_memory_mb:<6.2f} MB | {res.v_s0:<8.4f} | Yes")
        except StateSpaceTooLargeError as e:
            rec = {
                "scenario": name,
                "rows": r,
                "cols": c,
                "veg": veg,
                "wind": wind,
                "states": f">{max_states}",
                "solve_time": f">{time_limit:.1f}s",
                "peak_mem_mb": "N/A (Capped)",
                "v_s0": "N/A",
                "feasible": False,
                "error": str(e),
            }
            records.append(rec)
            print(f"{name:<28} | {f'>{max_states}':<8} | {f'>{time_limit:.0f}s':<8} | {'Exceeded':<9} | {'N/A':<8} | No (Intractable)")

    return records


def run_baseline_comparison(grid_name: str, rows: int, cols: int, veg: str, wind: str, episodes: int = 500) -> None:
    print(f"\n--- Baseline Comparison on {grid_name} ({episodes} episodes) ---")
    cfg = make_config(rows, cols, vegetation=veg, wind=wind, k=1, lam=0.2, ignition=(0,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)
    solver.solve(method="backward_dp")

    agents = {
        "NoOp": NoOpAgent(),
        "B0-Random": RandomAgent(),
        "B1-NearestFront": NearestFrontAgent(),
        "B2-Greedy": GreedyAgent(),
        "Exact (Stage 2)": ExactAgent(solver),
    }

    seeds = range(episodes)
    print(f"{'Agent':<18} | {'Return':>16} | {'Cells Saved':>16} | {'Firebreaks':>12}")
    print("-" * 70)
    for ag_name, ag in agents.items():
        res = evaluate(env, ag, seeds)
        r_m, r_ci = mean_ci([x["return"] for x in res])
        s_m, s_ci = mean_ci([x["saved"] for x in res])
        fb_m = float(np.mean([x["firebreaks"] for x in res]))
        print(f"{ag_name:<18} | {r_m:>7.2f} +/-{r_ci:<5.2f} | {s_m:>7.2f} +/-{s_ci:<5.2f} | {fb_m:>12.2f}")


def export_handover_artifacts() -> None:
    print("\n" + "=" * 70)
    print("STEP 3: EXPORTING HANDOVER ARTIFACTS FOR STAGE 3 (BASMA)")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    # Export 3x3 reference policy & values
    cfg_3x3 = make_config(3, 3, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env_3x3 = WildfireEnv(cfg_3x3)
    solver_3x3 = ExactSolver(env_3x3)
    solver_3x3.solve(method="both")

    path_3x3 = "results/stage2_reference_3x3.json"
    solver_3x3.save(path_3x3)
    print(f"Saved 3x3 reference table: {path_3x3} ({len(solver_3x3.V)} states)")

    # Export 3x4 reference policy & values
    cfg_3x4 = make_config(3, 4, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env_3x4 = WildfireEnv(cfg_3x4)
    solver_3x4 = ExactSolver(env_3x4)
    solver_3x4.solve(method="backward_dp")

    path_3x4 = "results/stage2_reference_3x4.json"
    solver_3x4.save(path_3x4)
    print(f"Saved 3x4 reference table: {path_3x4} ({len(solver_3x4.V)} states)")

    # Export scaling summary
    scaling_records = run_scaling_benchmarks()
    path_scaling = "results/stage2_scaling_summary.json"
    with open(path_scaling, "w") as f:
        json.dump(scaling_records, f, indent=2)
    print(f"Saved scaling summary: {path_scaling}")


def main():
    # 1. Theoretical vs Monte Carlo cross check
    run_cross_check(episodes=1500)

    # 2. Baseline comparisons on 3x3 and 3x4
    run_baseline_comparison("3x3 mixed (wind=E)", 3, 3, "mixed", "E", episodes=500)
    run_baseline_comparison("3x4 mixed (wind=E)", 3, 4, "mixed", "E", episodes=300)

    # 3. Handover artifacts for Basma
    export_handover_artifacts()
    print("\n>>> Stage 2 verification & benchmarking complete.")


if __name__ == "__main__":
    main()
