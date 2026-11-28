"""Stage 5b (Basma): Failure-Case Analysis.

Reproduces and documents three concrete failure cases from Section 6 of the proposal:
  Case 1: MCTS with too few rollouts on high-variance / large-action grids
          (noisy decisions, high optimality gap, losing to B1/B2).
  Case 2: Model mismatch (agent plans with wrong opposite wind W when true wind is E),
          where model-free B1-NearestFront beats model-based MCTS and B2-Greedy.
  Case 3: Myopic trap where 1-step B2-Greedy and B1-NearestFront pick a suboptimal
          immediate move, while Exact and high-budget MCTS find the better move.

Run from the project root:
    python experiments/failure_cases.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wildfire import (  # noqa: E402
    ExactAgent,
    ExactSolver,
    GreedyAgent,
    MCTSAgent,
    NearestFrontAgent,
    RandomAgent,
    WildfireEnv,
    evaluate,
    make_config,
    mean_ci,
)
from wildfire.robustness import run_mismatch_scenario  # noqa: E402


def summarize(rows: list[dict]) -> tuple[float, float, float]:
    ret_m, ret_ci = mean_ci([r["return"] for r in rows])
    sav_m, _ = mean_ci([r["saved"] for r in rows])
    return ret_m, ret_ci, sav_m


def case_1_low_rollout_budget(records: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("FAILURE CASE 1: Too Few MCTS Rollouts on High-Variance & Large Grids")
    print("=" * 75)

    # 1A: 3x3 high-flammability grass + East wind
    cfg_small = make_config(3, 3, vegetation="grass", wind="E", k=1, lam=0.2, ignition=(0,))
    env_small = WildfireEnv(cfg_small)
    solver = ExactSolver(env_small)
    solver.solve(method="backward_dp")
    s0 = env_small.reset()

    seeds = list(range(60))
    agents_small = {
        "B0-Random": RandomAgent(),
        "MCTS(N=15, random)": MCTSAgent(n_rollouts=15, rollout_policy="random"),
        "MCTS(N=300, nearest)": MCTSAgent(n_rollouts=300, rollout_policy="nearest"),
        "Exact": ExactAgent(solver),
    }
    print("\n[1A] 3x3 Grass + Wind E (60 episodes):")
    for name, ag in agents_small.items():
        res = evaluate(env_small, ag, seeds)
        rm, rci, sm = summarize(res)
        # Measure root decision optimality gap across 30 seeds
        gaps = [
            solver.optimality_gap(s0, ag.select_action(s0, env_small, np.random.default_rng(sd)))
            for sd in range(30)
        ]
        gm, gci = mean_ci(gaps)
        print(f"  {name:<22} Return={rm:6.2f} +/-{rci:.2f} | Saved={sm:5.2f} | Root Gap={gm:.4f} +/-{gci:.4f}")
        records.append({
            "case": "Case1A_3x3_grass_E", "agent": name,
            "return_mean": round(rm, 4), "return_ci": round(rci, 4),
            "saved_mean": round(sm, 4), "note": f"root_gap={gm:.4f}",
        })

    # 1B: 6x6 mixed + East wind (36 legal actions, N=25 rollouts cannot even visit each move once!)
    cfg_big = make_config(6, 6, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env_big = WildfireEnv(cfg_big)
    seeds_big = list(range(25))
    agents_big = {
        "B0-Random": RandomAgent(),
        "B1-NearestFront": NearestFrontAgent(),
        "B2-Greedy": GreedyAgent(),
        "MCTS(N=25, nearest)": MCTSAgent(n_rollouts=25, rollout_policy="nearest"),
        "MCTS(N=200, nearest)": MCTSAgent(n_rollouts=200, rollout_policy="nearest"),
    }
    print("\n[1B] 6x6 Mixed + Wind E (36 actions; N=25 < 36 actions, 25 episodes):")
    for name, ag in agents_big.items():
        res = evaluate(env_big, ag, seeds_big)
        rm, rci, sm = summarize(res)
        print(f"  {name:<22} Return={rm:6.2f} +/-{rci:.2f} | Saved={sm:5.2f}")
        records.append({
            "case": "Case1B_6x6_underbudget", "agent": name,
            "return_mean": round(rm, 4), "return_ci": round(rci, 4),
            "saved_mean": round(sm, 4), "note": "36 legal actions at root",
        })


def case_2_model_mismatch(records: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("FAILURE CASE 2: Wrong Wind Model (True=E, Believed=W) -> B1 Beats MCTS")
    print("=" * 75)

    true_env = WildfireEnv(make_config(5, 5, vegetation="mixed", veg_seed=0, wind="E", k=1, lam=0.2))
    wrong_env = WildfireEnv(make_config(5, 5, vegetation="mixed", veg_seed=0, wind="W", k=1, lam=0.2))
    seeds = list(range(40))

    factories = {
        "B1-NearestFront": lambda e: NearestFrontAgent(),
        "B2-Greedy": lambda e: GreedyAgent(),
        "MCTS(N=100, nearest)": lambda e: MCTSAgent(n_rollouts=100, rollout_policy="nearest"),
    }

    print("\n5x5 Mixed (40 episodes):")
    for cond_name, believed in (("Correct Model (wind=E)", true_env), ("Wrong Model (believed=W)", wrong_env)):
        _, summaries = run_mismatch_scenario(cond_name, true_env, believed, factories, seeds)
        print(f"  Condition: {cond_name}")
        for s in summaries:
            print(f"    {s.agent:<22} Return={s.return_mean:6.2f} +/-{s.return_ci:.2f} | Saved={s.saved_mean:5.2f}")
            records.append({
                "case": f"Case2_{cond_name}", "agent": s.agent,
                "return_mean": round(s.return_mean, 4), "return_ci": round(s.return_ci, 4),
                "saved_mean": round(s.saved_mean, 4), "note": cond_name,
            })


def case_3_myopic_trap(records: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("FAILURE CASE 3: Myopic Trap (Greedy/Nearest Miss Multi-Step Planning)")
    print("=" * 75)

    # Reference trap from STAGE2_NOTES.md Section 4: 3x3 mixed, center ignition (4), no wind
    cfg = make_config(3, 3, vegetation="mixed", veg_seed=0, wind=None, k=1, lam=0.2, ignition=(4,))
    env = WildfireEnv(cfg)
    s0 = env.reset()

    solver = ExactSolver(env)
    solver.solve(initial_state=s0, method="backward_dp")
    v_star = solver.get_value(s0)
    pi_star = solver.get_action(s0)

    print(f"\n3x3 Mixed (veg_seed=0, wind=None, ignition=(4,)):")
    print(f"  Optimal V*(s0) = {v_star:.5f}, Optimal action = {pi_star}")

    for name, ag in (
        ("B1-NearestFront", NearestFrontAgent()),
        ("B2-Greedy", GreedyAgent()),
        ("Exact", ExactAgent(solver)),
    ):
        a = ag.select_action(s0, env, np.random.default_rng(0))
        q = solver.q_value(s0, a)
        gap = solver.optimality_gap(s0, a)
        print(f"  {name:<18} picks {str(a):<6} -> Q*(s0,a)={q:.5f} | Gap={gap:.5f}")
        records.append({
            "case": "Case3_myopic_trap", "agent": name,
            "return_mean": round(q, 5), "return_ci": 0.0,
            "saved_mean": 0.0, "note": f"action={a}, gap={gap:.5f}",
        })

    # Check whether MCTS finds the optimal multi-step action as budget grows
    for n_roll in (50, 300):
        ag = MCTSAgent(n_rollouts=n_roll, rollout_policy="nearest")
        gaps, optimal_hits = [], 0
        for sd in range(20):
            a = ag.select_action(s0, env, np.random.default_rng(sd))
            g = solver.optimality_gap(s0, a)
            gaps.append(g)
            if g < 1e-6:
                optimal_hits += 1
        gm, gci = mean_ci(gaps)
        print(f"  MCTS(N={n_roll:<3})        finds optimal {optimal_hits}/20 times | Avg Gap={gm:.5f} +/-{gci:.5f}")
        records.append({
            "case": "Case3_myopic_trap", "agent": f"MCTS(N={n_roll})",
            "return_mean": round(v_star - gm, 5), "return_ci": round(gci, 5),
            "saved_mean": 0.0, "note": f"optimal_hits={optimal_hits}/20, avg_gap={gm:.5f}",
        })


def main() -> None:
    records: list[dict] = []
    case_1_low_rollout_budget(records)
    case_2_model_mismatch(records)
    case_3_myopic_trap(records)

    out_dir = ROOT / "results" / "stage5b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "failure_cases_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved Stage 5b failure-case summary to {out_csv}")


if __name__ == "__main__":
    main()