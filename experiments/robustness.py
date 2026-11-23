"""STAGE 5a (Zainab): robustness experiments (wrong model and wind shift).

Run from the project root:
    python experiments/robustness.py            # full run
    python experiments/robustness.py --quick    # tiny smoke test (seconds)

Question: how much does each method lose when the model it plans with is wrong?
The fire always follows the TRUE environment; the agent plans with a BELIEVED one.

Conditions (per scenario group):
  correct           believed = true                               (reference)
  belief_wind_N/W   the agent thinks the wind blows N / W (true wind is E)
  belief_wind_none  the agent thinks there is no wind
  belief_spread_x0.7 / x1.3   the agent under/over-estimates every spread probability
  shift_unaware     true wind shifts E -> N at --shift-turn (default turn 1 on 3x3, 2 on 5x5); the agent does not know
  shift_aware       same world, but the agent is told about the shift (control)

Raw results go to results/stage5a/ (not committed): the Stage 4 raw/summary CSVs plus
stage5a_conditions.csv (what each scenario_id means) and stage5a_degradation.csv (paired
changes in return vs the reference). Model-free agents (NoOp, Random, Nearest) cannot be
hurt by a wrong belief, which is a built-in sanity check.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import wildfire` works from any folder

from wildfire import (make_config, WildfireEnv, NoOpAgent, RandomAgent, NearestFrontAgent,   # noqa: E402
                      GreedyAgent, MCTSAgent, ExactAgent, ExactSolver, StateSpaceTooLargeError)
from wildfire.experiments import ExperimentSuiteResult                                          # noqa: E402
from wildfire.robustness import run_mismatch_scenario, paired_delta                             # noqa: E402

BASE_P = (0.6, 0.45, 0.3)              # default (grass, shrub, forest) ignition probabilities


def make_env(rows, cols, veg, veg_seed, wind, p_scale=1.0, k=1, lam=0.2):
    p = tuple(min(0.95, v * p_scale) for v in BASE_P)
    return WildfireEnv(make_config(rows, cols, veg, veg_seed=veg_seed, wind=wind, p_veg=p, k=k, lam=lam))


def exact_factory(env):
    """ExactAgent for the env it plans with, pre-solved for every single-cell ignition."""
    solver = ExactSolver(env)
    try:
        for c in range(env.n):
            solver.solve_backward_dp(initial_state=env.reset(ignition=(c,)), max_states=80000, time_limit=15.0)
    except StateSpaceTooLargeError:
        pass                            # falls back to solving on the fly
    return ExactAgent(solver)


def conditions(rows, cols, veg, veg_seed, true_wind, shift_turn, k, lam):
    """List of dicts: name, believed env, optional shifted env, whether the agent knows."""
    mk = lambda wind, scale=1.0: make_env(rows, cols, veg, veg_seed, wind, scale, k, lam)
    true = mk(true_wind)
    shifted = mk("N")
    out = [dict(name="correct", true=true, believed=true, ref=None)]
    for w in ("N", "W", None):
        out.append(dict(name=f"belief_wind_{w or 'none'}", true=true, believed=mk(w), ref="correct"))
    for s in (0.7, 1.3):
        out.append(dict(name=f"belief_spread_x{s}", true=true, believed=mk(true_wind, s), ref="correct"))
    out.append(dict(name="shift_aware", true=true, believed=true, shifted=shifted, turn=shift_turn,
                    knows=True, ref=None))
    out.append(dict(name="shift_unaware", true=true, believed=true, shifted=shifted, turn=shift_turn,
                    knows=False, ref="shift_aware"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--seeds-small", type=int, default=100, help="episodes per agent on the 3x3 grid")
    ap.add_argument("--seeds-large", type=int, default=40, help="episodes per agent on the 5x5 grid")
    ap.add_argument("--shift-turn", type=int, default=None,
                    help="turn at which the wind shifts E -> N (default: 1 on 3x3, 2 on 5x5)")
    ap.add_argument("--out", default=str(ROOT / "results" / "stage5a"))
    args = ap.parse_args()
    n_small, n_large = (8, 6) if args.quick else (args.seeds_small, args.seeds_large)
    mcts_small, mcts_large = (20, 20) if args.quick else (150, 100)

    groups = [  # (id, rows, cols, vegetation, veg_seed, seeds, default shift turn, agents)
        ("3x3_mixed_E", 3, 3, "mixed", 0, n_small, 1, {
            "NoOp": lambda e: NoOpAgent(), "Random": lambda e: RandomAgent(),
            "Nearest": lambda e: NearestFrontAgent(), "Greedy": lambda e: GreedyAgent(),
            "MCTS": lambda e: MCTSAgent(n_rollouts=mcts_small, rollout_policy="nearest"),
            "Exact": exact_factory}),
        ("5x5_mixed_E", 5, 5, "mixed", 0, n_large, 2, {
            "NoOp": lambda e: NoOpAgent(), "Random": lambda e: RandomAgent(),
            "Nearest": lambda e: NearestFrontAgent(), "Greedy": lambda e: GreedyAgent(),
            "MCTS": lambda e: MCTSAgent(n_rollouts=mcts_large, rollout_policy="nearest")}),
    ]

    suite = ExperimentSuiteResult(name="stage5a_robustness")
    meta, delta_rows = [], []
    for gid, r, c, veg, vs, n, turn, agents in groups:
        seeds = list(range(n))
        print(f"\n=== {gid}: {n} seeds per agent ===")
        for cond in conditions(r, c, veg, vs, "E", args.shift_turn if args.shift_turn is not None else turn, k=1, lam=0.2):
            sid = f"{gid}__{cond['name']}"
            t, s = run_mismatch_scenario(sid, cond["true"], cond["believed"], agents, seeds,
                                         cond.get("shifted"), cond.get("turn"), cond.get("knows", False))
            suite.trials += t
            suite.summaries += s
            meta.append(dict(scenario_id=sid, group=gid, condition=cond["name"], reference=cond["ref"],
                             true_wind="E", believed_wind=cond["believed"].cfg.wind,
                             believed_p_veg=cond["believed"].cfg.p_veg,
                             shift_to="N" if cond.get("shifted") else None, shift_turn=cond.get("turn"),
                             agent_knows_shift=cond.get("knows")))
            if cond["ref"]:
                delta_rows += paired_delta(suite.trials, sid, f"{gid}__{cond['ref']}")
            print(f"  {cond['name']:<20}" + "  ".join(f"{x.agent.split('(')[0]}={x.return_mean:6.2f}" for x in s))

    out = Path(args.out)
    paths = suite.save(str(out))
    for name, rows in (("stage5a_conditions.csv", meta), ("stage5a_degradation.csv", delta_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\nChange in mean return vs the reference (paired by seed; negative = worse):")
    for row in delta_rows:
        if any(k in row["agent"] for k in ("Greedy", "MCTS", "Exact")):
            print(f"  {row['scenario_id']:<42}{row['agent']:<24}{row['delta_mean']:+.2f} +/- {row['delta_ci']:.2f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
