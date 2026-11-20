"""Quick comparison of the baselines.   Example:
    python demo.py --rows 5 --cols 5 --wind E --episodes 500
    python demo.py --show greedy --seed 3        # watch one episode
"""
import argparse

from wildfire import make_config, WildfireEnv, AGENTS, evaluate, run_episode, mean_ci
from wildfire.visualize import render_vegetation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--veg", default="mixed", choices=["grass", "shrub", "forest", "mixed"])
    ap.add_argument("--wind", default=None, choices=[None, "N", "E", "S", "W"])
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="first episode seed / veg seed")
    ap.add_argument("--show", choices=list(AGENTS), help="print one episode for this agent")
    args = ap.parse_args()

    cfg = make_config(args.rows, args.cols, args.veg, veg_seed=args.seed,
                      wind=args.wind, k=args.k, lam=args.lam)
    env = WildfireEnv(cfg)
    print(render_vegetation(env), "\n")

    if args.show:
        run_episode(env, AGENTS[args.show](), args.seed, verbose=True)
        return

    seeds = range(args.seed, args.seed + args.episodes)
    print(f"{'agent':<18}{'return':>16}{'cells saved':>16}{'firebreaks':>12}")
    for key, cls in AGENTS.items():
        agent = cls()
        res = evaluate(env, agent, seeds)
        r, rci = mean_ci([x["return"] for x in res])
        s, sci = mean_ci([x["saved"] for x in res])
        fb = sum(x["firebreaks"] for x in res) / len(res)
        print(f"{agent.name:<18}{r:>7.2f} +/-{rci:<5.2f}{s:>7.2f} +/-{sci:<5.2f}{fb:>12.2f}")


if __name__ == "__main__":
    main()
