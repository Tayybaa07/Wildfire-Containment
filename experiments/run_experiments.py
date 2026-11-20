"""STAGE 4-5 (Tayyba): experiment runner.

Run from the project root:   python experiments/run_experiments.py
Results (CSV, plots) go to results/, which is not committed to Git.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import wildfire` works

from wildfire import make_config, WildfireEnv, AGENTS, evaluate, mean_ci   # noqa: E402


def main():
    # TODO(Tayyba): sweep grid sizes / winds / rollout budgets, add the solver and MCTS
    # agents once they exist, write a CSV to results/, produce plots.
    raise NotImplementedError


if __name__ == "__main__":
    main()
