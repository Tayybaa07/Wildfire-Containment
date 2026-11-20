"""Episode runner and small evaluation helpers.

Common random numbers: for a given `seed`, the ignition cell and the
environment's random stream are the same for every agent. The agent gets its
own independent stream, so changing the agent does not change the env seed.
"""
from __future__ import annotations

import numpy as np

from .config import X


def run_episode(env, agent, seed: int, verbose: bool = False) -> dict:
    env_seq, agent_seq = np.random.SeedSequence(seed).spawn(2)
    env_rng = np.random.default_rng(env_seq)
    agent_rng = np.random.default_rng(agent_seq)

    state = env.reset(seed=seed)
    total, firebreaks, steps = 0.0, 0, 0
    if verbose:
        from .visualize import render_state
        print(f"--- {agent.name}, seed {seed}: start ---")
        print(render_state(state, env))

    for _ in range(env.horizon + 1):                 # safeguard: T = R*C
        if env.is_terminal(state):
            break
        action = agent.select_action(state, env, agent_rng)
        state, reward, _ = env.step(state, action, env_rng)
        total += reward
        firebreaks += len(action)
        steps += 1
        if verbose:
            print(f"\nturn {steps}: action={action} reward={reward:.2f}")
            print(render_state(state, env))
    else:
        raise RuntimeError("Episode exceeded the horizon (should be impossible).")

    burnt = env.count(state, X)                      # includes the ignition cell(s)
    return {"return": total, "burnt": burnt, "saved": env.n - burnt,
            "firebreaks": firebreaks, "steps": steps}


def evaluate(env, agent, seeds) -> list:
    return [run_episode(env, agent, s) for s in seeds]


def mean_ci(values):
    """Mean and 95% confidence half-width (normal approximation)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
