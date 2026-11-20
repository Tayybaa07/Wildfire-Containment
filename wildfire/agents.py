"""Baseline agents (Stage 1).

Every agent follows the SAME interface so the experiment runner (Stage 4) and
the other stages can use any agent without special cases:

    agent.select_action(state, env, rng) -> action   (a legal action tuple)

`rng` is the agent's own numpy Generator (used only for random choices).
"""
from __future__ import annotations

import numpy as np


class Agent:
    name = "agent"

    def select_action(self, state, env, rng: np.random.Generator):
        raise NotImplementedError


class NoOpAgent(Agent):
    """Sanity baseline: never builds a firebreak."""
    name = "NoOp"

    def select_action(self, state, env, rng):
        return ()


class RandomAgent(Agent):
    """B0: place k firebreaks uniformly at random among unburnt cells."""
    name = "B0-Random"

    def select_action(self, state, env, rng):
        unburnt = env.unburnt_cells(state)
        k_eff = min(env.cfg.k, len(unburnt))
        if env.is_terminal(state) or k_eff == 0:
            return ()
        chosen = rng.choice(unburnt, size=k_eff, replace=False)
        return tuple(sorted(int(c) for c in chosen))


class NearestFrontAgent(Agent):
    """B1: firebreaks on the unburnt cells closest (Manhattan) to any burning
    cell. Ties are broken by vegetation flammability (more flammable first),
    then by cell index, so this agent is deterministic."""
    name = "B1-NearestFront"

    def select_action(self, state, env, rng):
        burning = env.burning_cells(state)
        unburnt = env.unburnt_cells(state)
        k_eff = min(env.cfg.k, len(unburnt))
        if not burning or k_eff == 0:
            return ()

        def key(c):
            dist = min(env.manhattan(c, b) for b in burning)
            return (dist, -env.p_cell[c], c)

        ranked = sorted(unburnt, key=key)
        return tuple(sorted(ranked[:k_eff]))


class GreedyAgent(Agent):
    """B2: one-step greedy. For every legal action it computes the EXACT
    expected immediate reward and picks the best one (ties broken at random).

        E[r | a] = -(sum of ignition probs after a) - lam * |a|

    Because ignitions are independent, the ignition probabilities are computed
    once and the effect of each action is just removing its cells' terms.
    Short-sighted by design: it only sees the next turn."""
    name = "B2-Greedy"

    def select_action(self, state, env, rng):
        if env.is_terminal(state):
            return ()
        probs = env.ignition_probs(state)            # no action applied yet
        total = sum(probs.values())
        lam = env.cfg.lam

        best_val, best_actions = None, []
        for a in env.legal_actions(state):
            val = -(total - sum(probs.get(c, 0.0) for c in a)) - lam * len(a)
            val = round(val, 12)                     # avoid float-tie noise
            if best_val is None or val > best_val:
                best_val, best_actions = val, [a]
            elif val == best_val:
                best_actions.append(a)
        return best_actions[int(rng.integers(len(best_actions)))]


AGENTS = {
    "noop": NoOpAgent,
    "random": RandomAgent,
    "nearest": NearestFrontAgent,
    "greedy": GreedyAgent,
}
