"""Robustness tools (Stage 5a): planning with a wrong model, and mid-episode wind shifts.

Every agent plans with the environment object it is handed:
    action = agent.select_action(state, planner_env, rng)
while the fire itself follows another environment:
    state, reward, done = world_env.step(state, action, rng)

So a "wrong model" experiment is simply planner_env != world_env:
  * believed_env : what the agent thinks (e.g. wrong wind direction or spread probabilities)
  * true_env     : what really happens
  * shifted_env  : (optional) the world the fire follows from turn `shift_turn` on
                   (a wind shift). The agent keeps planning with believed_env unless an
                   `agent_after` is given, which models an agent that is TOLD about the shift.

All environments must share grid size and k (firebreaks per turn), so a state and an
action from one are valid in the others. Vegetation, wind, spread probabilities and lambda
may differ. Results use the same TrialResult / ScenarioSummary records as Stage 4.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agents import Agent
from .config import X
from .env import WildfireEnv
from .experiments import ScenarioSummary, TrialResult
from .simulate import mean_ci

AgentFactory = Callable[[WildfireEnv], Agent]      # builds an agent that plans with the given env


def check_compatible(a: WildfireEnv, b: WildfireEnv) -> None:
    """States and actions must be interchangeable between the two environments."""
    if (a.R, a.C) != (b.R, b.C):
        raise ValueError(f"Grid sizes differ: {a.R}x{a.C} vs {b.R}x{b.C}.")
    if a.cfg.k != b.cfg.k:
        raise ValueError(f"Firebreaks per turn differ: k={a.cfg.k} vs k={b.cfg.k}.")


def run_mismatch_episode(
    true_env: WildfireEnv,
    believed_env: WildfireEnv,
    agent: Agent,
    seed: int,
    scenario_id: str = "mismatch",
    shifted_env: Optional[WildfireEnv] = None,
    shift_turn: Optional[int] = None,
    agent_after: Optional[Agent] = None,
) -> TrialResult:
    """Play one episode. Random streams follow wildfire.simulate.run_episode (common random
    numbers): same seed -> same ignition cell and environment stream for every agent."""
    check_compatible(true_env, believed_env)
    if shifted_env is not None:
        check_compatible(true_env, shifted_env)
        if shift_turn is None or shift_turn < 0:
            raise ValueError("shift_turn must be a turn index >= 0 when shifted_env is given.")

    env_seq, agent_seq = np.random.SeedSequence(seed).spawn(2)
    env_rng = np.random.default_rng(env_seq)
    agent_rng = np.random.default_rng(agent_seq)

    state = true_env.reset(seed=seed)
    total, firebreaks, steps = 0.0, 0, 0
    decision_ms: List[float] = []
    tree_nodes: List[int] = []
    t_start = time.perf_counter()

    for t in range(true_env.horizon + 1):
        if true_env.is_terminal(state):
            break
        shifted = shifted_env is not None and t >= shift_turn
        world = shifted_env if shifted else true_env
        planner, actor = believed_env, agent
        if shifted and agent_after is not None:        # the agent is told about the shift
            planner, actor = shifted_env, agent_after

        d0 = time.perf_counter()
        action = actor.select_action(state, planner, agent_rng)
        decision_ms.append((time.perf_counter() - d0) * 1000.0)
        nodes = getattr(actor, "last_tree_nodes", None)
        if nodes is not None:
            tree_nodes.append(int(nodes))

        state, reward, _ = world.step(state, action, env_rng)
        total += reward
        firebreaks += len(action)
        steps += 1
    else:
        raise RuntimeError("Episode exceeded the horizon.")

    burnt = sum(1 for v in state if v == X)
    return TrialResult(
        scenario_id=scenario_id, agent=agent.name, seed=int(seed), return_=float(total),
        saved=int(true_env.n - burnt), burnt=int(burnt), firebreaks=int(firebreaks), steps=int(steps),
        time_total_ms=float((time.perf_counter() - t_start) * 1000.0),
        time_per_decision_ms=float(np.mean(decision_ms)) if decision_ms else 0.0,
        tree_nodes_per_decision=float(np.mean(tree_nodes)) if tree_nodes else None,
    )


def summarize(scenario_id: str, agent_name: str, trials: Sequence[TrialResult]) -> ScenarioSummary:
    """Aggregate one (scenario, agent) pair with the Stage 4 summary schema."""
    def ci(vals):
        return tuple(float(v) for v in mean_ci(vals))
    r, s, b = ci([t.return_ for t in trials]), ci([t.saved for t in trials]), ci([t.burnt for t in trials])
    f, tm = ci([t.firebreaks for t in trials]), ci([t.time_per_decision_ms for t in trials])
    nodes = [t.tree_nodes_per_decision for t in trials if t.tree_nodes_per_decision is not None]
    n = ci(nodes) if nodes else (None, None)
    return ScenarioSummary(
        scenario_id=scenario_id, agent=agent_name, n_episodes=len(trials),
        return_mean=r[0], return_ci=r[1], saved_mean=s[0], saved_ci=s[1],
        burnt_mean=b[0], burnt_ci=b[1], firebreaks_mean=f[0], firebreaks_ci=f[1],
        time_per_decision_ms_mean=tm[0], time_per_decision_ms_ci=tm[1],
        tree_nodes_mean=n[0], tree_nodes_ci=n[1],
        optimality_gap_mean=None, optimality_gap_ci=None,
        peak_memory_mb=float("nan"),                     # not measured in Stage 5a
    )


def run_mismatch_scenario(
    scenario_id: str,
    true_env: WildfireEnv,
    believed_env: WildfireEnv,
    agent_factories: Dict[str, AgentFactory],
    seeds: Sequence[int],
    shifted_env: Optional[WildfireEnv] = None,
    shift_turn: Optional[int] = None,
    agent_knows_shift: bool = False,
) -> Tuple[List[TrialResult], List[ScenarioSummary]]:
    """Run every agent on the same seeds. Each factory is called with the env the agent plans
    with (believed_env, and shifted_env too when agent_knows_shift is True)."""
    trials: List[TrialResult] = []
    summaries: List[ScenarioSummary] = []
    for factory in agent_factories.values():
        agent = factory(believed_env)
        after = factory(shifted_env) if (shifted_env is not None and agent_knows_shift) else None
        mine = [run_mismatch_episode(true_env, believed_env, agent, s, scenario_id,
                                     shifted_env, shift_turn, after) for s in seeds]
        trials.extend(mine)
        summaries.append(summarize(scenario_id, agent.name, mine))
    return trials, summaries


def paired_delta(trials: Sequence[TrialResult], scenario_id: str, reference_id: str) -> List[dict]:
    """Per agent: mean and 95% CI of (return in scenario - return in reference), paired by seed.
    Negative = the scenario is worse than the reference."""
    rows = []
    agents = sorted({t.agent for t in trials if t.scenario_id == scenario_id})
    for name in agents:
        a = {t.seed: t.return_ for t in trials if t.scenario_id == scenario_id and t.agent == name}
        b = {t.seed: t.return_ for t in trials if t.scenario_id == reference_id and t.agent == name}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        m, h = mean_ci([a[s] - b[s] for s in common])
        rows.append({"scenario_id": scenario_id, "reference_id": reference_id, "agent": name,
                     "n_paired": len(common), "delta_mean": float(m), "delta_ci": float(h)})
    return rows
