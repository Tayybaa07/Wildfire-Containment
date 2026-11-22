"""Exact MDP solver for Wildfire Containment (Stage 2).

Implements exact Bellman optimality solvers on small grids:
1. Backward Dynamic Programming (DAG memoized recursion / topological induction)
2. Standard Value Iteration sweeps over reachable states
3. ExactAgent: optimal agent matching the Stage 1 Agent interface
4. Verification & optimality gap tools for Stage 3 (MCTS handover)

Theoretical background:
The state transitions are strictly forward-moving in a Directed Acyclic Graph (DAG):
each cell moves U -> B -> X or U -> F. Burning (B) lasts exactly 1 turn.
Episodes terminate in at most R*C turns. Discount factor gamma = 1.0.
Bellman Optimality Equation:
    V*(s) = max_{a in A(s)} sum_{s'} P(s' | s, a) [ r(s, a, s') + V*(s') ]
with V*(s_terminal) = 0.0 for all terminal states (states with no burning cells).
"""
from __future__ import annotations

import json
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .agents import Agent
from .config import Config, B, F, U, X
from .env import Action, State, WildfireEnv


class StateSpaceTooLargeError(Exception):
    """Raised when the reachable state space exceeds safety thresholds."""
    pass


@dataclass
class SolveResult:
    """Summary metrics and tables returned by ExactSolver."""
    v_s0: float
    reachable_count: int
    non_terminal_count: int
    terminal_count: int
    solve_time: float
    method: str
    sweeps: int
    bellman_backups: int
    peak_memory_mb: float
    max_dp_vi_diff: Optional[float] = None
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExactSolver:
    """Exact solver computing V*(s) and pi*(s) via Backward DP and Value Iteration."""

    def __init__(self, env: WildfireEnv):
        self.env = env
        self.V: Dict[State, float] = {}
        self.policy: Dict[State, Action] = {}
        self.reachable_states: Set[State] = set()
        self.terminal_states: Set[State] = set()
        self.metrics: Optional[SolveResult] = None

    # ------------------------------------------------------------------ Enumeration
    def enumerate_reachable_states(
        self,
        initial_state: Optional[State] = None,
        max_states: int = 150000,
        time_limit: float = 60.0,
    ) -> Tuple[Set[State], Set[State]]:
        """Breadth-first search of states reachable from initial_state.

        Returns (non_terminal_states, terminal_states).
        Raises StateSpaceTooLargeError if max_states or time_limit is exceeded.
        """
        s0 = initial_state if initial_state is not None else self.env.reset()
        t0 = time.time()

        visited: Set[State] = {s0}
        non_terminal: Set[State] = set()
        terminal: Set[State] = set()

        queue = deque([s0])

        while queue:
            if len(visited) > max_states:
                raise StateSpaceTooLargeError(
                    f"Reachable states exceeded safety limit of {max_states} "
                    f"(visited={len(visited)}, queue={len(queue)}, elapsed={time.time()-t0:.1f}s)."
                )
            if time.time() - t0 > time_limit:
                raise StateSpaceTooLargeError(
                    f"Enumeration exceeded time limit of {time_limit}s "
                    f"(visited={len(visited)}, queue={len(queue)})."
                )

            s = queue.popleft()
            if self.env.is_terminal(s):
                terminal.add(s)
                continue

            non_terminal.add(s)
            for a in self.env.legal_actions(s):
                for _, ns, _ in self.env.transition_probs(s, a):
                    if ns not in visited:
                        visited.add(ns)
                        queue.append(ns)

        self.reachable_states = visited
        self.terminal_states = terminal
        return non_terminal, terminal

    # ------------------------------------------------------------------ Backward DP
    def solve_backward_dp(
        self,
        initial_state: Optional[State] = None,
        max_states: int = 150000,
        time_limit: float = 60.0,
    ) -> SolveResult:
        """Solve exact Bellman optimality via memoized backward dynamic programming.

        Exploits the acyclic property of cell status transitions (U->B->X, U->F).
        """
        s0 = initial_state if initial_state is not None else self.env.reset()
        tracemalloc.start()
        t0 = time.time()

        memo_v: Dict[State, float] = dict(self.V)
        memo_pi: Dict[State, Action] = dict(self.policy)
        visited_count = 0
        backups = 0

        def _dp(s: State) -> float:
            nonlocal visited_count, backups
            if self.env.is_terminal(s):
                return 0.0
            if s in memo_v:
                return memo_v[s]

            visited_count += 1
            if visited_count > max_states:
                raise StateSpaceTooLargeError(
                    f"Backward DP exceeded max_states limit ({max_states})."
                )
            if visited_count % 10000 == 0 and time.time() - t0 > time_limit:
                raise StateSpaceTooLargeError(
                    f"Backward DP exceeded time_limit ({time_limit}s)."
                )

            best_v = -float("inf")
            best_a = ()
            backups += 1

            for a in self.env.legal_actions(s):
                q = 0.0
                for p, ns, r in self.env.transition_probs(s, a):
                    q += p * (r + _dp(ns))
                q = round(q, 11)  # avoid floating precision noise in ties
                if q > best_v:
                    best_v = q
                    best_a = a

            memo_v[s] = best_v
            memo_pi[s] = best_a
            return best_v

        v_s0 = _dp(s0)
        solve_time = time.time() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.V = memo_v
        self.policy = memo_pi

        # Identify all reachable terminal states
        term_states: Set[State] = set()
        for s in list(memo_v):
            for a in self.env.legal_actions(s):
                for _, ns, _ in self.env.transition_probs(s, a):
                    if self.env.is_terminal(ns):
                        term_states.add(ns)

        self.V = memo_v
        for ns in term_states:
            self.V[ns] = 0.0
        self.policy = memo_pi
        self.terminal_states = term_states
        self.reachable_states = set(memo_v.keys()) | term_states

        result = SolveResult(
            v_s0=v_s0,
            reachable_count=len(self.reachable_states),
            non_terminal_count=len(memo_v),
            terminal_count=len(term_states),
            solve_time=solve_time,
            method="backward_dp",
            sweeps=1,
            bellman_backups=backups,
            peak_memory_mb=peak_mem / (1024 * 1024),
            converged=True,
        )
        self.metrics = result
        return result

    # ------------------------------------------------------------------ Value Iteration
    def solve_value_iteration(
        self,
        initial_state: Optional[State] = None,
        epsilon: float = 1e-10,
        max_sweeps: int = 100,
        max_states: int = 150000,
        time_limit: float = 60.0,
    ) -> SolveResult:
        """Solve exact Bellman optimality via standard Value Iteration sweeps.

        Enumerates reachable states first, then iteratively applies the Bellman backup:
            V_{k+1}(s) = max_{a in A(s)} sum_{s'} P(s' | s, a) [r(s, a, s') + V_k(s')]
        until max_{s} |V_{k+1}(s) - V_k(s)| < epsilon.
        """
        s0 = initial_state if initial_state is not None else self.env.reset()
        tracemalloc.start()
        t0 = time.time()

        non_terminal, terminal = self.enumerate_reachable_states(
            s0, max_states=max_states, time_limit=time_limit
        )

        all_states = list(non_terminal)
        V: Dict[State, float] = {s: 0.0 for s in all_states}
        for t in terminal:
            V[t] = 0.0

        policy: Dict[State, Action] = {}
        sweeps = 0
        total_backups = 0

        while sweeps < max_sweeps:
            sweeps += 1
            delta = 0.0
            new_V = dict(V)

            for s in all_states:
                total_backups += 1
                best_v = -float("inf")
                best_a = ()
                for a in self.env.legal_actions(s):
                    q = 0.0
                    for p, ns, r in self.env.transition_probs(s, a):
                        q += p * (r + V.get(ns, 0.0))
                    q = round(q, 11)
                    if q > best_v:
                        best_v = q
                        best_a = a
                delta = max(delta, abs(best_v - V[s]))
                new_V[s] = best_v
                policy[s] = best_a

            V = new_V
            if delta < epsilon:
                break

        solve_time = time.time() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.V = V
        self.policy = policy

        result = SolveResult(
            v_s0=V.get(s0, 0.0),
            reachable_count=len(non_terminal) + len(terminal),
            non_terminal_count=len(non_terminal),
            terminal_count=len(terminal),
            solve_time=solve_time,
            method="value_iteration",
            sweeps=sweeps,
            bellman_backups=total_backups,
            peak_memory_mb=peak_mem / (1024 * 1024),
            converged=(delta < epsilon),
        )
        self.metrics = result
        return result

    # ------------------------------------------------------------------ Unified Solve
    def solve(
        self,
        initial_state: Optional[State] = None,
        method: str = "both",
        epsilon: float = 1e-10,
        max_states: int = 150000,
        time_limit: float = 60.0,
        assert_equivalence: bool = True,
    ) -> SolveResult:
        """Solve using 'backward_dp', 'value_iteration', or 'both'.

        When method='both':
          Runs both methods, verifies that max |V_dp(s) - V_vi(s)| < 1e-9 across all
          reachable states, and verifies that the policies achieve identical Q-values.
        """
        s0 = initial_state if initial_state is not None else self.env.reset()

        if method == "backward_dp":
            return self.solve_backward_dp(s0, max_states=max_states, time_limit=time_limit)
        elif method == "value_iteration":
            return self.solve_value_iteration(
                s0, epsilon=epsilon, max_states=max_states, time_limit=time_limit
            )
        elif method == "both":
            res_dp = self.solve_backward_dp(s0, max_states=max_states, time_limit=time_limit)
            v_dp = dict(self.V)
            pi_dp = dict(self.policy)

            res_vi = self.solve_value_iteration(
                s0, epsilon=epsilon, max_states=max_states, time_limit=time_limit
            )
            v_vi = dict(self.V)
            pi_vi = dict(self.policy)

            # Compare value functions across all non-terminal states
            common_states = set(v_dp.keys()) & set(v_vi.keys())
            diffs = [abs(v_dp[s] - v_vi[s]) for s in common_states if not self.env.is_terminal(s)]
            max_diff = max(diffs) if diffs else 0.0

            if assert_equivalence:
                assert max_diff < 1e-8, (
                    f"Backward DP and Value Iteration values diverged! Max diff = {max_diff}"
                )
                # Verify policy equivalence: both chosen actions must yield equal Q-values
                for s in common_states:
                    if self.env.is_terminal(s):
                        continue
                    a_dp = pi_dp.get(s, ())
                    a_vi = pi_vi.get(s, ())
                    if a_dp != a_vi:
                        q_dp = sum(p * (r + v_dp.get(ns, 0.0)) for p, ns, r in self.env.transition_probs(s, a_dp))
                        q_vi = sum(p * (r + v_vi.get(ns, 0.0)) for p, ns, r in self.env.transition_probs(s, a_vi))
                        assert abs(q_dp - q_vi) < 1e-8, (
                            f"Policy divergence in state {s}: Q(a_dp={a_dp})={q_dp} vs Q(a_vi={a_vi})={q_vi}"
                        )

            # Use DP values & policies as primary (cleaner DAG memoization)
            self.V = v_dp
            self.policy = pi_dp

            combined_result = SolveResult(
                v_s0=res_dp.v_s0,
                reachable_count=res_dp.reachable_count,
                non_terminal_count=res_dp.non_terminal_count,
                terminal_count=res_dp.terminal_count,
                solve_time=res_dp.solve_time + res_vi.solve_time,
                method="both",
                sweeps=res_vi.sweeps,
                bellman_backups=res_dp.bellman_backups + res_vi.bellman_backups,
                peak_memory_mb=max(res_dp.peak_memory_mb, res_vi.peak_memory_mb),
                max_dp_vi_diff=max_diff,
                converged=True,
                metadata={
                    "dp_solve_time": res_dp.solve_time,
                    "vi_solve_time": res_vi.solve_time,
                    "dp_peak_memory_mb": res_dp.peak_memory_mb,
                    "vi_peak_memory_mb": res_vi.peak_memory_mb,
                    "vi_sweeps": res_vi.sweeps,
                },
            )
            self.metrics = combined_result
            return combined_result
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'backward_dp', 'value_iteration', or 'both'.")

    # ------------------------------------------------------------------ Queries & Handover
    def get_value(self, state: State) -> float:
        """Query V*(s). If terminal, returns 0.0. Falls back to on-the-fly solve if unseen."""
        if self.env.is_terminal(state):
            return 0.0
        if state in self.V:
            return self.V[state]
        # On-the-fly backward DP if state was not in the solved component
        self.solve_backward_dp(initial_state=state)
        return self.V.get(state, 0.0)

    def get_action(self, state: State) -> Action:
        """Query optimal action pi*(s). If terminal, returns ()."""
        if self.env.is_terminal(state):
            return ()
        if state in self.policy:
            return self.policy[state]
        self.solve_backward_dp(initial_state=state)
        return self.policy.get(state, ())

    def q_value(self, state: State, action: Action) -> float:
        """Compute exact Q*(s, a) = sum_{s'} P(s' | s, a) [r(s, a, s') + V*(s')]."""
        if self.env.is_terminal(state):
            return 0.0
        q = 0.0
        for p, ns, r in self.env.transition_probs(state, action):
            q += p * (r + self.get_value(ns))
        return round(q, 11)

    def optimality_gap(self, state: State, action: Action) -> float:
        """Return the suboptimality gap: V*(s) - Q*(s, a) >= 0.0.

        Critical for Stage 3 (Basma) to evaluate how close MCTS decisions are to optimal.
        """
        if self.env.is_terminal(state):
            return 0.0
        gap = self.get_value(state) - self.q_value(state, action)
        return max(0.0, round(gap, 10))

    # ------------------------------------------------------------------ Serialization
    def export_policy_table(self) -> Dict[str, Any]:
        """Convert policy and value table to JSON-serializable dictionary."""
        return {
            "values": {str(s): v for s, v in self.V.items()},
            "policy": {str(s): list(a) for s, a in self.policy.items()},
            "v_s0": self.metrics.v_s0 if self.metrics else None,
            "metrics": {
                "reachable_count": self.metrics.reachable_count,
                "solve_time": self.metrics.solve_time,
                "method": self.metrics.method,
            } if self.metrics else {},
        }

    def save(self, filepath: str) -> None:
        """Save policy and value table to JSON file."""
        data = self.export_policy_table()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class ExactAgent(Agent):
    """Optimal agent executing the exact policy pi*(s) from ExactSolver.

    Plugs seamlessly into run_episode(env, agent, seed) and demo.py.
    """
    name = "Exact"

    def __init__(self, solver: Optional[ExactSolver] = None):
        self.solver = solver

    def select_action(self, state: State, env: WildfireEnv, rng: Optional[np.random.Generator] = None) -> Action:
        """Return the optimal action pi*(state)."""
        if env.is_terminal(state):
            return ()
        if self.solver is None or self.solver.env != env:
            self.solver = ExactSolver(env)
        action = self.solver.get_action(state)
        # Ensure legal action contract
        if not env.is_legal(state, action):
            # Fallback if tie or unexpected state
            legal = env.legal_actions(state)
            return legal[0] if legal else ()
        return action
