"""Monte Carlo Tree Search (UCT) agent (Stage 3).

Builds a search tree from the current state using `n_rollouts` simulations:
  1. Selection       - walk down the tree using UCB1 (normalized returns in [0, 1])
  2. Expansion       - try one new untried action and sample a next state
  3. Rollout         - finish the pretend game using a fast policy (random or B1)
  4. Backpropagation - send the total score back up to update visit counts & averages
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agents import Agent, NearestFrontAgent, RandomAgent
from .env import Action, State, WildfireEnv


class _MCTSNode:
    """One board state inside the search tree."""

    def __init__(self, state: State, env: WildfireEnv):
        self.state = state
        self.visits: int = 0
        # Actions we have NOT tried yet from this state
        self.untried_actions: List[Action] = (
            [] if env.is_terminal(state) else list(env.legal_actions(state))
        )
        # Stats per tried action: N(s, a) and sum of returns
        self.action_visits: Dict[Action, int] = {}
        self.action_value_sum: Dict[Action, float] = {}
        # Stochastic chance branching: children[action][next_state] -> child node
        self.children: Dict[Action, Dict[State, _MCTSNode]] = {}

    def get_or_add_child(
        self, action: Action, next_state: State, env: WildfireEnv
    ) -> Tuple["_MCTSNode", bool]:
        """Return child node for (action, next_state), creating it if new."""
        outcomes = self.children.setdefault(action, {})
        if next_state in outcomes:
            return outcomes[next_state], False
        child = _MCTSNode(next_state, env)
        outcomes[next_state] = child
        return child, True


class MCTSAgent(Agent):
    """UCT Monte Carlo Tree Search agent."""

    def __init__(
        self,
        n_rollouts: int = 200,
        c: float = 1.4,
        rollout_policy: str = "nearest",
        name: Optional[str] = None,
    ):
        if n_rollouts < 1:
            raise ValueError("n_rollouts must be >= 1.")
        if rollout_policy not in ("random", "nearest", "heuristic"):
            raise ValueError("rollout_policy must be 'random' or 'nearest' ('heuristic').")

        self.n_rollouts = n_rollouts
        self.c = c
        self.rollout_policy = rollout_policy
        self.name = name or f"MCTS(N={n_rollouts}, {rollout_policy})"

        # Fast rollout helper using Zainab's Stage 1 agents
        self._rollout_agent = (
            RandomAgent() if rollout_policy == "random" else NearestFrontAgent()
        )

        # Stats saved from the most recent decision (useful for Stage 4 & 6a plots)
        self.last_tree_nodes: int = 0

    def _ucb1_action(
        self, node: _MCTSNode, min_ret: float, max_ret: float
    ) -> Action:
        """Pick the tried action with highest UCB1 score."""
        span = max_ret - min_ret
        log_parent = math.log(node.visits)

        best_score = -float("inf")
        best_action: Action = ()

        for action, n_sa in node.action_visits.items():
            mean_q = node.action_value_sum[action] / n_sa
            # Normalize return to [0, 1] so exploration constant `c` works on any grid size
            norm_q = (mean_q - min_ret) / span if span > 0 else 0.0
            norm_q = max(0.0, min(1.0, norm_q))

            explore_bonus = self.c * math.sqrt(log_parent / n_sa)
            score = norm_q + explore_bonus

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _rollout(self, state: State, env: WildfireEnv, rng: np.random.Generator) -> float:
        """Play out the rest of a pretend game from `state` to the end."""
        sim_state = state
        total_reward = 0.0
        while not env.is_terminal(sim_state):
            action = self._rollout_agent.select_action(sim_state, env, rng)
            sim_state, reward, _ = env.step(sim_state, action, rng)
            total_reward += reward
        return total_reward

    def select_action(
        self,
        state: State,
        env: WildfireEnv,
        rng: Optional[np.random.Generator] = None,
    ) -> Action:
        if env.is_terminal(state):
            self.last_tree_nodes = 0
            return ()

        if rng is None:
            rng = np.random.default_rng()

        # Bounds used to normalize returns into [0, 1] for UCB1
        max_ret = 0.0
        min_ret = -float(env.n) * max(1.0, env.cfg.lam)

        root = _MCTSNode(state, env)
        tree_nodes = 1

        for _ in range(self.n_rollouts):
            node = root
            path: List[Tuple[_MCTSNode, Action, float]] = []

            # 1. SELECTION: follow UCB1 down while all actions have been tried
            while not env.is_terminal(node.state) and not node.untried_actions:
                action = self._ucb1_action(node, min_ret, max_ret)
                next_state, reward, _ = env.step(node.state, action, rng)
                path.append((node, action, reward))

                node, is_new = node.get_or_add_child(action, next_state, env)
                if is_new:
                    tree_nodes += 1
                    break

            # 2. EXPANSION: try one brand-new action if available
            if not env.is_terminal(node.state) and node.untried_actions:
                idx = int(rng.integers(len(node.untried_actions)))
                # Fast pop without shifting the whole list
                node.untried_actions[idx], node.untried_actions[-1] = (
                    node.untried_actions[-1],
                    node.untried_actions[idx],
                )
                action = node.untried_actions.pop()

                next_state, reward, _ = env.step(node.state, action, rng)
                path.append((node, action, reward))

                node, is_new = node.get_or_add_child(action, next_state, env)
                if is_new:
                    tree_nodes += 1

            # 3. ROLLOUT: simulate from `node` until the fire ends
            g = self._rollout(node.state, env, rng)

            # 4. BACKPROPAGATION: walk back up the path and update stats
            for parent, action, step_reward in reversed(path):
                g = step_reward + g
                parent.visits += 1
                parent.action_visits[action] = parent.action_visits.get(action, 0) + 1
                parent.action_value_sum[action] = (
                    parent.action_value_sum.get(action, 0.0) + g
                )

        self.last_tree_nodes = tree_nodes

        # Pick the root action visited the most (tie-break by highest average return)
        best_action = max(
            root.action_visits,
            key=lambda a: (
                root.action_visits[a],
                root.action_value_sum[a] / root.action_visits[a],
            ),
        )
        return best_action