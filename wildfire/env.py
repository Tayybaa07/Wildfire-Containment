"""Wildfire containment environment (the MDP).

STATE   : flat tuple of length R*C, each entry in {U, B, X, F}  (hashable, so
          it can be used as a dictionary key by the value-iteration solver).
ACTION  : tuple of flat cell indices (sorted). () is the no-op. Otherwise
          exactly min(k, n_unburnt) distinct UNBURNT cells become firebreaks.
STEP    : step(state, action, rng) -> (next_state, reward, done)
REWARD  : -(cells that caught fire this turn) - lam * (firebreaks placed)

Turn order (as in the proposal):
  1. firebreaks are placed (selected U cells -> F)
  2. every U cell with >= 1 burning neighbour ignites independently with
         P = 1 - prod_j (1 - q_j),   q_j = min(max_p, p_veg(cell) * wind(j->cell))
  3. cells that were burning become X; newly ignited cells become B.
Burning lasts exactly one turn, so statuses only move forward and the state
graph is acyclic: an episode always ends in at most R*C steps.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, U, B, X, F

State = Tuple[int, ...]
Action = Tuple[int, ...]

# (d_row, d_col) the wind blows TOWARDS
_WIND_VEC = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

# transition_probs enumerates 2^m outcomes for m at-risk cells; refuse huge m
MAX_ENUM_AT_RISK = 20


class WildfireEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.R, self.C = cfg.rows, cfg.cols
        self.n = self.R * self.C
        self.horizon = self.n                      # safeguard T = R*C

        # flat vegetation and base ignition probability per cell
        self.veg = tuple(cfg.vegetation[r][c]
                         for r in range(self.R) for c in range(self.C))
        self.p_cell = tuple(cfg.p_veg[v] for v in self.veg)

        self.neighbors = tuple(self._neighbors_of(i) for i in range(self.n))
        # edge_p[i][j] = prob. that burning neighbour j ignites cell i
        self.edge_p = tuple(
            {j: min(cfg.max_p, self.p_cell[i] * self._wind_multiplier(j, i))
             for j in self.neighbors[i]}
            for i in range(self.n))

    # ------------------------------------------------------------------ geometry
    def rc(self, i: int) -> Tuple[int, int]:
        return divmod(i, self.C)

    def idx(self, r: int, c: int) -> int:
        return r * self.C + c

    def _neighbors_of(self, i: int) -> List[int]:
        r, c = self.rc(i)
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.R and 0 <= cc < self.C:
                out.append(self.idx(rr, cc))
        return out

    def manhattan(self, i: int, j: int) -> int:
        (r1, c1), (r2, c2) = self.rc(i), self.rc(j)
        return abs(r1 - r2) + abs(c1 - c2)

    def _wind_multiplier(self, j: int, i: int) -> float:
        """Multiplier for fire spreading from burning cell j to cell i."""
        wind = self.cfg.wind
        if wind is None:
            return 1.0
        (rj, cj), (ri, ci) = self.rc(j), self.rc(i)
        wr, wc = _WIND_VEC[wind]
        dot = (ri - rj) * wr + (ci - cj) * wc      # +1 downwind, -1 upwind, 0 cross
        down, cross, up = self.cfg.wind_factors
        return down if dot > 0 else (up if dot < 0 else cross)

    # ------------------------------------------------------------------ state helpers
    def reset(self, seed=None, ignition=None) -> State:
        """Initial state. Ignition: argument > config > random cell from seed."""
        cells = ignition if ignition is not None else self.cfg.ignition
        if cells is None:
            cells = (int(np.random.default_rng(seed).integers(self.n)),)
        s = [U] * self.n
        for c in cells:
            s[c] = B
        return tuple(s)

    def cells(self, state: State, code: int) -> List[int]:
        return [i for i, v in enumerate(state) if v == code]

    def unburnt_cells(self, state: State) -> List[int]:
        return self.cells(state, U)

    def burning_cells(self, state: State) -> List[int]:
        return self.cells(state, B)

    def count(self, state: State, code: int) -> int:
        return sum(1 for v in state if v == code)

    def is_terminal(self, state: State) -> bool:
        return B not in state

    def cells_saved(self, state: State) -> int:
        """Cells that never burnt (still U, or protected F)."""
        return self.count(state, U) + self.count(state, F)

    # ------------------------------------------------------------------ actions
    def legal_actions(self, state: State) -> List[Action]:
        """No-op plus every set of min(k, n_unburnt) distinct unburnt cells.
        Size = C(n_U, k) + 1 (as in the proposal)."""
        if self.is_terminal(state):
            return [()]
        unburnt = self.unburnt_cells(state)
        k_eff = min(self.cfg.k, len(unburnt))
        actions: List[Action] = [()]
        if k_eff > 0:
            actions.extend(combinations(unburnt, k_eff))
        return actions

    def is_legal(self, state: State, action: Action) -> bool:
        if len(action) == 0:
            return True
        if self.is_terminal(state):
            return False
        if len(set(action)) != len(action):
            return False
        if len(action) != min(self.cfg.k, self.count(state, U)):
            return False
        return all(0 <= c < self.n and state[c] == U for c in action)

    # ------------------------------------------------------------------ dynamics
    def _apply_action(self, state: State, action: Action) -> List[int]:
        s = list(state)
        for c in action:
            s[c] = F
        return s

    def _ignition_probs(self, s: List[int]) -> Dict[int, float]:
        """P(ignite) for every unburnt cell with >= 1 burning neighbour."""
        probs: Dict[int, float] = {}
        for i in range(self.n):
            if s[i] != U:
                continue
            survive, at_risk = 1.0, False
            for j, q in self.edge_p[i].items():
                if s[j] == B:
                    survive *= (1.0 - q)
                    at_risk = True
            if at_risk:
                probs[i] = 1.0 - survive
        return probs

    def ignition_probs(self, state: State, action: Action = ()) -> Dict[int, float]:
        """Ignition probability of each at-risk cell AFTER applying `action`."""
        return self._ignition_probs(self._apply_action(state, action))

    def _reward(self, n_new: int, action: Action) -> float:
        return -float(n_new) - self.cfg.lam * len(action)

    def step(self, state: State, action: Action, rng: np.random.Generator):
        """Sample one turn. Returns (next_state, reward, done)."""
        action = tuple(action)
        if not self.is_legal(state, action):
            raise ValueError(f"Illegal action {action} in state {state}")
        if self.is_terminal(state):
            return state, 0.0, True

        s = self._apply_action(state, action)
        probs = self._ignition_probs(s)
        newly = [i for i, p in probs.items() if rng.random() < p]   # ascending i

        nxt = [X if v == B else v for v in s]
        for i in newly:
            nxt[i] = B
        nxt = tuple(nxt)
        return nxt, self._reward(len(newly), action), self.is_terminal(nxt)

    def transition_probs(self, state: State, action: Action):
        """Exact transition distribution: list of (prob, next_state, reward).
        Used by the value-iteration solver. Enumerates 2^m outcomes where m is
        the number of at-risk cells after the action."""
        action = tuple(action)
        if not self.is_legal(state, action):
            raise ValueError(f"Illegal action {action} in state {state}")
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]

        s = self._apply_action(state, action)
        probs = self._ignition_probs(s)
        at_risk = list(probs)
        if len(at_risk) > MAX_ENUM_AT_RISK:
            raise ValueError(f"{len(at_risk)} at-risk cells: too many to enumerate.")

        base = [X if v == B else v for v in s]
        outcomes = []
        for flips in product((0, 1), repeat=len(at_risk)):
            p, nxt, n_new = 1.0, list(base), 0
            for cell, flip in zip(at_risk, flips):
                p *= probs[cell] if flip else 1.0 - probs[cell]
                if flip:
                    nxt[cell] = B
                    n_new += 1
            if p > 0.0:
                outcomes.append((p, tuple(nxt), self._reward(n_new, action)))
        return outcomes
