"""Configuration for the wildfire MDP.

Everything that defines a scenario (grid size, vegetation, wind, spread
probabilities, reward weight, number of firebreaks per turn) lives in ONE
frozen Config object, so experiments are easy to log and reproduce.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ---- Cell status codes (the MDP state is a flat tuple of these) -------------
U, B, X, F = 0, 1, 2, 3          # Unburnt, Burning, Burnt, Firebreak
STATUS_NAMES = {U: "Unburnt", B: "Burning", X: "Burnt", F: "Firebreak"}

# ---- Vegetation codes (fixed per scenario, NOT part of the state) -----------
GRASS, SHRUB, FOREST = 0, 1, 2
VEG_NAMES = {GRASS: "grass", SHRUB: "shrub", FOREST: "forest"}

WIND_DIRECTIONS = (None, "N", "E", "S", "W")


@dataclass(frozen=True)
class Config:
    rows: int
    cols: int
    # vegetation[r][c] in {GRASS, SHRUB, FOREST}
    vegetation: Tuple[Tuple[int, ...], ...]
    # Wind blows TOWARDS this direction (None = no wind).
    wind: Optional[str] = None
    # Base ignition probability per vegetation type: (grass, shrub, forest)
    p_veg: Tuple[float, float, float] = (0.6, 0.45, 0.3)
    # Wind multipliers: (downwind, crosswind, upwind)
    wind_factors: Tuple[float, float, float] = (1.5, 1.0, 0.5)
    # Cap on the spread probability along a single burning->target edge
    max_p: float = 0.95
    # Cost of one firebreak (lambda in the reward)
    lam: float = 0.2
    # Firebreaks placed per turn
    k: int = 1
    # Initial burning cells as flat indices; None = one random cell (seeded)
    ignition: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1:
            raise ValueError("Grid must have at least one row and one column.")
        if len(self.vegetation) != self.rows or any(
                len(row) != self.cols for row in self.vegetation):
            raise ValueError("vegetation must be a rows x cols table.")
        if any(v not in (GRASS, SHRUB, FOREST)
               for row in self.vegetation for v in row):
            raise ValueError("vegetation codes must be 0 (grass), 1 (shrub) or 2 (forest).")
        if self.wind not in WIND_DIRECTIONS:
            raise ValueError(f"wind must be one of {WIND_DIRECTIONS}.")
        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if self.lam < 0:
            raise ValueError("lam must be >= 0.")
        if not 0 < self.max_p <= 1:
            raise ValueError("max_p must be in (0, 1].")
        if self.ignition is not None:
            n = self.rows * self.cols
            if len(self.ignition) == 0 or any(not 0 <= c < n for c in self.ignition):
                raise ValueError("ignition cells must be valid flat indices.")


def make_vegetation(rows: int, cols: int, kind: str = "mixed", seed: int = 0):
    """Build a vegetation map. kind: 'grass' | 'shrub' | 'forest' | 'mixed'."""
    uniform = {"grass": GRASS, "shrub": SHRUB, "forest": FOREST}
    if kind in uniform:
        return tuple(tuple(uniform[kind] for _ in range(cols)) for _ in range(rows))
    if kind == "mixed":
        rng = np.random.default_rng(seed)
        grid = rng.integers(0, 3, size=(rows, cols))
        return tuple(tuple(int(v) for v in row) for row in grid)
    raise ValueError("kind must be 'grass', 'shrub', 'forest' or 'mixed'.")


def make_config(rows: int, cols: int, vegetation: str = "mixed",
                veg_seed: int = 0, **kwargs) -> Config:
    """Convenience builder: make_config(4, 4, 'mixed', wind='E', k=1)."""
    veg = make_vegetation(rows, cols, vegetation, veg_seed)
    return Config(rows=rows, cols=cols, vegetation=veg, **kwargs)
