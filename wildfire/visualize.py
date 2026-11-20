"""Console visualisation (the project is about AI reasoning, not graphics)."""
from __future__ import annotations

from .config import U, B, X, F, GRASS, SHRUB, FOREST

_SYMBOL = {U: "U", B: "B", X: "X", F: "F"}
_COLOR = {U: "\033[32m", B: "\033[31;1m", X: "\033[90m", F: "\033[34;1m"}
_RESET = "\033[0m"
_VEG_SYMBOL = {GRASS: "g", SHRUB: "s", FOREST: "t"}   # t = trees


def render_state(state, env, color: bool = False) -> str:
    """U/B/X/F grid. Set color=True for ANSI colours in a terminal."""
    lines = []
    for r in range(env.R):
        cells = []
        for c in range(env.C):
            v = state[env.idx(r, c)]
            ch = _SYMBOL[v]
            cells.append(f"{_COLOR[v]}{ch}{_RESET}" if color else ch)
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_vegetation(env) -> str:
    """g = grass, s = shrub, t = trees (forest), plus the wind direction."""
    lines = [" ".join(_VEG_SYMBOL[env.veg[env.idx(r, c)]] for c in range(env.C))
             for r in range(env.R)]
    lines.append(f"wind: {env.cfg.wind or 'none'}")
    return "\n".join(lines)
