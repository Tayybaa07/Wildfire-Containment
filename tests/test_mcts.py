"""Unit tests for MCTSAgent (Stage 3)."""
import numpy as np
import pytest

from wildfire import (
    ExactSolver,
    MCTSAgent,
    RandomAgent,
    WildfireEnv,
    evaluate,
    make_config,
)


def test_mcts_legal_action_and_terminal():
    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, ignition=(0,))
    env = WildfireEnv(cfg)
    s0 = env.reset(seed=0)
    rng = np.random.default_rng(0)

    agent = MCTSAgent(n_rollouts=50)
    action = agent.select_action(s0, env, rng)
    assert env.is_legal(s0, action)
    assert agent.last_tree_nodes > 0

    # Terminal state (no burning cells) must return no-op ()
    terminal_state = (0,) * env.n
    assert agent.select_action(terminal_state, env, rng) == ()


def test_mcts_reproducibility():
    cfg = make_config(4, 4, vegetation="mixed", wind="E", k=1, ignition=(5,))
    env = WildfireEnv(cfg)
    s0 = env.reset(seed=1)

    a1 = MCTSAgent(n_rollouts=80).select_action(s0, env, np.random.default_rng(42))
    a2 = MCTSAgent(n_rollouts=80).select_action(s0, env, np.random.default_rng(42))
    assert a1 == a2


def test_mcts_rollout_policies_and_k2():
    cfg = make_config(3, 3, vegetation="mixed", wind="None" if False else None, k=2, ignition=(4,))
    env = WildfireEnv(cfg)
    s0 = env.reset(seed=0)

    for policy in ("random", "nearest", "heuristic"):
        agent = MCTSAgent(n_rollouts=60, rollout_policy=policy)
        action = agent.select_action(s0, env, np.random.default_rng(7))
        assert env.is_legal(s0, action)


def test_mcts_invalid_params():
    with pytest.raises(ValueError):
        MCTSAgent(n_rollouts=0)
    with pytest.raises(ValueError):
        MCTSAgent(rollout_policy="invalid")


def test_mcts_beats_random_on_3x3():
    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, ignition=(0,))
    env = WildfireEnv(cfg)

    rand_res = evaluate(env, RandomAgent(), seeds=range(40))
    mcts_res = evaluate(env, MCTSAgent(n_rollouts=100), seeds=range(40))

    mean_rand = float(np.mean([r["return"] for r in rand_res]))
    mean_mcts = float(np.mean([r["return"] for r in mcts_res]))
    assert mean_mcts > mean_rand


def test_mcts_small_optimality_gap_against_exact():
    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env = WildfireEnv(cfg)
    s0 = env.reset()

    solver = ExactSolver(env)
    solver.solve(method="backward_dp")

    agent = MCTSAgent(n_rollouts=200, rollout_policy="nearest")
    gaps = [
        solver.optimality_gap(s0, agent.select_action(s0, env, np.random.default_rng(seed)))
        for seed in range(5)
    ]
    assert np.mean(gaps) < 0.3