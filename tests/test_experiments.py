"""Unit tests for Stage 4 Experiment Runner (wildfire/experiments.py)."""
import os
import json
import csv
import numpy as np
import pytest

from wildfire import (
    WildfireEnv,
    make_config,
    NoOpAgent,
    RandomAgent,
    NearestFrontAgent,
    GreedyAgent,
    ExactAgent,
    ExactSolver,
    MCTSAgent,
    run_monitored_episode,
    run_scenario,
    run_suite,
    TrialResult,
    ScenarioSummary,
    ExperimentSuiteResult,
)


def test_deterministic_reproducibility():
    """Running an experiment twice with the same seed sequence must yield identical results."""
    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, ignition=(0,))
    env = WildfireEnv(cfg)
    agent = MCTSAgent(n_rollouts=50, rollout_policy="nearest")

    res1 = [run_monitored_episode(env, agent, seed=s, compute_gaps=False) for s in range(5)]
    res2 = [run_monitored_episode(env, agent, seed=s, compute_gaps=False) for s in range(5)]

    for t1, t2 in zip(res1, res2):
        assert t1.seed == t2.seed
        assert t1.return_ == t2.return_
        assert t1.saved == t2.saved
        assert t1.burnt == t2.burnt
        assert t1.firebreaks == t2.firebreaks
        assert t1.steps == t2.steps


def test_common_random_numbers_across_agents():
    """Different agents evaluated on the same seed must receive the exact same initial state and env randomness."""
    cfg = make_config(4, 4, vegetation="mixed", wind="E", k=1)
    env = WildfireEnv(cfg)

    # env.reset(seed) must be identical for any agent on seed 17
    s_noop = env.reset(seed=17)
    s_greedy = env.reset(seed=17)
    s_mcts = env.reset(seed=17)
    assert s_noop == s_greedy == s_mcts

    # Step transition on same initial state and same action must be identical
    seq1_env, _ = np.random.SeedSequence(17).spawn(2)
    seq2_env, _ = np.random.SeedSequence(17).spawn(2)
    nxt1, r1, d1 = env.step(s_noop, (), np.random.default_rng(seq1_env))
    nxt2, r2, d2 = env.step(s_noop, (), np.random.default_rng(seq2_env))
    assert nxt1 == nxt2
    assert r1 == r2
    assert d1 == d2


def test_optimality_gap_logging_and_skipping():
    """Optimality gap must be 0.0 for ExactAgent, >= 0.0 for others, and None for large grids."""
    # 3x3 grid: solver exists
    cfg_3 = make_config(3, 3, vegetation="mixed", wind="E", k=1, ignition=(0,))
    env_3 = WildfireEnv(cfg_3)
    solver = ExactSolver(env_3)
    solver.solve(method="backward_dp")

    exact_res = run_monitored_episode(env_3, ExactAgent(solver), seed=1, solver=solver, compute_gaps=True)
    greedy_res = run_monitored_episode(env_3, GreedyAgent(), seed=1, solver=solver, compute_gaps=True)
    mcts_res = run_monitored_episode(env_3, MCTSAgent(n_rollouts=50), seed=1, solver=solver, compute_gaps=True)

    assert exact_res.optimality_gap == pytest.approx(0.0)
    assert greedy_res.optimality_gap is not None and greedy_res.optimality_gap >= 0.0
    assert mcts_res.optimality_gap is not None and mcts_res.optimality_gap >= 0.0

    # 6x6 grid: solver is None, gap should be None
    cfg_6 = make_config(6, 6, vegetation="mixed", wind="E", k=1)
    env_6 = WildfireEnv(cfg_6)
    large_res = run_monitored_episode(env_6, GreedyAgent(), seed=1, solver=None, compute_gaps=False)
    assert large_res.optimality_gap is None


def test_artifact_generation(tmp_path):
    """Test saving CSV and JSON raw trials and scenario summaries to disk."""
    cfg = make_config(3, 3, vegetation="grass", wind=None, k=1, ignition=(0,))
    env = WildfireEnv(cfg)
    agents = {
        "NoOp": NoOpAgent(),
        "Greedy": GreedyAgent(),
    }

    trials, summaries = run_scenario(
        scenario_id="test_scen",
        env=env,
        agents=agents,
        seeds=range(3),
        compute_gaps=True,
    )

    suite = ExperimentSuiteResult(name="test_suite", trials=trials, summaries=summaries)
    out_dir = str(tmp_path / "stage4_test")
    raw_csv, raw_json, sum_csv, sum_json = suite.save(out_dir)

    assert os.path.exists(raw_csv)
    assert os.path.exists(raw_json)
    assert os.path.exists(sum_csv)
    assert os.path.exists(sum_json)

    # Verify CSV headers and row count
    with open(raw_csv, "r") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 6  # 2 agents * 3 seeds
        assert "return" in reader[0]
        assert "time_per_decision_ms" in reader[0]

    with open(sum_csv, "r") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2  # 2 agents
        assert "return_mean" in reader[0]
        assert "peak_memory_mb" in reader[0]

    # Verify JSON content
    with open(raw_json, "r") as f:
        data = json.load(f)
        assert len(data) == 6

    with open(sum_json, "r") as f:
        data = json.load(f)
        assert len(data) == 2
