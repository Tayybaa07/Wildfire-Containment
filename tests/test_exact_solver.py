"""Unit tests for Stage 2 Exact Solver (Backward DP and Value Iteration)."""
import numpy as np
import pytest

from wildfire import (
    Config,
    WildfireEnv,
    make_config,
    run_episode,
    evaluate,
    mean_ci,
    NoOpAgent,
    RandomAgent,
    NearestFrontAgent,
    GreedyAgent,
    ExactSolver,
    ExactAgent,
    StateSpaceTooLargeError,
    U, B, X, F, GRASS, FOREST,
)


# ---------------------------------------------------------------- Equivalence: DP vs VI
@pytest.mark.parametrize("wind", [None, "E"])
@pytest.mark.parametrize("veg", ["grass", "mixed"])
def test_dp_and_vi_produce_identical_values(wind, veg):
    """Verify memoized backward DP and iterative VI produce identical V*(s) values."""
    cfg = make_config(3, 3, vegetation=veg, wind=wind, k=1, lam=0.2, ignition=(4,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)

    # Solve using both methods and check mathematical equivalence
    res = solver.solve(method="both", epsilon=1e-10, assert_equivalence=True)

    assert res.converged
    assert res.max_dp_vi_diff is not None
    assert res.max_dp_vi_diff < 1e-8
    assert res.reachable_count > 0
    assert res.v_s0 < 0.0


# ---------------------------------------------------------------- Terminal States & Legality
def test_terminal_state_values_are_zero():
    """V*(s) must be 0 for all terminal states."""
    cfg = make_config(3, 3, vegetation="mixed", ignition=(0,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)
    solver.solve(method="backward_dp")

    for s, v in solver.V.items():
        if env.is_terminal(s):
            assert v == pytest.approx(0.0)

    # Directly query an arbitrary terminal state
    all_burnt = tuple([X] * 9)
    assert solver.get_value(all_burnt) == 0.0
    assert solver.get_action(all_burnt) == ()


def test_exact_agent_legal_action_compliance():
    """ExactAgent must return legal actions across every step of an episode."""
    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=2, lam=0.2)
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)
    agent = ExactAgent(solver)
    rng = np.random.default_rng(42)

    for seed in range(25):
        state = env.reset(seed=seed)
        for _ in range(env.horizon + 1):
            if env.is_terminal(state):
                break
            action = agent.select_action(state, env, rng)
            assert env.is_legal(state, action), f"Illegal action {action} in state {state}"
            state, _, _ = env.step(state, action, rng)


# ---------------------------------------------------------------- Monte Carlo Consistency
def test_vs0_matches_monte_carlo_rollout():
    """Theoretical V*(s0) must lie within the 95% CI of ExactAgent rollouts."""
    cfg = make_config(3, 3, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)
    res = solver.solve(method="backward_dp")
    v_theoretical = res.v_s0

    agent = ExactAgent(solver)
    seeds = range(1500)
    returns = [run_episode(env, agent, seed=s)["return"] for s in seeds]
    emp_mean, emp_ci = mean_ci(returns)

    assert abs(emp_mean - v_theoretical) <= emp_ci, (
        f"Theoretical V*(s0)={v_theoretical:.4f} outside empirical 95% CI: "
        f"{emp_mean:.4f} +/- {emp_ci:.4f}"
    )


# ---------------------------------------------------------------- Strict Optimality vs Baselines
def test_exact_agent_strictly_outperforms_baselines_on_3x3():
    """ExactAgent must strictly outperform NoOp, Random, NearestFront, and Greedy in true Q-value."""
    # Specific setup where Greedy is suboptimal: mixed vegetation, ignition at center (4)
    cfg = make_config(3, 3, vegetation="mixed", wind=None, k=1, lam=0.2, ignition=(4,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)
    solver.solve(method="both")

    s0 = env.reset()
    opt_a = solver.get_action(s0)
    probs = env.ignition_probs(s0)
    tot = sum(probs.values())
    greedy_a = max(
        env.legal_actions(s0),
        key=lambda a: -(tot - sum(probs.get(c, 0.0) for c in a)) - 0.2 * len(a),
    )

    # In this configuration, Greedy picks (3,) while optimal action is (5,)
    assert opt_a != greedy_a
    q_opt = solver.q_value(s0, opt_a)
    q_greedy = solver.q_value(s0, greedy_a)
    assert q_opt > q_greedy, f"Q(opt)={q_opt} should be strictly greater than Q(greedy)={q_greedy}"

    # Verify simulation returns over 1500 episodes
    seeds = range(1500)
    r_exact = np.mean([run_episode(env, ExactAgent(solver), s)["return"] for s in seeds])
    r_greedy = np.mean([run_episode(env, GreedyAgent(), s)["return"] for s in seeds])
    r_nearest = np.mean([run_episode(env, NearestFrontAgent(), s)["return"] for s in seeds])
    r_random = np.mean([run_episode(env, RandomAgent(), s)["return"] for s in seeds])
    r_noop = np.mean([run_episode(env, NoOpAgent(), s)["return"] for s in seeds])

    assert r_exact > r_greedy
    assert r_greedy > r_nearest
    assert r_nearest > r_random
    assert r_random > r_noop


def test_exact_solver_on_3x4_grid():
    """Verify ExactSolver correctly solves 3x4 grid within safety limits."""
    cfg = make_config(3, 4, vegetation="mixed", wind="E", k=1, lam=0.2, ignition=(0,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)

    # Solve via Backward DP on 3x4
    res = solver.solve(method="backward_dp", max_states=100000, time_limit=30.0)
    assert res.reachable_count > 10000
    assert res.v_s0 < 0.0
    assert res.solve_time < 10.0

    # Ensure agent works on 3x4
    agent = ExactAgent(solver)
    out = run_episode(env, agent, seed=42)
    assert out["steps"] > 0
    assert out["saved"] > 0


# ---------------------------------------------------------------- Optimality Gap & Handover
def test_optimality_gap_calculation():
    """optimality_gap must be 0 for optimal actions and positive for suboptimal ones."""
    cfg = make_config(3, 3, vegetation="mixed", wind=None, k=1, lam=0.2, ignition=(4,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)
    solver.solve(method="backward_dp")

    s0 = env.reset()
    opt_a = solver.get_action(s0)
    assert solver.optimality_gap(s0, opt_a) == 0.0

    # Suboptimal action has positive gap
    gap_noop = solver.optimality_gap(s0, ())
    assert gap_noop > 0.0


# ---------------------------------------------------------------- Safety Guard
def test_safety_guard_triggers_on_exceeded_states():
    """StateSpaceTooLargeError should be raised when state limit is exceeded."""
    cfg = make_config(4, 4, vegetation="grass", ignition=(0,))
    env = WildfireEnv(cfg)
    solver = ExactSolver(env)

    with pytest.raises(StateSpaceTooLargeError):
        # Force tiny max_states to test safety guard
        solver.solve_backward_dp(max_states=50)
