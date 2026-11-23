import pytest

from wildfire import (make_config, WildfireEnv, Agent, NoOpAgent, RandomAgent, NearestFrontAgent,
                      GreedyAgent, MCTSAgent, ExactAgent, run_episode)
from wildfire.robustness import (check_compatible, run_mismatch_episode, run_mismatch_scenario,
                                 paired_delta)


def env(rows=3, cols=3, wind="E", veg="mixed", k=1, p_scale=1.0):
    p = tuple(min(0.95, v * p_scale) for v in (0.6, 0.45, 0.3))
    return WildfireEnv(make_config(rows, cols, veg, wind=wind, p_veg=p, k=k))


class Spy(Agent):
    """Never builds a firebreak; records which environment it was asked to plan with."""
    name = "Spy"

    def __init__(self):
        self.seen = []

    def select_action(self, state, env, rng):
        self.seen.append(env)
        return ()


def test_correct_belief_matches_run_episode():
    e = env(5, 5)
    for seed in range(10):
        ref = run_episode(e, GreedyAgent(), seed)
        got = run_mismatch_episode(e, e, GreedyAgent(), seed)
        assert (got.return_, got.burnt, got.steps, got.firebreaks) == (
            ref["return"], ref["burnt"], ref["steps"], ref["firebreaks"])


def test_same_seed_same_result():
    t, b = env(wind="E"), env(wind="N")
    a = run_mismatch_episode(t, b, GreedyAgent(), 4)
    c = run_mismatch_episode(t, b, GreedyAgent(), 4)
    assert (a.return_, a.burnt, a.steps) == (c.return_, c.burnt, c.steps)


@pytest.mark.parametrize("agent_cls", [NoOpAgent, RandomAgent, NearestFrontAgent])
def test_model_free_agents_ignore_the_belief(agent_cls):
    t = env(5, 5, wind="E")
    wrong = [env(5, 5, wind="N"), env(5, 5, wind="W"), env(5, 5, wind=None), env(5, 5, p_scale=0.7)]
    for seed in range(15):
        ref = run_mismatch_episode(t, t, agent_cls(), seed).return_
        for b in wrong:
            assert run_mismatch_episode(t, b, agent_cls(), seed).return_ == ref


def test_greedy_is_affected_by_the_belief():
    t, wrong = env(5, 5, wind="E"), env(5, 5, wind="W")
    diffs = [run_mismatch_episode(t, wrong, GreedyAgent(), s).return_ !=
             run_mismatch_episode(t, t, GreedyAgent(), s).return_ for s in range(40)]
    assert any(diffs)


def test_shift_at_turn_zero_equals_running_in_the_shifted_world():
    e_world, n_world = env(5, 5, wind="E"), env(5, 5, wind="N")
    for seed in range(10):
        a = run_mismatch_episode(e_world, e_world, GreedyAgent(), seed, shifted_env=n_world, shift_turn=0)
        b = run_mismatch_episode(n_world, e_world, GreedyAgent(), seed)
        assert (a.return_, a.burnt, a.steps) == (b.return_, b.burnt, b.steps)


def test_late_shift_equals_no_shift():
    e_world, n_world = env(5, 5, wind="E"), env(5, 5, wind="N")
    for seed in range(5):
        a = run_mismatch_episode(e_world, e_world, GreedyAgent(), seed, shifted_env=n_world, shift_turn=999)
        b = run_mismatch_episode(e_world, e_world, GreedyAgent(), seed)
        assert a.return_ == b.return_


def test_planner_environment_before_and_after_a_shift():
    e_world, n_world = env(5, 5, wind="E"), env(5, 5, wind="N")
    unaware, aware_before, aware_after = Spy(), Spy(), Spy()
    run_mismatch_episode(e_world, e_world, unaware, 1, shifted_env=n_world, shift_turn=1)
    assert all(x is e_world for x in unaware.seen)
    run_mismatch_episode(e_world, e_world, aware_before, 1, shifted_env=n_world, shift_turn=1,
                         agent_after=aware_after)
    assert aware_before.seen == [e_world]                     # only turn 0
    assert aware_after.seen and all(x is n_world for x in aware_after.seen)


def test_incompatible_environments_are_rejected():
    with pytest.raises(ValueError):
        check_compatible(env(3, 3), env(4, 4))
    with pytest.raises(ValueError):
        check_compatible(env(3, 3, k=1), env(3, 3, k=2))
    with pytest.raises(ValueError):
        run_mismatch_episode(env(3, 3), env(3, 3), NoOpAgent(), 0, shifted_env=env(3, 3))   # no shift_turn


def test_scenario_summaries_and_paired_delta():
    t, wrong = env(5, 5, wind="E"), env(5, 5, wind="W")
    facs = {"NoOp": lambda e: NoOpAgent(), "Greedy": lambda e: GreedyAgent(),
            "MCTS": lambda e: MCTSAgent(n_rollouts=10)}
    t1, s1 = run_mismatch_scenario("g__correct", t, t, facs, range(6))
    t2, s2 = run_mismatch_scenario("g__wrong", t, wrong, facs, range(6))
    assert [s.n_episodes for s in s1] == [6, 6, 6] and s1[0].scenario_id == "g__correct"
    assert s1[0].tree_nodes_mean is None and s1[2].tree_nodes_mean is not None
    rows = {r["agent"]: r for r in paired_delta(t1 + t2, "g__wrong", "g__correct")}
    assert rows["NoOp"]["delta_mean"] == 0.0 and rows["NoOp"]["n_paired"] == 6      # model-free: unchanged
    assert set(rows) == {s.agent for s in s1}


def test_exact_agent_runs_with_a_wrong_belief():
    t, wrong = env(3, 3, wind="E"), env(3, 3, wind="N")
    res = run_mismatch_episode(t, wrong, ExactAgent(), 1)
    assert res.steps >= 1 and res.return_ <= 0
    # the same seed under the right belief runs the same fire (same ignition cell)
    right = run_mismatch_episode(t, t, ExactAgent(), 1)
    assert right.steps >= 1 and right.seed == res.seed == 1
