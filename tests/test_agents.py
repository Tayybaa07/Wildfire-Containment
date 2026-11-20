import numpy as np
import pytest

from wildfire import (make_config, Config, WildfireEnv, AGENTS, RandomAgent,
                      NearestFrontAgent, GreedyAgent, run_episode, GRASS, FOREST, B)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name", list(AGENTS))
def test_agents_return_legal_actions(name, k):
    env = WildfireEnv(make_config(4, 4, "mixed", wind="E", k=k))
    agent = AGENTS[name]()
    rng = np.random.default_rng(0)
    for seed in range(30):
        s = env.reset(seed=seed)
        for _ in range(env.horizon):
            if env.is_terminal(s):
                break
            a = agent.select_action(s, env, rng)
            assert env.is_legal(s, a), (name, s, a)
            s, _, _ = env.step(s, a, rng)


def test_nearest_front_picks_cell_next_to_fire_and_prefers_flammable():
    # 1 x 3 row; fire at 0 -> cell 1 is the nearest unburnt cell
    cfg = Config(rows=1, cols=3, vegetation=((GRASS, GRASS, GRASS),), ignition=(0,))
    env = WildfireEnv(cfg)
    assert NearestFrontAgent().select_action(env.reset(), env, None) == (1,)

    # tie on distance: fire in the middle of a column, grass above, forest below
    cfg = Config(rows=3, cols=1, vegetation=((GRASS,), (GRASS,), (FOREST,)),
                 ignition=(1,))
    env = WildfireEnv(cfg)
    assert NearestFrontAgent().select_action(env.reset(), env, None) == (0,)  # grass is more flammable


def test_greedy_blocks_highest_probability_cell():
    # wind E: cell to the east of the fire (downwind) is the most threatened
    env = WildfireEnv(make_config(3, 3, "grass", wind="E", ignition=(3,)))
    s = env.reset()
    a = GreedyAgent().select_action(s, env, np.random.default_rng(0))
    assert a == (4,)


def test_greedy_noop_when_firebreaks_too_expensive():
    env = WildfireEnv(make_config(3, 3, "grass", ignition=(4,), lam=5.0))
    s = env.reset()
    assert GreedyAgent().select_action(s, env, np.random.default_rng(0)) == ()


def test_run_episode_reproducible_and_common_ignition():
    env = WildfireEnv(make_config(5, 5, "mixed", wind="E"))
    r1 = run_episode(env, RandomAgent(), seed=11)
    r2 = run_episode(env, RandomAgent(), seed=11)
    assert r1 == r2
    # same seed -> same ignition cell regardless of the agent
    assert env.reset(seed=11) == env.reset(seed=11)


def test_greedy_beats_random_on_average():
    env = WildfireEnv(make_config(5, 5, "mixed", wind="E"))
    seeds = range(300)
    mean = lambda ag: np.mean([run_episode(env, ag, s)["return"] for s in seeds])
    assert mean(GreedyAgent()) > mean(RandomAgent())
