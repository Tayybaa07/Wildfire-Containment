from math import comb

import numpy as np
import pytest

from wildfire import Config, make_config, WildfireEnv, U, B, X, F, GRASS


def grass_row(n, **kw):
    """1 x n all-grass strip."""
    return Config(rows=1, cols=n, vegetation=((GRASS,) * n,), **kw)


# ---------------------------------------------------------------- reset / state
def test_reset_one_burning_cell_and_deterministic():
    env = WildfireEnv(make_config(4, 4))
    s1, s2 = env.reset(seed=5), env.reset(seed=5)
    assert s1 == s2
    assert s1.count(B) == 1 and s1.count(U) == 15
    assert isinstance(s1, tuple)


def test_reset_fixed_ignition():
    env = WildfireEnv(make_config(3, 3, ignition=(4,)))
    assert env.reset()[4] == B


# ---------------------------------------------------------------- actions
@pytest.mark.parametrize("k", [1, 2])
def test_legal_action_count(k):
    env = WildfireEnv(make_config(3, 3, ignition=(0,), k=k))
    s = env.reset()
    assert len(env.legal_actions(s)) == comb(8, k) + 1       # C(n_U,k) + no-op


def test_illegal_actions_raise():
    env = WildfireEnv(make_config(3, 3, ignition=(0,)))
    s = env.reset()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        env.step(s, (0,), rng)            # burning cell
    with pytest.raises(ValueError):
        env.step(s, (1, 2), rng)          # wrong number of firebreaks (k=1)
    with pytest.raises(ValueError):
        env.step(s, (99,), rng)           # out of range


def test_terminal_state_only_noop():
    env = WildfireEnv(make_config(3, 3, ignition=(0,)))
    s = tuple([X] + [U] * 8)
    assert env.is_terminal(s)
    assert env.legal_actions(s) == [()]


# ---------------------------------------------------------------- dynamics
def test_burning_becomes_burnt_and_firebreak_applied():
    env = WildfireEnv(grass_row(5, ignition=(0,)))
    s = env.reset()
    nxt, _, _ = env.step(s, (4,), np.random.default_rng(0))
    assert nxt[0] == X and nxt[4] == F


def test_firebreak_never_ignites():
    env = WildfireEnv(grass_row(3, ignition=(0,), wind="E"))
    s = env.reset()
    for seed in range(300):
        nxt, _, _ = env.step(s, (1,), np.random.default_rng(seed))
        assert nxt[1] == F and nxt[2] == U


def test_status_only_moves_forward_and_episode_terminates():
    env = WildfireEnv(make_config(5, 5, "mixed", wind="E", k=1))
    rng = np.random.default_rng(1)
    for seed in range(50):
        s = env.reset(seed=seed)
        for _ in range(env.horizon + 1):
            if env.is_terminal(s):
                break
            a = env.legal_actions(s)[rng.integers(len(env.legal_actions(s)))]
            nxt, _, _ = env.step(s, a, rng)
            for old, new in zip(s, nxt):
                assert (old, new) in {(U, U), (U, B), (U, F), (B, X), (X, X), (F, F)}
            s = nxt
        else:
            pytest.fail("episode did not terminate")


def test_step_is_reproducible():
    env = WildfireEnv(make_config(4, 4, ignition=(5,)))
    s = env.reset()
    a = env.step(s, (), np.random.default_rng(7))
    b = env.step(s, (), np.random.default_rng(7))
    assert a == b


# ---------------------------------------------------------------- probabilities
def test_ignition_probability_two_burning_neighbours():
    env = WildfireEnv(grass_row(3, ignition=(0, 2)))         # no wind
    p = env.ignition_probs(env.reset())
    assert p[1] == pytest.approx(1 - 0.4 * 0.4)              # 0.84


def test_wind_multipliers():
    env = WildfireEnv(grass_row(3, ignition=(1,), wind="E"))
    p = env.ignition_probs(env.reset())
    assert p[2] == pytest.approx(0.9)       # downwind 0.6 * 1.5
    assert p[0] == pytest.approx(0.3)       # upwind   0.6 * 0.5


def test_crosswind_unchanged():
    env = WildfireEnv(make_config(3, 3, "grass", wind="E", ignition=(4,)))
    p = env.ignition_probs(env.reset())
    assert p[1] == pytest.approx(0.6) and p[7] == pytest.approx(0.6)   # N / S of centre


def test_probability_cap():
    env = WildfireEnv(grass_row(3, ignition=(1,), wind="E", p_veg=(0.8, 0.45, 0.3)))
    assert env.ignition_probs(env.reset())[2] == pytest.approx(0.95)   # 0.8*1.5 capped


# ---------------------------------------------------------------- reward
def test_reward_includes_firebreak_cost():
    env = WildfireEnv(grass_row(1 + 3, ignition=(0,), lam=0.2))
    s = env.reset()
    # firebreak on the only at-risk cell: nothing can ignite, cost = lambda
    _, r, done = env.step(s, (1,), np.random.default_rng(0))
    assert r == pytest.approx(-0.2) and done


# ---------------------------------------------------------------- exact transitions
@pytest.mark.parametrize("wind", [None, "E"])
def test_transition_probs_sum_to_one_and_match_sampling(wind):
    env = WildfireEnv(make_config(3, 3, "mixed", wind=wind, ignition=(4,), lam=0.2))
    s = env.reset()
    action = (0,)
    outs = env.transition_probs(s, action)
    assert sum(p for p, _, _ in outs) == pytest.approx(1.0)
    exact_r = sum(p * r for p, _, r in outs)

    rng = np.random.default_rng(0)
    samples = [env.step(s, action, rng)[1] for _ in range(20000)]
    assert np.mean(samples) == pytest.approx(exact_r, abs=0.03)

    # sampled next states must be among the enumerated ones
    states = {ns for _, ns, _ in outs}
    for _ in range(200):
        assert env.step(s, action, rng)[0] in states


def test_transition_probs_terminal():
    env = WildfireEnv(make_config(3, 3, ignition=(0,)))
    t = tuple([X] * 9)
    assert env.transition_probs(t, ()) == [(1.0, t, 0.0)]
