# Wildfire Containment: MDP-Based Resource Allocation (CS272)

Stage 1 (Zainab): environment, simulator, baselines B0-B2, console visualisation, unit tests.

## Setup
```bash
pip install -r requirements.txt
python -m pytest -q                       # 31 tests
python demo.py --rows 5 --cols 5 --wind E --episodes 500
python demo.py --show greedy --seed 3 --rows 4 --cols 4 --wind E   # watch one episode
```
Run everything from the project root so `import wildfire` works.

## Project layout
```
wildfire/
  config.py     Config dataclass, status/vegetation codes, make_config()
  env.py        WildfireEnv: reset, legal_actions, step, transition_probs
  agents.py     NoOp, B0 Random, B1 NearestFront, B2 Greedy
  simulate.py   run_episode, evaluate, mean_ci (common random numbers)
  visualize.py  console rendering
demo.py         quick baseline comparison
tests/          pytest unit tests
```
Dependencies point one way only: `config <- env <- agents/solver/mcts <- experiments`.
The environment knows nothing about the agents (no circular imports).

## Interface contract (Tayyba and Basma code against this)
| Item | Definition |
|---|---|
| State | flat tuple of length R*C, entries `U=0, B=1, X=2, F=3` (hashable) |
| Action | sorted tuple of flat cell indices; `()` = no-op; otherwise exactly `min(k, n_unburnt)` distinct unburnt cells |
| Flat index | `i = r * C + c` (`env.rc(i)`, `env.idx(r, c)`) |
| `env.reset(seed=None, ignition=None)` | initial state (ignition: argument > config > random cell from seed) |
| `env.legal_actions(state)` | list of actions, size C(n_U, k) + 1 |
| `env.step(state, action, rng)` | `(next_state, reward, done)`, samples one turn |
| `env.transition_probs(state, action)` | exact list of `(prob, next_state, reward)` for value iteration |
| `env.is_terminal(state)` | true when no cell is burning |
| `env.ignition_probs(state, action=())` | dict `{cell: P(ignite)}` after applying the action |
| `agent.select_action(state, env, rng)` | returns a legal action; every agent uses this signature |

`transition_probs` and `step` share the same internal probability code, so the exact
model and the simulator cannot drift apart (a test also checks them against each other).

## Modelling decisions (please confirm as a group)
1. **Reward counts cells that CATCH FIRE this turn**, not cells that turn from burning to
   burnt. Reason: burning cells always become burnt, so that definition would not depend on
   the action and one-step greedy would never build a firebreak. The initial ignition cell(s)
   are not counted (constant offset). `cells_saved` / `burnt` in results still include them.
2. **Wind blows TOWARDS the given direction** (wind `E` pushes fire east). Downwind x1.5,
   crosswind x1.0, upwind x0.5. No wind = x1.0 everywhere.
3. **The 0.95 cap is applied per burning-neighbour edge**, then edges are combined with
   `P = 1 - prod(1 - q_j)`.
4. **No total firebreak budget** (k per turn only), so the state is just the grid.
   If we add a total budget later, the budget must be added to the state.
5. If fewer than k unburnt cells remain, the agent places min(k, n_unburnt) firebreaks.
6. B1 and B0 always place firebreaks (never no-op); B2 can choose the no-op when no
   firebreak is worth lambda.

## Baselines
- **NoOp**: never acts (sanity lower bound).
- **B0 Random**: k random unburnt cells.
- **B1 NearestFront**: closest unburnt cells to the fire (Manhattan), ties by flammability then index.
- **B2 Greedy**: exact expected immediate reward of every legal action; picks the best.

## Reproducibility
`run_episode(env, agent, seed)` splits the seed into two independent streams: one for the
environment, one for the agent. So the same seed gives the same ignition cell and the same
environment stream for every agent (common random numbers). Log: Python version, NumPy
version, seeds, grid/vegetation/wind config, lambda, k.

## Handover checklist (Stage 1 -> Stage 2)
- [ ] All tests pass (`python -m pytest -q`)
- [ ] Tayyba has run `demo.py` and read this contract
- [ ] Walkthrough done: she can explain `step`, `ignition_probs` and `transition_probs`
- [ ] Modelling decisions above confirmed by the group
- [ ] Tag the version: `git tag stage1-done`

## AI assistance declaration
Stage 1 code was drafted with generative-AI assistance (Claude). It has been reviewed,
tested and must be understood by all members. Declare this in the final report.
