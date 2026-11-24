# Learning to play Kuhn poker

The project studies two-player Kuhn poker. The code is split into the three implementations it went through, in order. Each folder has its own code, logs and tests. Run every command from the repository root with Python 3.10+ (standard library only, unless noted). The LaTeX report is on the `Writeup` branch.

```
common/          shared exact evaluator and log helpers (used to score all three)
hill_climbing/   1. hill climbing against a fixed opponent toward a known EV
q_learning/      2. partial Q-learning: 4 decision points, 8 Q-values (Brian)
self_play/       3. Q-learning over all 6 information sets, fixed bots and self-play (Kevin)
```

Run all tests:

```sh
python -m unittest discover -t .          # 26 tests; 4 skip unless open-spiel is installed
```

## 1. Hill climbing (`hill_climbing/`)

Peter (player one) has four probabilities, `bet_J`, `bet_Q`, `bet_K` and `call_Q`. PB&J (player two) plays a fixed strategy with parameter `q` (default 1/3, the equilibrium). Starting from random values, we nudge each probability by `lr = 0.005` and keep a change only if Peter's exact EV goes up. We then check the EV by simulating `M = 200,000` hands.

| File | What it is |
| --- | --- |
| `hill_climbing.py` | The hill climber. `python -m hill_climbing.hill_climbing` runs one random start. |
| `log_runs.py` | Runs 10,000 random starts and writes `logs/log.txt` (about 11 minutes). |
| `logs/log.txt` | The 10,000-run log: learned probabilities and simulated EV for each run. |
| `tests/test_evaluate.py` | Checks the hill climber's exact EV formula against the shared evaluator, and the evaluator against brute force. |

Result: every run reached the optimal value −1/18 and learned `bet_Q = 0`, but `bet_J`, `bet_K` and `call_Q` stayed wherever they started. The first logged policy `(0.406, 0, 0.740, 0.341)` earns −0.05556 against the fixed equilibrium bot but −0.18833 against its best counter. Matching the game value against one opponent is not proof of equilibrium. Score the whole log with `python -m common.evaluate log hill_climbing/logs/log.txt`.

## 2. Partial Q-learning (`q_learning/`)

Peter learns only from the chips he wins in simulated hands, without knowing PB&J's strategy. The states are his four real decisions (bet or check with each card, and call or fold with a Queen after checking into a bet). A Jack facing a bet always folds and a King always calls. Updates use step size 1/visits and ε-greedy exploration (ε = 0.1).

| File | What it is |
| --- | --- |
| `partial_q_learning.py` | Brian's learner. `python -m q_learning.partial_q_learning` trains against three values of `q`. |
| `logs/partial_q_learning.{json,txt}` | q ∈ {0, 0.2, 1/3, 0.5, 1}, seeds 0–9, policy frozen and scored exactly after 10³, 10⁴, 10⁵ and 10⁶ hands. The JSON keeps each policy, its EV, its worst-case EV and the full Q-table. |

Result: 40 of 50 runs were already a best response after 10³ hands, 49 of 50 after 10⁴, and all 50 after 10⁵. This folder has no unit tests of its own. Its policies are scored by `common/evaluate.py`, which is tested in `hill_climbing/tests` and `common/tests`.

## 3. Self-play (`self_play/`)

Both players learn from their own winnings. Each player has a Q-table over all six of its information sets, and nothing is hard-coded. There are four variants: Peter against four fixed bots, naive self-play, fictitious self-play with Q-learned replies, and fictitious self-play with exact replies. See [`self_play/EVALUATION.md`](self_play/EVALUATION.md) for the full method and results.

| File | What it is |
| --- | --- |
| `q_learning.py` | Six-information-set Q-learner against fixed bots. `python -m self_play.q_learning --opponent all --episodes 100000 --seeds 0 1` |
| `self_play.py` | Naive and fictitious self-play. |
| `fixed_curve.py`, `self_curve.py` | Kevin's learning curves (seeds 0–4) to `results/`. `python -m self_play.self_curve`. Matplotlib is needed only for the figures. |
| `three_player.py` | Extension: three-player Kuhn poker solved with CFR in OpenSpiel (see below). |
| `logs/fixed_opponents.{json,txt}` | Four bots × seeds 0–9, frozen after 10³–10⁶ hands. |
| `logs/selfplay.{json,txt}` | Exploitability of current and average strategies, seeds 0–9, up to 1,000 rounds (10⁶ hands). |
| `results/` | Kevin's original curves and figures, plus the three-player run. |
| `tests/` | Q-learner, self-play, learning-curve CLI, OpenSpiel cross-check and three-player tests. |

Result: against fixed bots, every seed found an exact best response by 10⁴ hands. In self-play, naive and learned-reply fictitious play cut exploitability from 0.458 to about 0.04 chips per hand within 10⁵ hands and then plateau. Fictitious play with exact replies keeps falling, to 0.006 after 1,000 rounds.

### Three-player extension

`self_play/three_player.py` runs OpenSpiel's four-card, three-seat Kuhn game: each player antes one chip and gets one private card, one card stays out, and there is one betting round with no raises. All three seats learn by counterfactual regret minimization (CFR). This uses OpenSpiel's game and solver, not the two-player Q-learner.

```sh
python -m pip install open-spiel==2.0.2
python -m self_play.three_player > three_player_run.json
python -m unittest discover -t . -s self_play -p test_three_player.py -v
```

| Full-tree CFR iterations | NashConv (chips/hand) |
| ---: | ---: |
| 0 (uniform actions) | 2.062500 |
| 10 | 0.312481 |
| 100 | 0.037016 |
| 1,000 | 0.003922 |

At each checkpoint we freeze the average strategies, let each seat switch to its best response while the other two stay fixed, and sum the three possible gains (NashConv). At 1,000 iterations the largest single-seat gain was about 0.00152 chips/hand. This measures how close this profile came to equilibrium; it does not establish an exact equilibrium. One iteration is a full game-tree update, not one sampled hand. [Saved checkpoints and final strategy](self_play/results/three_player.json).

## Shared code (`common/`)

| File | What it is |
| --- | --- |
| `evaluate.py` | Exact evaluator: EV, best responses, worst case and exploitability for any pair of policies. `python -m common.evaluate policy --p1 .406 0 .74 .341` |
| `curve_io.py` | Writes the JSON logs. |
| `run_experiments.py` | Regenerates the logs for implementations 2 and 3: `python -m common.run_experiments all` (about 2 minutes; `--quick` writes a small smoke test to `logs/quick/`). |
| `logs/run_console.log` | Console output of the full run. |
| `tests/test_cli.py` | Tests the evaluator's command line. |
