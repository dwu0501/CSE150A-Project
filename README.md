# Learning to play Kuhn poker

The class project studies two-player Kuhn poker. The original checklist is complete; the three-player experiment below is a separate extension. Run the commands from the repository root on `main`. The team's LaTeX draft is on the `Writeup` branch.

## Two-player results

| Original task | Result |
| --- | --- |
| Choose the simulation size and hill-climbing step | `M = 200,000` simulated hands per final policy; `lr = 0.005` per hill-climbing move. This is not the Q-learning step size. |
| Run from multiple random starts | The [run log](https://github.com/dwu0501/CSE150A-Project/blob/run_log/log.txt) records 10,000 final policies. In the rounded log `bet_Q` was always zero; `bet_J`, `bet_K`, and `call_Q` each ranged from 0 to 1. |
| Compare the policies with equilibrium | The equilibrium family requires `bet_J = α`, `bet_K = 3α`, `call_Q = α + 1/3`, `bet_Q = 0` for `0 ≤ α ≤ 1/3`. The first logged policy `(0.406, 0, 0.740, 0.341)` does not satisfy those relations. As Player 1 it earns −0.05556 chips/hand against the fixed equilibrium bot, but −0.18833 against its best counter. Matching the game value against one opponent is not proof of equilibrium. |

To reproduce the original hill climber, run `python beta_logic.py`. For Brian's fixed-opponent Q-learning experiment, run `python logic.py`. The [evaluation guide](https://github.com/dwu0501/CSE150A-Project/blob/main/EVALUATION.md) covers exact best responses, the comparison learning curves, and two-player self-play. No manual log uploads are needed.

## Three-player extension

[`three_player.py`](https://github.com/dwu0501/CSE150A-Project/blob/main/three_player.py) runs OpenSpiel's four-card, three-seat Kuhn game: each player antes one chip and gets one private card, one card stays out, and there is one betting round with no raises. All three seats learn by counterfactual regret minimization (CFR). This uses OpenSpiel's game and solver, not the two-player Q-learner.

```sh
python -m pip install open-spiel==2.0.2
python three_player.py > three_player_run.json
python -m unittest discover -s tests -p test_three_player.py -v
```

| Full-tree CFR iterations | NashConv (chips/hand) |
| ---: | ---: |
| 0 (uniform actions) | 2.062500 |
| 10 | 0.312481 |
| 100 | 0.037016 |
| 1,000 | 0.003922 |

At each checkpoint we freeze the average strategies, let each seat switch to its best response while the other two stay fixed, and sum the three possible gains (NashConv). At 1,000 iterations the largest single-seat gain was about 0.00152 chips/hand. This measures how close this profile came to equilibrium; it does not establish an exact equilibrium. One iteration is a full game-tree update, not one sampled hand.

[Saved checkpoints and final strategy](https://github.com/dwu0501/CSE150A-Project/blob/main/results/three_player.json) include the three expected payoffs, each seat's deviation gain, and action probabilities at all 48 decision information sets of the final average profile. To change the budget, e.g. run `python three_player.py --checkpoints 0 50 500`.
