# Project Specifications
## Terminology
1. The "perfect" player (PB&J) plays Player 2's fixed equilibrium strategy, controlled by a single value `q` (default 1/3)
2. The "training" player (Peter) is Player 1 and is given four probabilities to converge on: `bet_J`, `bet_Q`, `bet_K` (chance of betting with each card) and `call_Q` (chance of calling a bet after checking with Q)
3. The check probabilities are just the complements, e.g. check with `J = 1 - bet_J`. The `1-alpha` and `alpha` branches in the game tree are probabilities we converge on
4. Learning rate `lr` is the step size of each hill climbing move. Currently `0.005`

## Axioms
1. Peter is always the first player
2. Peter only sees his own card, so his strategy depends only on that card
3. Alpha is a range of optimal values

## Process
1. Start Peter's four probabilities at random values in `[0,1]`
2. Calculate Peter's exact EV per hand in place of an MSE
3. Shift each probability by `+lr` or `-lr` and keep a change only if it strictly improves EV
4. Keep sweeping 2,3 until no change improves EV OR the max number of sweeps is hit
5. Run a simulation of `M` hands with the converged probabilities as a sanity check on the exact EV
6. Compare exact EV and simulated EV to the equilibrium value of `-1/18`

## Evaluating a policy

`evaluate.py` scores a pair of policies in chips per hand without sampling or training. It averages over all six deals and returns a best reply for each seat. Each reply must choose the same action for the same private card and public history: it cannot see the other player's card.

```sh
# First policy saved in run_log/log.txt, against the default equilibrium P2
python3 evaluate.py policy --p1 .406 0 .740 .341
# Same P1 against a P2 that chooses each legal action with equal probability
python3 evaluate.py policy --p1 .406 0 .740 .341 --p2 .5 .5 .5 .5 .5 .5
# Rescore the logged policies without switching branches
# (run git fetch origin first if origin/run_log isn't present)
git show origin/run_log:log.txt | python3 evaluate.py log -
```

For a learner, call `from evaluate import evaluate` and pass `evaluate(p1, p2)` its **action probabilities**, not raw Q-values. Each seat takes six numbers: three bet probabilities (J, Q, K), then three call probabilities (J, Q, K). P1 bets at the start and calls after check-bet; P2 bets after a check and calls after an opening bet. P1 can also use the original four `logic.py` parameters. Omitting P2 selects the equilibrium opponent, not a best reply.

| Output | Meaning (all EVs use P1's sign) |
| --- | --- |
| `p1_ev` | Expected payoff if both supplied policies stay fixed. |
| `p1_worst_case_ev` | P1's payoff if P2 switches to its best reply. |
| `p1_security_gap` | How much worse that guarantee is than the game value, −1/18. Zero means P1 guarantees the game value. |
| `p1_best_response_ev` | P1's highest payoff if it switches to its best reply against the supplied P2. |
| `nash_conv`, `profile_exploitability` | The gap between best-response and worst-case EV, and half that gap, respectively. These measure the **pair** of policies. |

For the first logged policy, P1 scores **−0.05556** against the fixed opponent but **−0.18833** against its best reply. Across the 10,000 rounded policies in log commit `c21c708`, the median worst-case EV is **−0.23733**. These are exact expectations for the saved probabilities, not the logged simulation estimates.

Run `python3 -m unittest discover -s tests -v` from the repo root. The tests compare the best replies to exhaustive search over each seat's 64 deterministic strategies. For the optional independent check, install `open-spiel==2.0.2` in a virtual environment and run that same command there. The OpenSpiel test checks expected payoff, both best-response values and profile exploitability for equilibrium, deterministic and mixed policies. The evaluator itself needs only Python's standard library.

## Operator Instructions
1. **ALWAYS PULL THE LATEST VERSION OF THE REPO BEFORE RUNNING**
2. Run `logic.py`. It prints `bet_J`, `bet_Q`, `bet_K`, `call_Q`, the exact EV, and the simulated EV
3. If exact EV is `~-1/18`, log the probabilities and EV/(-1/18) in `log.md`, along with the starting values it began from
4. If exact EV is not `~-1/18`, log the probabilities and EV in `log.md`, then rerun from a new random start
5. **MAKE SURE TO PUSH YOUR `log.md` AND PROBABILITIES ONTO THE REPO**

## TO DO
1. Determine `M` and the final learning rate `lr`
2. Run multiple random starts to see which probabilities we converge to
3. Compare the converged probabilities to the known equilibrium:
    * `bet_J = alpha`
    * `bet_K = 3*alpha`
    * `call_Q = alpha + 1/3`
    * `bet_Q = 0`

**99999**. 3 man Kuhn Poker
