# Project Specifications
## Terminology
1. The "perfect" player (PB&J) is Player 2 and plays a fixed strategy, controlled by a value `q`. `q` is both how often PB&J bluffs with J and how often PB&J calls a bet with Q. `q = 1/3` is the equilibrium strategy
2. The "training" player (Peter) is Player 1 and learns with Q-learning. He keeps a Q-table, which is one score per (state, action) pair, his current estimate of the chips he wins on average by taking that action in that state
3. A state is Peter's card plus where he is in the hand, so `(card, "start")` with actions `bet`/`check`, and `(Q, "facing_bet")` with actions `call`/`fold` (e.g Peter checked, then PB&J bet). That is 4 states x 2 actions = 8 Q-values
4. Exploration rate `eps` is how often Peter picks a random action instead of the one with the higher Q-value. Currently `0.1`
5. Step size `lr` is how far each Q-value moves toward its target. Currently `"visits"`, meaning `1/N` where `N` is how many times that Q-value has been updated

## Axioms
1. Peter is always the first player
2. Peter only sees his own card, so his Q-table depends only on that card and the betting so far
3. Peter never sees PB&J's strategy or `q`. He only learns from the chips he wins or loses
4. Facing a bet with J, Peter always folds. 
5. Facing a bet with K, Peter always calls.

## Process
1. Start every Q-value at `0`
2. Play a hand, Peter picks the action with the higher Q-value, or a random action with probability `eps`
3. Update each Q-value Peter used that hand: `Q = Q + lr * (target - Q)`
    * If the hand ended after that action, `target` = chips won or lost
    * If Peter checked and PB&J bet, `target` = the higher Q-value of `(Q, "facing_bet")`
4. Repeat 2,3 for `M` hands (`M=200,000` right now)
5. Peter's strategy is the action with the higher Q-value in each state, written as `bet_J`, `bet_Q`, `bet_K`, `call_Q` (each `0` or `1`)
6. Run a simulation of `M` hands with that strategy to get the simulated EV
7. Do 1-6 against three versions of PB&J: `q = 0.2`, `q = 1/3`, `q = 0.5`

## Operator Instructions
1. **ALWAYS PULL THE LATEST VERSION OF THE REPO BEFORE RUNNING**
2. Run `logic.py`. For each `q` it prints the learned Q-table, the greedy policy (`bet_J`, `bet_Q`, `bet_K`, `call_Q`), and the simulated EV `+/-` its confidence interval
3. Runs are seeded (`seed=0` in `train`), so the same settings always give the same output. Change `seed` to get a different run
4. Log the Q-table, policy, simulated EV, and the settings used (`seed`, `eps`, `lr`, number of hands) in `log.md`
5. **MAKE SURE TO PUSH YOUR `log.md` AND RESULTS ONTO THE REPO**

## TO DO
1. Evaluate the learned Q-values against the exact Q-values for each `q`
2. Compare different `eps`, `lr`, and number of hands to see how fast and how stably the Q-values settle
4. Note in the report that Q-learning finds a best response to a fixed PB&J, not the mixed equilibrium:
    * `bet_J = alpha`
    * `bet_K = 3*alpha`
    * `call_Q = alpha + 1/3`
    * `bet_Q = 0`

**99999**. 3 man Kuhn Poker
