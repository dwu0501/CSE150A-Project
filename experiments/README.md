# Experiments for the final report

These scripts produce every stage-2 and stage-3 number in `writeup/writeup_v1.2.tex`.
They run the team's existing learners and add no new algorithms:

| Stage | Learner | Code |
|---|---|---|
| 2 | Partial Q-learning: 4 decision points, 8 Q-values (Brian) | `partial_q_learning.py` |
| 3 | Q-learning over all 6 information sets vs. fixed bots (Kevin) | `q_learning.py` |
| 3 | Naive self-play and fictitious self-play, with learned or exact replies (Kevin) | `self_play.py` |

Every frozen policy is scored exactly with `evaluate.py` (no sampling). Stage 1 (hill climbing, `logic.py`) was not re-run; its numbers come from the earlier 10k-restart log on the `run_log` branch.

## Reproduce

From the repo root, with Python 3.10+ and only the standard library:

```sh
python experiments/run_experiments.py all      # ~2 minutes; writes experiments/logs/
python experiments/make_tables.py > experiments/logs/latex_snippets.txt
python experiments/run_experiments.py all --quick   # smoke test, writes experiments/logs/quick/
```

## Budget

| Run | Settings | Seeds | Training length | Time |
|---|---|---|---|---|
| Stage 2 | P2 q ∈ {0, 0.2, 1/3, 0.5, 1}, ε = 0.1, step 1/N | 0–9 | checkpoints at 10³, 10⁴, 10⁵, 10⁶ hands | 52 s |
| Stage 3, fixed bots | always-pass, always-bet, random, equilibrium; ε = 0.2 | 0–9 | checkpoints at 10³, 10⁴, 10⁵, 10⁶ hands | 21 s |
| Stage 3, self-play | naive and learned (ε = 0.2, 1,000 hands per round); exact | 0–9 (exact: one deterministic run) | 1,000 rounds (10⁶ hands), checkpoints at rounds 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 | 36 s |

## Logs (`experiments/logs/`)

- `stage2_partial_q.json`: one row per (q, seed, checkpoint). Each row has the greedy policy `[bet_J, bet_Q, bet_K, call_Q]`, the exact EV, the best possible EV, whether the policy is a best response, its worst-case EV, and the full Q-table with visit counts.
- `stage3_fixed_opponents.json`: one row per (opponent, seed, checkpoint), with the 6-number policy, the EV, the worst-case EV, the security gap, the best possible EV and whether the policy is a best response.
- `stage3_selfplay.json`: one row per (method, seed, round), with the current and average policy pairs and their exploitability.
- `*.txt`: human-readable summary tables from the same runs.
- `run_console.log`: console output of the full run.
- `latex_snippets.txt`: the table rows and pgfplots coordinates pasted into the report.

Each JSON file records the git commit, the start time, the settings and the runtime.

## Check

For seeds 0–4 at 100 rounds, the self-play results reproduce Kevin's `results/selfplay.json` exactly: median average exploitability 0.04526 (naive) and 0.04043 (learned replies).
