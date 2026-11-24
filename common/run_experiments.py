"""Run the stage-2 and stage-3 learning experiments and save their logs.

Stage 2 : Brian's partial Q-learner (partial_q_learning.py), 4 decision points,
          against a fixed P2 with bluff/call rate q.
Stage 3 : Kevin's learners -- Q-learning against four fixed bots (q_learning.py)
          and two-seat self-play / fictitious play (self_play.py).

Nothing here is a new algorithm: every learner is the team's existing code.
This file only loops over seeds/settings, freezes policies at checkpoints,
scores them exactly with evaluate.py, and writes JSON + text logs.

Logs go to q_learning/logs/ (stage 2) and self_play/logs/ (stage 3).

Usage (from the repo root):
    python -m common.run_experiments stage2
    python -m common.run_experiments fixed
    python -m common.run_experiments selfplay
    python -m common.run_experiments all
Add --quick for a small smoke-test budget.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q_learning import partial_q_learning as pq                # noqa: E402  (Brian)
from common.evaluate import evaluate                           # noqa: E402  (Kevin)
from self_play.q_learning import OPPONENTS, checkpoint_run     # noqa: E402  (Kevin)
from self_play.self_play import run as selfplay_run            # noqa: E402  (Kevin)
from common.curve_io import save_curve                         # noqa: E402  (Kevin)

# Each implementation keeps its own logs.
LOGS = {'stage2': ROOT / 'q_learning' / 'logs', 'fixed': ROOT / 'self_play' / 'logs',
        'selfplay': ROOT / 'self_play' / 'logs'}
TOL = 1e-9

# ---------------------------------------------------------------- budgets
FULL = dict(
    seeds=list(range(10)),
    hand_checkpoints=(1_000, 10_000, 100_000, 1_000_000),
    selfplay_rounds=(0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    batch=1000,
)
QUICK = dict(
    seeds=[0, 1],
    hand_checkpoints=(1_000, 10_000),
    selfplay_rounds=(0, 1, 5, 10),
    batch=200,
)
STAGE2_QS = {'0': 0.0, '0.2': 0.2, '1/3': 1 / 3, '0.5': 0.5, '1': 1.0}
STAGE2_EPS = 0.1      # Brian's default
STAGE3_EPS = 0.2      # Kevin's default


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def p2_from_q(q):
    """Brian's P2 in Kevin's 6-number format: bets after check (J,Q,K), calls vs bet (J,Q,K)."""
    return (q, 0.0, 1.0, 0.0, q, 1.0)


def header(name, settings):
    return dict(experiment=name, commit=git_commit(),
                started=time.strftime('%Y-%m-%d %H:%M:%S'), **settings)


# ---------------------------------------------------------------- stage 2
def stage2_run(q, seed, checkpoints):
    """One uninterrupted run of Brian's learner, frozen and scored at each checkpoint."""
    rng = random.Random(seed)
    Qt = pq.new_table()
    N = {key: 0 for key in Qt}
    wanted = set(checkpoints)
    p2 = p2_from_q(q)
    rows = []
    for hand in range(1, checkpoints[-1] + 1):
        trajectory, reward = pq.play_hand(Qt, q, rng, STAGE2_EPS)
        pq.q_update(Qt, N, trajectory, reward, 'visits')
        if hand in wanted:
            params = pq.to_params(Qt)
            s = evaluate(params, p2)
            rows.append(dict(
                hands=hand, seed=seed, policy=params,
                p1_ev=s['p1_ev'], best_possible_ev=s['p1_best_response_ev'],
                at_best_response=s['p1_ev'] >= s['p1_best_response_ev'] - TOL,
                worst_case_ev=s['p1_worst_case_ev'],
                q_values={f'{pq.CARD_NAMES[c]},{node},{a}': Qt[((c, node), a)]
                          for (c, node), a in Qt},
                visits={f'{pq.CARD_NAMES[c]},{node},{a}': N[((c, node), a)]
                        for (c, node), a in N}))
    return rows


def stage2(budget):
    t0 = time.time()
    data = header('stage2_partial_q_learning', dict(
        epsilon=STAGE2_EPS, step_size='1/N(s,a)', seeds=budget['seeds'],
        checkpoints=list(budget['hand_checkpoints']), opponent_q=list(STAGE2_QS)))
    rows = []
    for label, q in STAGE2_QS.items():
        for seed in budget['seeds']:
            for row in stage2_run(q, seed, budget['hand_checkpoints']):
                rows.append(dict(q=label, **row))
            print(f'  stage2 q={label} seed={seed} done ({time.time() - t0:.0f}s)', flush=True)
    data['seconds'] = round(time.time() - t0, 1)
    data['rows'] = rows
    save_curve(data, LOGS['stage2'] / 'partial_q_learning.json')

    lines = [f"Stage 2: partial Q-learning, eps={STAGE2_EPS}, {len(budget['seeds'])} seeds, "
             f"{data['seconds']} s", '',
             f"{'q':>5} {'hands':>9} {'at BR':>6} {'median EV':>10} {'best EV':>9} {'median worst':>13}"]
    for label in STAGE2_QS:
        for n in budget['hand_checkpoints']:
            rs = [r for r in rows if r['q'] == label and r['hands'] == n]
            lines.append(f"{label:>5} {n:>9,} {sum(r['at_best_response'] for r in rs):>3}/{len(rs):<2} "
                         f"{median(r['p1_ev'] for r in rs):>10.5f} {rs[0]['best_possible_ev']:>9.5f} "
                         f"{median(r['worst_case_ev'] for r in rs):>13.5f}")
    write_summary(LOGS['stage2'] / 'partial_q_learning.txt', lines)


# ---------------------------------------------------------------- stage 3a
def fixed(budget):
    t0 = time.time()
    data = header('stage3_fixed_opponents', dict(
        epsilon=STAGE3_EPS, step_size='1/N(s,a)', seeds=budget['seeds'],
        checkpoints=list(budget['hand_checkpoints']), opponents=list(OPPONENTS)))
    rows = []
    for name, p2 in OPPONENTS.items():
        for seed in budget['seeds']:
            for row in checkpoint_run(name, budget['hand_checkpoints'], STAGE3_EPS, seed):
                best = evaluate(row['policy'], p2)['p1_best_response_ev']
                row['best_possible_ev'] = best
                row['at_best_response'] = row['p1_ev'] >= best - TOL
                rows.append(row)
            print(f'  fixed {name} seed={seed} done ({time.time() - t0:.0f}s)', flush=True)
    data['seconds'] = round(time.time() - t0, 1)
    data['rows'] = rows
    save_curve(data, LOGS['fixed'] / 'fixed_opponents.json')

    lines = [f"Stage 3a: Q-learning vs fixed bots, eps={STAGE3_EPS}, {len(budget['seeds'])} seeds, "
             f"{data['seconds']} s", '',
             f"{'opponent':>12} {'hands':>9} {'at BR':>6} {'median EV':>10} {'best EV':>9} {'median worst':>13}"]
    for name in OPPONENTS:
        for n in budget['hand_checkpoints']:
            rs = [r for r in rows if r['opponent'] == name and r['hands'] == n]
            lines.append(f"{name:>12} {n:>9,} {sum(r['at_best_response'] for r in rs):>3}/{len(rs):<2} "
                         f"{median(r['p1_ev'] for r in rs):>10.5f} {rs[0]['best_possible_ev']:>9.5f} "
                         f"{median(r['p1_worst_case_ev'] for r in rs):>13.5f}")
    write_summary(LOGS['fixed'] / 'fixed_opponents.txt', lines)


# ---------------------------------------------------------------- stage 3b
def selfplay(budget):
    t0 = time.time()
    rounds = budget['selfplay_rounds']
    data = header('stage3_selfplay', dict(
        epsilon=STAGE3_EPS, hands_per_round=budget['batch'], seeds=budget['seeds'],
        round_checkpoints=list(rounds)))
    rows = []
    for method in ('naive', 'learned', 'exact'):
        for seed in (budget['seeds'] if method != 'exact' else [None]):
            rows.extend(selfplay_run(method, rounds, budget['batch'], STAGE3_EPS, seed))
            print(f'  selfplay {method} seed={seed} done ({time.time() - t0:.0f}s)', flush=True)
    data['seconds'] = round(time.time() - t0, 1)
    data['rows'] = rows
    save_curve(data, LOGS['selfplay'] / 'selfplay.json')

    lines = [f"Stage 3b: self-play exploitability, eps={STAGE3_EPS}, {budget['batch']} hands/round, "
             f"{len(budget['seeds'])} seeds, {data['seconds']} s", '',
             f"{'method':>8} {'round':>6} {'hands':>10} {'current (med)':>14} "
             f"{'average (med)':>14} {'average min':>12} {'average max':>12}"]
    for method in ('naive', 'learned', 'exact'):
        for r in rounds:
            rs = [x for x in rows if x['method'] == method and x['round'] == r]
            avg = [x['average_exploitability'] for x in rs]
            lines.append(f"{method:>8} {r:>6} {rs[0]['hands']:>10,} "
                         f"{median(x['current_exploitability'] for x in rs):>14.5f} "
                         f"{median(avg):>14.5f} {min(avg):>12.5f} {max(avg):>12.5f}")
    write_summary(LOGS['selfplay'] / 'selfplay.txt', lines)


def write_summary(path, lines):
    text = '\n'.join(lines) + '\n'
    path.write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['stage2', 'fixed', 'selfplay', 'all'])
    parser.add_argument('--quick', action='store_true', help='tiny smoke-test budget')
    args = parser.parse_args()
    if args.quick:
        for key in LOGS:
            LOGS[key] = LOGS[key] / 'quick'
    for path in LOGS.values():
        path.mkdir(parents=True, exist_ok=True)
    budget = QUICK if args.quick else FULL
    jobs = {'stage2': stage2, 'fixed': fixed, 'selfplay': selfplay}
    for name in (jobs if args.stage == 'all' else [args.stage]):
        jobs[name](budget)


if __name__ == '__main__':
    main()
