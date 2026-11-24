"""Exact evaluation of two-player Kuhn poker policies (chips per hand)."""

import argparse
import json
import statistics
import sys
from itertools import permutations

CARDS = range(3)  # J, Q, K
DEALS = tuple(permutations(CARDS, 2))
GAME_VALUE = -1 / 18  # P1's equilibrium payoff
EQUILIBRIUM_P2 = (1/3, 0, 1, 0, 1/3, 1)  # bets after check, calls vs bet


def _policy(values, seat):
    values = tuple(values)
    if seat == 1 and len(values) == 4:
        # logic.py fixes call_J = 0 and call_K = 1.
        values = (*values[:3], 0, values[3], 1)
    if len(values) != 6:
        raise ValueError(f'P{seat} needs {"4 or 6" if seat == 1 else "6"} probabilities')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 for x in values):
        raise ValueError(f'P{seat} probabilities must be finite numbers in [0, 1]')
    return values


def _actions(i, j, call, p2):
    """P1 payoffs for opening bet/check on a single deal, with a given call rate."""
    s = 1 if i > j else -1
    bet = (1 - p2[3+j]) + p2[3+j] * 2*s
    check = (1 - p2[j]) * s + p2[j] * (call * 2*s - (1-call))
    return bet, check


def _payoff(p1, p2):
    total = 0.0
    for i, j in DEALS:
        bet, check = _actions(i, j, p1[3+i], p2)
        total += p1[i]*bet + (1-p1[i])*check
    return total / len(DEALS)


def _best_p1(p2):
    # Work backward. Facing a bet, call rather than fold if its gain is
    # positive, weighted by how often each hidden card bets after our check.
    bets, calls = [], []
    for i in CARDS:
        others = [j for j in CARDS if j != i]
        call_gain = sum(p2[j] * (1 + (2 if i > j else -2)) for j in others)
        call = int(call_gain > 0)
        choices = [_actions(i, j, call, p2) for j in others]
        bets.append(int(sum(bet - check for bet, check in choices) > 0))
        calls.append(call)
    return tuple(bets + calls)


def _best_p2(p1):
    # P2 minimizes P1's return. Condition on P1's public action by weighting
    # both possible hidden cards by its probability; normalization cancels.
    bets, calls = [], []
    for j in CARDS:
        bet_delta = call_delta = 0.0
        for i in CARDS:
            if i == j:
                continue
            s = 1 if i > j else -1
            call_delta += p1[i] * (2*s - 1)  # call vs fold facing an open bet
            p1_call = p1[3+i]
            bet_delta += (1-p1[i]) * (p1_call*2*s - (1-p1_call) - s)
        bets.append(int(bet_delta < 0))
        calls.append(int(call_delta < 0))
    return tuple(bets + calls)


def evaluate(p1, p2=EQUILIBRIUM_P2):
    """Score a strategy pair without sampling. EVs below use P1's sign.

    For either seat: (bet_J,Q,K, call_J,Q,K). P1 may also use logic.py's
    four-parameter format. Returns concrete best replies as six probabilities.
    """
    p1, p2 = _policy(p1, 1), _policy(p2, 2)
    br1, br2 = _best_p1(p2), _best_p2(p1)
    ceiling, floor = _payoff(br1, p2), _payoff(p1, br2)
    # Clip negative roundoff at equilibrium; don't round reported payoffs.
    nash_conv = max(0.0, ceiling - floor)
    return {
        'p1_ev': _payoff(p1, p2),
        'p1_best_response_ev': ceiling,
        'p1_worst_case_ev': floor,
        'p1_security_gap': max(0.0, GAME_VALUE - floor),
        'nash_conv': nash_conv,
        'profile_exploitability': nash_conv / 2,
        'p1_best_response': br1,
        'p2_best_response': br2,
    }


def summarize_log(lines):
    """Rescore rounded policy rows from run_log/log.txt, ignoring simulated EVs."""
    fixed_evs, floors = [], []
    for line_no, line in enumerate(lines, 1):
        if not line.lstrip().startswith('params'):
            continue
        label, _, payload = line.strip().partition(':')
        try:
            if label != 'params (bet_J, bet_Q, bet_K, call_Q)':
                raise ValueError('unexpected policy column order')
            params = json.loads(payload)
            if not isinstance(params, list) or len(params) != 4:
                raise ValueError('expected four logged probabilities')
            score = evaluate(params)
        except ValueError as exc:
            raise ValueError(f'line {line_no}: {exc}') from exc
        fixed_evs.append(score['p1_ev'])
        floors.append(score['p1_worst_case_ev'])
    if not floors:
        raise ValueError('no policy rows found')
    return {'policies': len(floors), 'mean_p1_ev_vs_equilibrium': statistics.mean(fixed_evs),
            'min_p1_worst_case_ev': min(floors), 'median_p1_worst_case_ev': statistics.median(floors),
            'max_p1_worst_case_ev': max(floors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    single = modes.add_parser('policy', help='score a pair of policies')
    single.add_argument('--p1', type=float, nargs='+', required=True,
                        help='four logic.py parameters, or six: three bets then three calls (J,Q,K)')
    single.add_argument('--p2', type=float, nargs=6, default=EQUILIBRIUM_P2,
                        help='three bets after check, then three calls facing a bet (J,Q,K)')
    batch = modes.add_parser('log', help='rescore policy rows from the run_log branch')
    batch.add_argument('file', help='path to log.txt or - for stdin')
    args = parser.parse_args()
    try:
        if args.mode == 'policy':
            result = evaluate(args.p1, args.p2)
        elif args.file == '-':
            result = summarize_log(sys.stdin)
        else:
            with open(args.file, encoding='utf-8') as stream:
                result = summarize_log(stream)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
