"""Train P1 from sampled hands against a fixed P2, then score the frozen policy."""

import argparse
import json
import random

from common.evaluate import DEALS, EQUILIBRIUM_P2, evaluate

OPPONENTS = {
    'always-pass': (0.0,) * 6,   # check / fold
    'always-bet': (1.0,) * 6,    # bet / call
    'random': (0.5,) * 6,
    'equilibrium': EQUILIBRIUM_P2,
}


def td_update(q, visits, state, action, reward, next_state):
    """One undiscounted Q-learning update; the step size is 1/visit count."""
    target = reward if next_state is None else reward + max(q[next_state])
    visits[state][action] += 1
    q[state][action] += (target - q[state][action]) / visits[state][action]


def step(state, action, other_card, p2, rng):
    """Environment only. Advance past P2 to the next P1 decision or terminal payoff."""
    card = state % 3
    showdown = 1 if card > other_card else -1
    if state >= 3:  # facing a bet after checking: fold or call
        return None, 2 * showdown if action else -1
    if action:  # P1 opens; P2 calls or folds
        return None, 2 * showdown if rng.random() < p2[3 + other_card] else 1
    if rng.random() < p2[other_card]:  # P2 bets after P1 checks
        return 3 + card, 0
    return None, showdown


def greedy_policy(q):
    """Probability of action 1 (bet/call) at each information set; ties split evenly."""
    return tuple(1.0 if a1 > a0 else 0.0 if a0 > a1 else 0.5 for a0, a1 in q)


def train(p2, *, episodes=100_000, epsilon=0.2, seed=150, on_hand=None):
    """Return Q-values and visit counts. Only the environment sees P2's card/policy."""
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon must be between 0 and 1')
    p2 = tuple(p2)
    if len(p2) != 6 or not all(0 <= x <= 1 for x in p2):
        raise ValueError('p2 must contain six probabilities in [0, 1]')
    rng = random.Random(seed)
    q = [[0.0, 0.0] for _ in range(6)]
    visits = [[0, 0] for _ in range(6)]
    for hand in range(1, episodes + 1):
        state, other_card = rng.choice(DEALS)
        while state is not None:
            passive, aggressive = q[state]
            if rng.random() < epsilon or passive == aggressive:
                action = rng.randrange(2)
            else:
                action = int(aggressive > passive)
            next_state, reward = step(state, action, other_card, p2, rng)
            td_update(q, visits, state, action, reward, next_state)
            state = next_state
        if on_hand is not None:
            on_hand(hand, q)
    return q, visits


def checkpoint_run(name, checkpoints, epsilon, seed):
    """Freeze greedy policies after completed hands of ONE uninterrupted run."""
    points = tuple(checkpoints)
    if (not points or any(type(n) is not int or n < 1 for n in points)
            or points != tuple(sorted(set(points)))):
        raise ValueError('checkpoints must be positive and strictly increasing')
    wanted = set(points)
    rows = []
    p2 = OPPONENTS[name]

    def snapshot(hand, q):
        if hand in wanted:
            policy = greedy_policy(q)
            scores = evaluate(policy, p2)
            rows.append(dict(opponent=name, seed=seed, hands=hand, policy=policy,
                             **{k: scores[k] for k in ('p1_ev', 'p1_worst_case_ev', 'p1_security_gap')}))

    train(p2, episodes=points[-1], epsilon=epsilon, seed=seed, on_hand=snapshot)
    return rows


def experiment(name, episodes, epsilon, seed):
    """The evaluator is used after training, never to choose or update actions."""
    p2 = OPPONENTS[name]
    q, visits = train(p2, episodes=episodes, epsilon=epsilon, seed=seed)
    policy = greedy_policy(q)
    scores = evaluate(policy, p2)
    return {
        'opponent': name, 'seed': seed, 'policy': policy, 'q': q, 'visits': visits,
        'untrained_ev_vs_opponent': evaluate((0.5,) * 6, p2)['p1_ev'],
        'trained_ev_vs_opponent': scores['p1_ev'],
        'best_possible_ev_vs_opponent': scores['p1_best_response_ev'],
        'trained_worst_case_ev': scores['p1_worst_case_ev'],
        'p1_security_gap': scores['p1_security_gap'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opponent', choices=['all', *OPPONENTS], default='random')
    parser.add_argument('--episodes', type=int, default=100_000, help='hands per opponent and seed')
    parser.add_argument('--epsilon', type=float, default=0.2, help='exploration during training only')
    parser.add_argument('--seeds', type=int, nargs='+', default=[150])
    args = parser.parse_args()
    opponents = list(OPPONENTS) if args.opponent == 'all' else [args.opponent]
    try:
        runs = [experiment(name, args.episodes, args.epsilon, seed)
                for name in opponents for seed in args.seeds]
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({'episodes_per_run': args.episodes, 'epsilon': args.epsilon, 'runs': runs}, indent=2))


if __name__ == '__main__':
    main()
