"""Two-seat Kuhn experiments. Policies are six bet/call probabilities per seat."""
import random

from evaluate import DEALS, evaluate
from q_learning import greedy_policy, td_update

UNIFORM = (.5,) * 6
METHODS = ('naive', 'learned', 'exact')


def hand(deal, act1, act2):
    """Deal and settlement are private to the environment. Return P1's net chips and local paths."""
    i, j = deal
    s = 1 if i > j else -1
    a = act1(i)
    p1 = [(i, a)]
    if a:
        b = act2(3+j)
        return (2*s if b else 1), p1, [(3+j, b)]
    b = act2(j)
    if not b:
        return s, p1, [(j, b)]
    c = act1(3+i)
    return (2*s if c else -1), p1 + [(3+i, c)], [(j, b)]


class Learner:
    def __init__(self, rng, epsilon):
        self.rng, self.epsilon = rng, epsilon
        self.q = [[0., 0.] for _ in range(6)]
        self.visits = [[0, 0] for _ in range(6)]

    def act(self, state):
        a, b = self.q[state]
        if self.rng.random() < self.epsilon or a == b:
            return self.rng.randrange(2)
        return int(b > a)

    def update(self, path, payoff):
        for index, (state, action) in enumerate(path):
            nxt = path[index+1][0] if index+1 < len(path) else None
            td_update(self.q, self.visits, state, action, payoff if nxt is None else 0, nxt)

    def policy(self):
        return greedy_policy(self.q)


class Average:
    """Behavior strategy of a uniform mixture of historical policies, weighted by own reach."""
    def __init__(self, seat):
        self.seat = seat
        self.total = [0.] * 6
        self.mass = [0.] * 6

    def add(self, policy):
        for i, prob in enumerate(policy):
            # P1 can call after checking only if *that same policy* checked with the card.
            # P2 acts just once, so its own reach weight is always 1, not P1's bet rate.
            w = 1 - policy[i-3] if self.seat == 0 and i >= 3 else 1.
            self.total[i] += w * prob
            self.mass[i] += w

    def policy(self):
        return tuple(v/w if w else .5 for v, w in zip(self.total, self.mass))


def learned_reply(seat, opp, budget, rng, epsilon):
    learner = Learner(rng, epsilon)
    def fixed(state):
        return int(rng.random() < opp[state])
    for _ in range(budget):
        deal = rng.choice(DEALS)
        if seat == 0:
            gain, path, _ = hand(deal, learner.act, fixed)
            learner.update(path, gain)
        else:
            gain, _, path = hand(deal, fixed, learner.act)
            learner.update(path, -gain)
    return learner.policy()


def run(method, rounds, batch, epsilon, seed):
    """Checkpoint in update rounds. Naive/learned spend batch sampled hands per round; exact spends none."""
    rounds = tuple(rounds)
    if (method not in METHODS or not rounds or any(type(r) is not int or r < 0 for r in rounds)
            or tuple(sorted(set(rounds))) != rounds or type(batch) is not int or batch < 2
            or batch % 2 or not 0 <= epsilon <= 1):
        raise ValueError('invalid method, increasing round checkpoints, even batch size or epsilon')
    if method == 'exact':
        seed = None  # full-tree calculation, no RNG or sampling
    rng = random.Random(seed) if method != 'exact' else None
    averages = (Average(0), Average(1))
    current = (UNIFORM, UNIFORM)
    agents = (Learner(rng, epsilon), Learner(rng, epsilon)) if method == 'naive' else None
    if method != 'naive':
        for avg in averages:
            avg.add(UNIFORM)  # one initial fictitious-play policy before the first reply
    rows = []
    wanted = set(rounds)
    for r in range(rounds[-1] + 1):
        if r and agents is not None:
            assert rng is not None
            for _ in range(batch):
                for avg, agent in zip(averages, agents):
                    avg.add(agent.policy())  # pre-hand greedy target, not epsilon behavior
                gain, path1, path2 = hand(rng.choice(DEALS), agents[0].act, agents[1].act)
                agents[0].update(path1, gain)
                agents[1].update(path2, -gain)
            current = (agents[0].policy(), agents[1].policy())
        elif r:
            # Freeze BOTH old averages before calculating either new reply.
            p1, p2 = (a.policy() for a in averages)
            if method == 'exact':
                br = evaluate(p1, p2)
                current = (br['p1_best_response'], br['p2_best_response'])
            else:
                current = (learned_reply(0, p2, batch//2, rng, epsilon),
                           learned_reply(1, p1, batch//2, rng, epsilon))
            for avg, reply in zip(averages, current):
                avg.add(reply)
        if r in wanted:
            average = tuple(a.policy() for a in averages)
            rows.append(dict(method=method, seed=seed, round=r, hands=0 if method == 'exact' else r*batch,
                             current=current, average=average,
                             current_exploitability=evaluate(*current)['profile_exploitability'],
                             average_exploitability=evaluate(*average)['profile_exploitability']))
    return rows
