import math
import random

"""Peter learns a best response to a fixed Player 2 with tabular Q-learning.

The states are Peter's information sets (his card + where he is in the hand):
    (card, "start")       -> actions "bet" / "check"
    (Q, "facing_bet")     -> actions "call" / "fold"   (Peter checked, then P2 bet)

Two decisions are solved by logic and hard-coded instead of learned, since they're trivial:
    J facing a bet -> always fold 
    K facing a bet -> always call
So, 4 states x 2 actions = 8 Q-values to learn.
"""

# Card ranks
J, Q, K = 0, 1, 2
CARD_NAMES = "JQK"
CARDS = [J, Q, K]

# All possible deals
DEALS = [(J, Q), (J, K), (Q, J), (Q, K), (K, J), (K, Q)]

# The two actions available
ACTIONS = {
    "start": ["bet", "check"],
    "facing_bet": ["call", "fold"],
}

# Facing a bet with J or K, we already know the answer
FORCED_FACING_BET = {
    J: "fold",
    K: "call",
}

# Every state where Peter has a decision to learn
STATES = []
for card in CARDS:
    STATES.append((card, "start"))
STATES.append((Q, "facing_bet"))


def strategies(q):
    """Player 2's fixed strategy, indexed by P2's card (J, Q, K). q = 1/3 is equilibrium."""
    call_vs_bet = [0.0, q, 1.0]     # P2 facing a bet
    bet_after_chk = [q, 0.0, 1.0]   # P2 after Peter checks
    return call_vs_bet, bet_after_chk


# ------------------------Environment----------------------------------------

def payoff(c1, c2, a0, p2_action, a1=None):
    """Peter's chip result for one finished hand.

    c1, c2     : Peter's card and P2's card
    a0         : Peter's first action ("bet" or "check")
    p2_action  : P2's response ("call"/"fold" after a bet, "bet"/"check" after a check)
    a1         : Peter's second action ("call" or "fold"), only if he checked and P2 bet
    """
    
    if c1 > c2:
        s = 1       # Peter wins a showdown
    else:
        s = -1      # Peter loses a showdown

    if a0 == "bet":
        if p2_action == "call":
            return 2 * s
        else:
            return 1                 # P2 folded

    # Peter checked
    if p2_action == "check":
        return s                     # check-check showdown

    # Peter checked and P2 bet
    if a1 == "call":
        return 2 * s
    else:
        return -1                    # Peter folded


def play_hand(Qt, q, rng, eps):
    """Play one hand, with Peter choosing actions from his Q-table.

    eps is the exploration rate (use 0 to always play the best-known action).
    Returns Peter's trajectory [(state, action), ...] and his reward.
    """

    p2_call, p2_bet = strategies(q)
    c1, c2 = rng.choice(DEALS)
    trajectory = []

    # Peter's first decision
    s0 = (c1, "start")
    a0 = choose_action(Qt, s0, eps, rng)
    trajectory.append((s0, a0))

    if a0 == "bet":
        if rng.random() < p2_call[c2]:
            p2_action = "call"
        else:
            p2_action = "fold"
        
        reward = payoff(c1, c2, a0, p2_action)
        return trajectory, reward

    # Peter checked, does P2 bet?
    if rng.random() < p2_bet[c2]:
        # Peter's second decision
        if c1 in FORCED_FACING_BET:
            # J or K
            a1 = FORCED_FACING_BET[c1]
        else:
            # Q, we actually learn here
            s1 = (c1, "facing_bet")
            a1 = choose_action(Qt, s1, eps, rng)
            trajectory.append((s1, a1))

        reward = payoff(c1, c2, a0, "bet", a1)
        return trajectory, reward

    reward = payoff(c1, c2, a0, "check")
    return trajectory, reward


# ------------------------Q-Learning functions----------------------------------------

def new_table():
    """Initialize a Q-value of 0 for every (state, action) pair."""
    table = {}
    for s in STATES:
        node = s[1]
        for a in ACTIONS[node]:
            table[(s, a)] = 0.0
    return table


def greedy(Qt, s):
    """The action with the higher Q-value in state s (first action wins ties)."""
    node = s[1]
    first_action = ACTIONS[node][0]
    second_action = ACTIONS[node][1]

    if Qt[(s, first_action)] >= Qt[(s, second_action)]:
        return first_action
    else:
        return second_action


def best_value(Qt, s):
    """The higher of the two Q-values in state s."""
    best_action = greedy(Qt, s)

    return Qt[(s, best_action)]


def choose_action(Qt, s, eps, rng):
    """Choose_action is epsilon-greedy, a random action with probability eps, otherwise the greedy one."""
    if eps > 0:
        if rng.random() < eps:
            node = s[1]
            return rng.choice(ACTIONS[node])   # explore
    return greedy(Qt, s)                       # greedy


def q_update(Qt, N, trajectory, reward, lr):
    """One Q-learning update per decision Peter made this hand

    Last decision:              target = reward (the hand is over)
    Check -> then facing a bet: target = max_a Q(facing_bet, a)  

    lr is a fixed step size, or "visits" for 1/N(s,a)
    """
    for t in range(len(trajectory)):
        s, a = trajectory[t]

        is_last_decision = (t == len(trajectory) - 1)
        if is_last_decision:
            target = reward
        else:
            next_state = trajectory[t + 1][0]
            target = best_value(Qt, next_state)

        N[(s, a)] += 1
        if lr == "visits":
            step = 1 / N[(s, a)]
        else:
            step = lr

        Qt[(s, a)] = Qt[(s, a)] + step * (target - Qt[(s, a)])


def train(q=1/3, hands=200_000, lr="visits", eps=0.1, seed=0):
    """Train Peter's Q-table by playing `hands` hands against P2."""
    rng = random.Random(seed)
    Qt = new_table()

    N = {}                      # how many times each (state, action) was updated
    for key in Qt:
        N[key] = 0

    for h in range(hands):
        trajectory, reward = play_hand(Qt, q, rng, eps)
        q_update(Qt, N, trajectory, reward, lr)

    return Qt


# ------------------------Results----------------------------------------

def simulated_ev(Qt, q, hands=200_000, seed=123):
    """Monte Carlo EV of the greedy policy"""
    rng = random.Random(seed)

    rewards = []
    for _ in range(hands):
        trajectory, reward = play_hand(Qt, q, rng, eps=0)
        rewards.append(reward)

    mean = sum(rewards) / hands

    squared_diffs = 0.0
    for r in rewards:
        squared_diffs += (r - mean) ** 2
    variance = squared_diffs / (hands - 1)

    half_width = 1.96 * math.sqrt(variance / hands)
    return mean, half_width


def to_params(Qt):
    """Greedy Q-table -> the old 4-number description [bet_J, bet_Q, bet_K, call_Q]."""
    params = []
    for c in CARDS:
        if greedy(Qt, (c, "start")) == "bet":
            params.append(1.0)
        else:
            params.append(0.0)

    if greedy(Qt, (Q, "facing_bet")) == "call":
        params.append(1.0)
    else:
        params.append(0.0)

    return params


def print_table(Qt):
    """Print every learned Q-value."""
    print(f"    {'state':<14}{'action':<7}{'Q-value':>9}")
    for s in STATES:
        card, node = s
        name = f"{CARD_NAMES[card]}, {node}"
        for a in ACTIONS[node]:
            print(f"    {name:<14}{a:<7}{Qt[(s, a)]:>9.3f}")


# ------------------------Main----------------------------------------

def main():
    # Train Peter against three versions of P2 and show what he learned
    for q in [0.2, 1/3, 0.5]:
        Qt = train(q=q)
        mean, ci = simulated_ev(Qt, q)

        print(f"=== P2 q = {q:.3f} ===")
        print_table(Qt)
        print("  greedy policy (bet_J, bet_Q, bet_K, call_Q):", to_params(Qt))
        print(f"  simulated EV {mean:+.4f} +/- {ci:.4f}\n")


if __name__ == "__main__":
    main()
