import random

# Card ranks
J, Q, K = 0, 1, 2

# All possible deals
DEALS = [(J, Q), (J, K), (Q, J), (Q, K), (K, J), (K, Q)]


def strategies(q):
    """Player 2's fixed equilibrium strategy, indexed by P2's card (J, Q, K)."""
    call_vs_bet   = [0.0, q, 1.0]   # P2 facing a bet
    bet_after_chk = [q, 0.0, 1.0]   # P2 after P1 checks
    return call_vs_bet, bet_after_chk


def run_game(bet_J, bet_Q, bet_K, call_Q, iterations=100, q=1/3):
    """Simulate Kuhn poker; return Player 1's average payoff per hand.

    P1's strategy depends only on P1's own card (legal information set):
      bet_J, bet_Q, bet_K : probability of betting with each card
      call_Q              : probability of calling a bet after checking with Q
    P1 always folds J and always calls with K when facing a bet after a check
    (the other actions are dominated).
    """

    bet  = [bet_J, bet_Q, bet_K]
    call = [0.0, call_Q, 1.0]
    p2_call, p2_bet = strategies(q)

    total = 0
    for _ in range(iterations):
        c1, c2 = random.choice(DEALS)
        sign = 1 if c1 > c2 else -1          # +1 if P1 wins at showdown

        if random.random() < bet[c1]:                    # P1 bets
            if random.random() < p2_call[c2]:            # P2 calls
                total += 2 * sign
            else:                                        # P2 folds
                total += 1
        else:                                            # P1 checks
            if random.random() < p2_bet[c2]:             # P2 bets
                if random.random() < call[c1]:           # P1 calls
                    total += 2 * sign
                else:                                    # P1 folds
                    total -= 1
            else:                                        # check-check
                total += sign

    return total / iterations


def expected_payoff(params, q=1/3):
    """Exact average payoff per hand (every coin flip replaced by its probability)."""
    bet_J, bet_Q, bet_K, call_Q = params
    bet  = [bet_J, bet_Q, bet_K]
    call = [0.0, call_Q, 1.0]
    p2_call, p2_bet = strategies(q)

    total = 0.0
    for c1, c2 in DEALS:
        # s = +1 if P1 wins at showdown, pc/pb = P2's call/bet probabilities
        s = 1 if c1 > c2 else -1

        pc, pb = p2_call[c2], p2_bet[c2]

        # P1's EV for each action given this deal
        bet_ev   = pc * 2 * s + (1 - pc) * 1
        check_ev = pb * (call[c1] * 2 * s - (1 - call[c1])) + (1 - pb) * s

        # Weight each action by how often P1 takes it
        total += bet[c1] * bet_ev + (1 - bet[c1]) * check_ev

    return total / 6   # average over the 6 deals


def hill_climb(params, ev_fn=expected_payoff, lr=0.005, max_sweeps=1000):
    """Coordinate hill climbing on [0,1]^n; keeps only strict improvements."""
    params = list(params)
    best = ev_fn(params)
    for _ in range(max_sweeps):
        improved = False   # tracks whether this sweep found any gain
        for i in range(len(params)):
            for step in (lr, -lr):   # try up, then down
                old = params[i]
                params[i] = min(1.0, max(0.0, old + step))
                ev = ev_fn(params)
                if ev > best + 1e-12:
                    best, improved = ev, True
                    break
                params[i] = old   # no gain, revert
        if not improved:   # converged!
            break
    return params, best


def main():
    # Random starting strategy, then hill_climb to a best response
    start = [random.random() for _ in range(4)]
    params, ev = hill_climb(start)
    
    print("params (bet_J, bet_Q, bet_K, call_Q):", [round(p, 3) for p in params])
    print("exact EV :", round(ev, 5), " (equilibrium value is -1/18 =", round(-1/18, 5), ")")
    print("sim EV   :", round(run_game(*params, iterations=200_000), 5), "(200k-hand check)")


if __name__ == "__main__":
    main()
