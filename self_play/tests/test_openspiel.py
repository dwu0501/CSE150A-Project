"""Optional independent check: python -m pip install open-spiel==2.0.2."""

import importlib.util
import random
import unittest

from common.evaluate import EQUILIBRIUM_P2, evaluate
from self_play.q_learning import OPPONENTS, greedy_policy, train
from self_play.self_play import METHODS, run


@unittest.skipUnless(importlib.util.find_spec('pyspiel'), 'optional open-spiel not installed')
class OpenSpielTest(unittest.TestCase):
    def test_exact_ev_best_replies_and_exploitability(self):
        import pyspiel
        from open_spiel.python import policy
        from open_spiel.python.algorithms import expected_game_score, exploitability

        game = pyspiel.load_game('kuhn_poker', {'players': 2})
        rng = random.Random(150)
        profiles = [((0, 0, 0, 0, 1/3, 1), EQUILIBRIUM_P2),
                    ((.406, 0, .74, 0, .341, 1), EQUILIBRIUM_P2),
                    ((0,)*6, (1,)*6), ((1,)*6, (0,)*6)]
        profiles += [(rng.choices([0, 1], k=6), rng.choices([0, 1], k=6)) for _ in range(8)]
        profiles += [([rng.random() for _ in range(6)], [rng.random() for _ in range(6)])
                     for _ in range(12)]
        # Independently verify the scores of real learned policies, including unreachable states.
        for opponent in OPPONENTS.values():
            q, _ = train(opponent, episodes=60_000, seed=150)
            profiles.append((greedy_policy(q), opponent))
        for method in METHODS:
            for row in run(method, (1, 8), 1000, .2, 3):
                profiles.extend([row['current'], row['average']])
        for p1, p2 in profiles:
            with self.subTest(p1=p1, p2=p2):
                table = policy.TabularPolicy(game)
                # OpenSpiel card IDs 0,1,2 = J,Q,K; actions 0,1 = pass,bet.
                # e.g. '1pb' = holding Q after check-bet, so action 1 calls.
                for card in range(3):
                    for suffix, prob in [('', p1[card]), ('pb', p1[3+card]),
                                         ('p', p2[card]), ('b', p2[3+card])]:
                        table.policy_for_key(str(card) + suffix)[:] = [1-prob, prob]
                ours = evaluate(p1, p2)
                ev = expected_game_score.policy_value(game.new_initial_state(), [table, table])
                self.assertAlmostEqual(ours['p1_ev'], ev[0], places=12)
                for player, key, sign in [(0, 'p1_best_response_ev', 1), (1, 'p1_worst_case_ev', -1)]:
                    reply = exploitability.best_response(game, table, player)
                    self.assertAlmostEqual(ours[key], sign * reply['best_response_value'], places=12)
                self.assertAlmostEqual(ours['profile_exploitability'],
                                       exploitability.exploitability(game, table), places=12)
