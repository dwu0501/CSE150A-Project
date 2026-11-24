import itertools
import random
import unittest

from common.evaluate import evaluate
from hill_climbing.hill_climbing import expected_payoff


class EvaluationTest(unittest.TestCase):
    def test_logged_policy_matches_game_value_but_is_beatable(self):
        result = evaluate([.406, 0, .740, .341])
        self.assertAlmostEqual(result['p1_ev'], -1/18)
        self.assertAlmostEqual(result['p1_worst_case_ev'], -.1883333333333333)
        self.assertAlmostEqual(result['p1_security_gap'], .1327777777777778)

    def test_equilibrium_family(self):
        for alpha in (0, .1, 1/3):
            score = evaluate([alpha, 0, 3*alpha, alpha+1/3])
            for key in ('p1_ev', 'p1_best_response_ev', 'p1_worst_case_ev'):
                self.assertAlmostEqual(score[key], -1/18)
            self.assertAlmostEqual(score['profile_exploitability'], 0)

    def test_four_parameter_adapter_matches_existing_engine(self):
        rng = random.Random(73)
        for _ in range(20):
            params = [rng.random() for _ in range(4)]
            self.assertAlmostEqual(evaluate(params)['p1_ev'], expected_payoff(params))
            self.assertEqual(evaluate(params), evaluate([*params[:3], 0, params[3], 1]))

    def test_replies_against_all_64_legal_pure_strategies_per_seat(self):
        pure = list(itertools.product((0, 1), repeat=6))
        rng = random.Random(73)
        profiles = [(p, pure[-1-i]) for i, p in enumerate(pure)]
        profiles += [([rng.random() for _ in range(6)], [rng.random() for _ in range(6)])
                     for _ in range(10)]
        for p1, p2 in profiles:
            score = evaluate(p1, p2)
            upper = max(evaluate(p, p2)['p1_ev'] for p in pure)
            lower = min(evaluate(p1, p)['p1_ev'] for p in pure)
            self.assertAlmostEqual(score['p1_best_response_ev'], upper)
            self.assertAlmostEqual(score['p1_worst_case_ev'], lower)
            self.assertAlmostEqual(evaluate(score['p1_best_response'], p2)['p1_ev'], upper)
            self.assertAlmostEqual(evaluate(p1, score['p2_best_response'])['p1_ev'], lower)
            self.assertAlmostEqual(score['profile_exploitability'], (upper-lower)/2)

    def test_invalid_probabilities_or_length(self):
        for bad in ([0]*5, [True]*4, [float('nan')]*4, [-1]*4, [float('inf')]*4, [1.1]*4):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluate(bad)
        with self.assertRaises(ValueError):
            evaluate([0]*4, [0]*4)
