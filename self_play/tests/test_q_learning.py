import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from common.evaluate import DEALS, evaluate
from self_play.q_learning import OPPONENTS, experiment, greedy_policy, step, td_update, train


class QLearningTest(unittest.TestCase):
    def test_bootstrap_uses_next_states_max_on_nonterminal_only(self):
        q = [[0.0, 0.0] for _ in range(6)]
        visits = [[0, 0] for _ in range(6)]
        q[4] = [-1.0, 1.5]
        td_update(q, visits, 1, 0, reward=0, next_state=4)
        self.assertEqual(q[1][0], 1.5)
        td_update(q, visits, 1, 0, reward=-1, next_state=None)
        self.assertEqual(q[1][0], .25)  # second visit: alpha = 1/2
        self.assertEqual(visits[1], [2, 0])

    def test_all_terminal_paths_all_deals(self):
        rng = random.Random(1)
        for i, j in DEALS:
            with self.subTest(p1_card=i, p2_card=j):
                sign = 1 if i > j else -1
                # Vary the two response tables independently so an index swap is visible.
                p2 = tuple(int(c == j) for c in range(3)) + (0, 0, 0)
                self.assertEqual(step(i, 1, j, p2, rng), (None, 1))
                self.assertEqual(step(i, 0, j, p2, rng), (3+i, 0))
                self.assertEqual(step(3+i, 0, j, p2, rng), (None, -1))
                self.assertEqual(step(3+i, 1, j, p2, rng), (None, 2*sign))
                p2 = (0, 0, 0) + tuple(int(c == j) for c in range(3))
                self.assertEqual(step(i, 0, j, p2, rng), (None, sign))
                self.assertEqual(step(i, 1, j, p2, rng), (None, 2*sign))

    def test_seeded_training_and_visits(self):
        config: dict[str, Any] = dict(episodes=5000, seed=23, epsilon=.2)
        each = train(OPPONENTS['random'], **config)
        self.assertEqual(each, train(OPPONENTS['random'], **config))
        q, visits = each
        self.assertEqual(len(q), 6)
        self.assertTrue(all(len(row) == 2 and min(row) > 0 for row in visits))
        self.assertEqual(sum(sum(row) for row in visits[:3]), config['episodes'])
        self.assertLessEqual(sum(sum(row) for row in visits[3:]), config['episodes'])

    def test_greedy_export_splits_ties_and_drops_exploration(self):
        self.assertEqual(greedy_policy([[0, 1], [0, -1], [0, 0]] * 2), (1, 0, .5)*2)

    def test_learns_fixed_opponents_without_enforcing_equilibrium(self):
        for name, p2 in OPPONENTS.items():
            with self.subTest(opponent=name):
                result = experiment(name, 60_000, .2, 150)
                score = evaluate(result['policy'], p2)
                self.assertAlmostEqual(result['trained_ev_vs_opponent'], score['p1_ev'])
                self.assertAlmostEqual(result['trained_worst_case_ev'], score['p1_worst_case_ev'])
                self.assertGreater(result['trained_ev_vs_opponent'], result['untrained_ev_vs_opponent'])
                self.assertAlmostEqual(result['trained_ev_vs_opponent'], score['p1_best_response_ev'])
                if name == 'random':
                    self.assertAlmostEqual(score['p1_ev'], .5)
                    self.assertGreater(score['p1_security_gap'], .01)

    def test_invalid_config(self):
        bad_configs: list[dict[str, Any]] = [dict(episodes=0), dict(episodes=1.5),
                                             dict(epsilon=-.01), dict(epsilon=float('nan'))]
        for bad in bad_configs:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                train(OPPONENTS['random'], **bad)
        for p2 in ([], [1]*5, [2]*6, [float('nan')]*6):
            with self.subTest(p2=p2), self.assertRaises(ValueError):
                train(p2, episodes=1)

    def test_cli_outputs_one_row_per_opponent_and_seed(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-m', 'self_play.q_learning', '--opponent', 'all', '--episodes', '200',
                                 '--seeds', '0', '1'], capture_output=True, text=True, check=True, cwd=root)
        data = json.loads(result.stdout)
        self.assertEqual(data['episodes_per_run'], 200)
        self.assertEqual([(r['opponent'], r['seed']) for r in data['runs']],
                         [(name, seed) for name in OPPONENTS for seed in (0, 1)])
