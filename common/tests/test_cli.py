import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROW = 'params (bet_J, bet_Q, bet_K, call_Q):[0.406, 0.0, 0.74, 0.341]\n'


def run(*args, text=''):
    return subprocess.run([sys.executable, '-m', 'common.evaluate', *args], input=text, cwd=ROOT,
                          capture_output=True, text=True)


class CommandLineTest(unittest.TestCase):
    def test_single_policy(self):
        result = run('policy', '--p1', '.406', '0', '.74', '.341')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(json.loads(result.stdout)['p1_worst_case_ev'], -.18833333333)

    def test_log_uses_stored_policies_not_simulation_lines(self):
        data = 'exact EV :-0.05556\n\n' + ROW + 'sim EV   :-0.05267(200k-hand check)\n' + ROW
        result = run('log', '-', text=data)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['policies'], 2)
        self.assertAlmostEqual(summary['median_p1_worst_case_ev'], -.18833333333)

    def test_bad_input_fails_instead_of_skipping_a_policy(self):
        for data in ('header only', ROW + 'params (bet_J, bet_Q, bet_K, call_Q):[1,2]\n'):
            with self.subTest(data=data):
                result = run('log', '-', text=data)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Traceback', result.stderr)
