import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import q_learning as ql
from evaluate import evaluate


class FixedCurveTest(unittest.TestCase):
    def test_curve_cli_saves_policies(self):
        cli = Path(__file__).resolve().parents[1] / 'fixed_curve.py'
        with tempfile.TemporaryDirectory() as out:
            cmd = [sys.executable, str(cli), '--out', out, '--checkpoints', '20', '60', '--seeds', '0', '1', '--data-only']
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((Path(out) / 'fixed.json').read_text())
            self.assertEqual(len(data['rows']), 24)  # 4 bots x 2 seeds x (baseline + 2)
            for row in data['rows']:
                self.assertEqual(len(row['policy']), 6)
                self.assertAlmostEqual(evaluate(row['policy'], ql.OPPONENTS[row['opponent']])['p1_ev'], row['p1_ev'])

    def test_snapshots_are_exact_seeded_prefixes(self):
        self.assertTrue(hasattr(ql, 'checkpoint_run'), 'checkpoint_run not implemented')
        hands = (100, 300, 1100)
        for row, n in zip(ql.checkpoint_run('random', hands, .2, 3), hands, strict=True):
            solo = ql.experiment('random', n, .2, 3)
            self.assertEqual((row['hands'], row['seed'], row['opponent']), (n, 3, 'random'))
            self.assertEqual(row['policy'], solo['policy'])
            scores = evaluate(row['policy'], ql.OPPONENTS['random'])
            for metric in ('p1_ev', 'p1_worst_case_ev', 'p1_security_gap'):
                self.assertEqual(row[metric], scores[metric])
