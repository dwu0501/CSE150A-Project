import random
import unittest

from evaluate import DEALS, evaluate


class SelfPlayTest(unittest.TestCase):
    def test_p2_reward_sign_finds_best_reply(self):
        import self_play as sp
        uniform = (.5,) * 6
        for seat in (0, 1):
            reply = sp.learned_reply(seat, uniform, 60000, random.Random(1), .2)
            scores = evaluate(reply, uniform) if seat == 0 else evaluate(uniform, reply)
            self.assertAlmostEqual(scores['p1_ev'], scores['p1_best_response_ev' if seat == 0 else 'p1_worst_case_ev'])

    def test_export_uses_one_deterministic_reference(self):
        from self_curve import collect
        rows = collect(2, 10, .2, [0,1])['rows']
        self.assertEqual(len(rows), 15)
        self.assertEqual({r['seed'] for r in rows if r['method']=='exact'}, {None})

    def test_paths_use_own_card_and_zero_sum_rewards(self):
        import self_play as sp
        for i, j in DEALS:
            for open_bet, after_check, facing_bet in ((1,0,0),(1,0,1),(0,0,0),(0,1,0),(0,1,1)):
                p1 = lambda state: open_bet if state < 3 else facing_bet
                p2 = lambda state: after_check if state < 3 else facing_bet
                reward, path1, path2 = sp.hand((i,j), p1, p2)
                sign = 1 if i > j else -1
                expected = (2*sign if facing_bet else 1) if open_bet else ((2*sign if facing_bet else -1) if after_check else sign)
                self.assertEqual(reward, expected)
                self.assertEqual(path1, [(i,open_bet)] + ([(i+3,facing_bet)] if not open_bet and after_check else []))
                self.assertEqual(path2, [(j+3, facing_bet)] if open_bet else [(j,after_check)])

    def test_average_equals_mixture_in_payoff_not_pointwise(self):
        import self_play as sp
        rng = random.Random(12)
        a, b, opp = [tuple(rng.random() for _ in range(6)) for _ in range(3)]
        for seat in (0,1):
            avg = sp.Average(seat)
            avg.add(a); avg.add(b)
            val = lambda p: evaluate(p,opp)['p1_ev'] if seat == 0 else evaluate(opp,p)['p1_ev']
            self.assertAlmostEqual(val(avg.policy()), (val(a)+val(b))/2, places=12)

    def test_checkpoints_rescore_and_hand_budgets(self):
        import self_play as sp
        for method in ('naive', 'learned', 'exact'):
            runner = lambda: sp.run(method, [0, 1, 5], 100, .2, 2)
            rows = runner()
            self.assertEqual(rows, runner())
            self.assertEqual([r['hands'] for r in rows], [0,100,500] if method != 'exact' else [0]*3)
            for row in rows:
                for k in ('current','average'):
                    p1,p2 = row[k]
                    self.assertEqual(evaluate(p1,p2)['profile_exploitability'], row[k+'_exploitability'])
            if method == 'exact':
                reference=evaluate((.5,)*6,(.5,)*6)
                self.assertEqual(rows[1]['current'], (reference['p1_best_response'],reference['p2_best_response']))
            else:
                self.assertNotEqual(rows[-1]['current'][1], (.5,)*6)
