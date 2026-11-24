"""Three-seat experiment checks; OpenSpiel is optional for the base project."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("pyspiel"), "install open-spiel for three-player Kuhn")
class ThreePlayerTests(unittest.TestCase):
    def test_uniform_baseline(self):
        from three_player import experiment

        row = experiment([0])["checkpoints"][0]
        # Exact expectations for equiprobable actions and four distinct cards.
        for actual, expected in zip(row["seat_ev"], [15/64, -3/64, -12/64], strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(row["nash_conv"], 33/16)

    def test_saved_profile_reproduces_final_checkpoint(self):
        import pyspiel
        from three_player import experiment

        data = experiment([10, 0, 10])
        self.assertEqual([c["iteration"] for c in data["checkpoints"]], [0, 10])
        first, last = data["checkpoints"]
        game = pyspiel.load_game(data["game"])
        self.assertEqual(game.num_players(), 3)
        self.assertEqual(len(data["final_policy"]), 48)
        policy = pyspiel.TabularPolicy(data["final_policy"])
        self.assertAlmostEqual(last["nash_conv"], pyspiel.nash_conv(game, policy))
        self.assertLess(last["nash_conv"], first["nash_conv"])
        self.assertAlmostEqual(sum(last["seat_ev"]), 0.0)
        self.assertTrue(all(g >= -1e-12 for g in last["deviation_gain"]))
        self.assertEqual(data["final_policy"], experiment([10])["final_policy"])

    def test_rejects_empty_or_invalid_checkpoints(self):
        from three_player import experiment

        for invalid in ([], [-1], [1.5], [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                experiment(invalid)
