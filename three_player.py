"""A separate three-seat Kuhn experiment using OpenSpiel's game and CFR solver."""
import argparse
import json
from importlib.metadata import version

GAME = "kuhn_poker(players=3)"
DEFAULT_CHECKPOINTS = (0, 10, 100, 1000)


def experiment(checkpoints=DEFAULT_CHECKPOINTS):
    """Train once; score the average policy at each full-tree iteration count."""
    import pyspiel

    targets = list(checkpoints)
    if not targets or any(type(n) is not int or n < 0 for n in targets):
        raise ValueError("checkpoints must be nonnegative integers")
    game = pyspiel.load_game(GAME)
    root = game.new_initial_state()
    solver = pyspiel.CFRSolver(game)
    rows = []
    trained = 0
    for target in sorted(set(targets)):
        while trained < target:
            solver.evaluate_and_update_policy()
            trained += 1
        profile = solver.tabular_average_policy()
        ev = pyspiel.expected_returns(root, profile, -1, True)
        # One legal best response per seat; the other two seats stay fixed.
        gain = [pyspiel.TabularBestResponse(game, seat, profile).value(root.history_str()) - value
                for seat, value in enumerate(ev)]
        rows.append({"iteration": trained, "seat_ev": ev, "deviation_gain": gain,
                     "nash_conv": sum(gain)})
    return {"game": GAME, "open_spiel_version": version("open-spiel"),
            "method": "CFR, average policy", "checkpoints": rows,
            "final_policy": solver.tabular_average_policy().policy_table()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS,
                        help="full-tree iterations (default: 0 10 100 1000)")
    args = parser.parse_args()
    if min(args.checkpoints) < 0:
        parser.error("checkpoints must be nonnegative")
    try:
        result = experiment(args.checkpoints)
    except ModuleNotFoundError as err:
        if err.name != "pyspiel":
            raise
        parser.exit(1, "This extension needs OpenSpiel: python -m pip install open-spiel==2.0.2\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
