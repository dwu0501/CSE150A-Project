"""Reproduce the fixed-opponent checkpoint data and plot (matplotlib for plotting only)."""
import argparse

from pathlib import Path
from statistics import median

from evaluate import GAME_VALUE, evaluate
from curve_io import save_curve
from q_learning import OPPONENTS, checkpoint_run


def collect(points, seeds, epsilon):
    rows = []
    for name, p2 in OPPONENTS.items():
        uniform = (0.5,) * 6
        scores = evaluate(uniform, p2)
        for seed in seeds:
            rows.append(dict(opponent=name, seed=seed, hands=0, policy=uniform,
                             **{k: scores[k] for k in ('p1_ev', 'p1_worst_case_ev', 'p1_security_gap')}))
            rows.extend(checkpoint_run(name, points, epsilon, seed))
    return dict(checkpoints=[0, *points], seeds=seeds, epsilon=epsilon, rows=rows)


def plot(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, panels = plt.subplots(2, 2, figsize=(9, 6.6), sharey=True, layout='constrained')
    ticks = data['checkpoints']
    x = range(len(ticks))  # categorical, so the untrained baseline is visible
    for ax, name in zip(panels.flat, OPPONENTS):
        rows = [r for r in data['rows'] if r['opponent'] == name]
        for col, label, color, marker in [('p1_ev', 'Training opponent', '#176993', 'o'),
                                          ('p1_worst_case_ev', 'Best counter', '#b1451c', 's')]:
            ys = [[r[col] for r in rows if r['hands'] == n] for n in ticks]
            ax.fill_between(x, [min(y) for y in ys], [max(y) for y in ys], color=color, alpha=.17)
            ax.plot(x, [median(y) for y in ys], color=color, marker=marker, label=label)
        ax.axhline(GAME_VALUE, color='#444444', ls=':', lw=1.1, label='Equilibrium value')
        ax.set(title=name, xticks=list(x), xticklabels=[f'{n:,}' for n in ticks])
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Fixed opponents: specialization versus worst-case payoff')
    fig.supxlabel('Training hands (checkpoint spacing not to scale)')
    fig.supylabel('Player 1 chips per hand · higher is better')
    panels.flat[0].legend(fontsize=8, loc='best')
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', type=int, default=[1000, 10000, 100000])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--data-only', action='store_true')
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error('duplicate seeds')
    try:
        data = collect(a.checkpoints, a.seeds, a.epsilon)
    except ValueError as exc:
        p.error(str(exc))
    a.out.mkdir(parents=True, exist_ok=True)
    save_curve(data, a.out / 'fixed.json')
    if not a.data_only:
        plot(data, a.out / 'fixed.png')


if __name__ == '__main__':
    main()
