"""Reproduce the two-seat comparisons; plotting is optional."""
import argparse

from pathlib import Path
from statistics import median

from self_play.self_play import METHODS, run
from common.curve_io import save_curve


def collect(rounds, batch, epsilon, seeds):
    if rounds < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('use positive rounds and distinct seeds')
    checkpoints = range(rounds+1)
    rows = [row for method in METHODS for seed in (seeds if method != 'exact' else [None])
            for row in run(method, checkpoints, batch, epsilon, seed)]
    return dict(rounds=rounds, hands_per_sampled_round=batch, epsilon=epsilon, seeds=seeds, rows=rows)


def plot(data, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, layout='constrained')
    for ax, method, title in zip(axes, METHODS, ('Naïve self-play', 'Fictitious: learned replies', 'Fictitious: exact replies')):
        rows = [r for r in data['rows'] if r['method'] == method]
        points = range(data['rounds']+1)
        x = [r if method == 'exact' else r*data['hands_per_sampled_round']/1000 for r in points]
        for key, color in [('current', '#777777'), ('average', '#006963')]:
            ys = [[v[key+'_exploitability'] for v in rows if v['round'] == r] for r in points]
            ax.fill_between(x, [min(y) for y in ys], [max(y) for y in ys], color=color, alpha=.13)
            ax.plot(x, [median(y) for y in ys], color=color, lw=1.8, label=key.capitalize())
        ax.set(title=title, xlabel='Exact updates (no sampled hands)' if method == 'exact' else 'Training hands (thousands)', ylim=(0,None))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Profile exploitability · chips/hand')
    axes[0].legend()
    fig.suptitle('Two-player strategies: lower exploitability is better')
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--batch', type=int, default=1000, help='sampled hands per round, even; split between seats in fictitious play')
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--out', type=Path, default=Path('self_play/results'))
    p.add_argument('--data-only', action='store_true')
    a = p.parse_args()
    try:
        data = collect(a.rounds, a.batch, a.epsilon, a.seeds)
    except ValueError as exc:
        p.error(str(exc))
    a.out.mkdir(parents=True, exist_ok=True)
    save_curve(data, a.out / 'selfplay.json')
    if not a.data_only:
        plot(data, a.out/'selfplay.png')


if __name__ == '__main__':
    main()
