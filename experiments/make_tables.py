"""Turn the JSON logs from run_experiments.py into LaTeX table rows and pgfplots data.

    python experiments/make_tables.py > experiments/logs/latex_snippets.txt

The output is pasted into writeup/writeup_v1.2.tex (the report inlines its plot data).
"""

import json
from pathlib import Path
from statistics import median

LOGS = Path(__file__).resolve().parent / 'logs'


def load(name):
    return json.loads((LOGS / name).read_text(encoding='utf-8'))


def frac(x):
    """Pretty-print a chip value as a LaTeX fraction when it is a simple one."""
    for den in (1, 2, 3, 4, 6, 9, 12, 18, 30, 36):
        num = round(x * den)
        if abs(num / den - x) < 1e-9:
            if num == 0:
                return '$0$'
            sign = '-' if num < 0 else '+'
            return f'${sign}{abs(num)}$' if den == 1 else f'${sign}{abs(num)}/{den}$'
    return f'${x:+.4f}$'


def stage2_table():
    d = load('stage2_partial_q.json')
    checkpoints = d['checkpoints']
    print('% Stage 2 table: runs at a best response (of 10), then median worst-case EV at 1M')
    for q in d['opponent_q']:
        cells = []
        for n in checkpoints:
            rs = [r for r in d['rows'] if r['q'] == q and r['hands'] == n]
            cells.append(f"{sum(r['at_best_response'] for r in rs)}/{len(rs)}")
        last = [r for r in d['rows'] if r['q'] == q and r['hands'] == checkpoints[-1]]
        q_tex = '$1/3$' if q == '1/3' else f'${q}$'
        print(f"{q_tex} & {frac(last[0]['best_possible_ev'])} & " + ' & '.join(cells)
              + f" & {frac(median(r['worst_case_ev'] for r in last))} \\\\")
    print()


def fixed_table():
    d = load('stage3_fixed_opponents.json')
    checkpoints = d['checkpoints']
    print('% Stage 3a table: runs at a best response by checkpoint, best EV, median worst case at 1M')
    for name in d['opponents']:
        cells = []
        for n in checkpoints:
            rs = [r for r in d['rows'] if r['opponent'] == name and r['hands'] == n]
            cells.append(f"{sum(r['at_best_response'] for r in rs)}/{len(rs)}")
        last = [r for r in d['rows'] if r['opponent'] == name and r['hands'] == checkpoints[-1]]
        print(f"{name} & {frac(last[0]['best_possible_ev'])} & " + ' & '.join(cells)
              + f" & {frac(median(r['p1_worst_case_ev'] for r in last))} \\\\")
    print()


def selfplay_plot():
    d = load('stage3_selfplay.json')
    rounds = [r for r in d['round_checkpoints'] if r > 0]
    print('% Stage 3b pgfplots: median AVERAGE-policy exploitability vs round (x log scale)')
    for method in ('naive', 'learned', 'exact'):
        pts, lo, hi = [], [], []
        for r in rounds:
            vals = [x['average_exploitability'] for x in d['rows']
                    if x['method'] == method and x['round'] == r]
            pts.append(f'({r},{median(vals):.5f})')
            lo.append(f'({r},{min(vals):.5f})')
            hi.append(f'({r},{max(vals):.5f})')
        print(f'% {method} median')
        print('coordinates {' + ' '.join(pts) + '};')
        if method != 'exact':
            print(f'% {method} min / max')
            print('coordinates {' + ' '.join(lo) + '};')
            print('coordinates {' + ' '.join(hi) + '};')
    print()
    print('% Stage 3b table: final round')
    last = rounds[-1]
    for method in ('naive', 'learned', 'exact'):
        rs = [x for x in d['rows'] if x['method'] == method and x['round'] == last]
        avg = [x['average_exploitability'] for x in rs]
        cur = [x['current_exploitability'] for x in rs]
        at100 = [x['average_exploitability'] for x in d['rows']
                 if x['method'] == method and x['round'] == 100]
        print(f"{method} & {median(at100):.4f} & {median(avg):.4f} & "
              f"{min(avg):.4f}--{max(avg):.4f} & {median(cur):.4f} \\\\")


if __name__ == '__main__':
    stage2_table()
    fixed_table()
    selfplay_plot()
