"""Print the results table as markdown, generated from results/v1/summary.json (stdlib only)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fmt(value, kind='rate'):
    if value is None:
        return 'n/a'
    if kind == 'ms':
        return f'{value:.0f} ms'
    return f'{value:.3f}'


def table(summary):
    lines = ['| run | slice | correct | accuracy | macro-F1 | overspend | underspend | ECE (p_chosen) | p50 | p95 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in summary.items():
        slices = [('all', r)] + [(f'split={k}', v) for k, v in r['by_split'].items()] \
            + [(f'band={k}', v) for k, v in r['by_band'].items()]
        for label, q in slices:
            timing = r['timing'] if label == 'all' else {'p50_ms': None, 'p95_ms': None}
            lines.append(f"| {name} | {label} | {q['correct']}/{q['n']} | {fmt(q['accuracy'])} | "
                         f"{fmt(q['macro_f1'])} | {q['overspend']} | {q['underspend']} | "
                         f"{fmt(q['ece_p_chosen_10_bins'])} | {fmt(timing['p50_ms'], 'ms')} | "
                         f"{fmt(timing['p95_ms'], 'ms')} |")
    lines.append('')
    lines.append('Quality uses the preselected first repeat; failed requests stay in every denominator. '
                 'Latency covers all timed repeats, excluding warmup.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/v1/summary.json')
    args = parser.parse_args()
    if not args.summary.is_file():
        sys.exit(f'No summary at {args.summary}; run audit.py --write-summary after archiving a run.')
    print(table(json.loads(args.summary.read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
