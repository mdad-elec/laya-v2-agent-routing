"""Score first-repeat routing quality and all-repeat timing without dropping failures.

A failed request stays in every quality denominator: it is never correct, and it is counted
as neither overspend nor underspend (it is reported under `errors`).
"""
import collections
import math
import statistics

from prompts import COST, TIERS


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (xs[hi] - xs[lo]) * (p - lo)


def ece(rows, bins=10):
    """Expected calibration error of p_chosen, equal-width bins; None when no probabilities."""
    scored = [r for r in rows if r.get('status') == 'ok' and r.get('p_chosen') is not None]
    if not scored:
        return None
    buckets = [[] for _ in range(bins)]
    for r in scored:
        p = r['p_chosen']
        buckets[min(bins - 1, int(p * bins))].append((p, r['predicted'] == r['expected']))
    return sum(len(b) / len(scored) * abs(statistics.mean(x for x, _ in b) - statistics.mean(y for _, y in b))
               for b in buckets if b)


def quality(rows):
    n = len(rows)
    ok = [r for r in rows if r.get('status') == 'ok']
    correct = sum(r.get('predicted') == r['expected'] for r in rows)
    matrix = {a: {b: 0 for b in TIERS + ['ERROR']} for a in TIERS}
    for r in rows:
        matrix[r['expected']][r.get('predicted') if r.get('status') == 'ok' else 'ERROR'] += 1
    f1 = []
    for t in [t for t in TIERS if any(r['expected'] == t for r in rows)]:  # labels with support
        tp = matrix[t][t]
        fp = sum(matrix[x][t] for x in TIERS if x != t)
        fn = sum(v for x, v in matrix[t].items() if x != t)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    over = sum(r.get('status') == 'ok' and COST[r['predicted']] > COST[r['expected']] for r in rows)
    under = sum(r.get('status') == 'ok' and COST[r['predicted']] < COST[r['expected']] for r in rows)
    return {'n': n, 'correct': correct, 'accuracy': correct / n if n else None, 'errors': n - len(ok),
            'macro_f1': statistics.mean(f1) if n else None,
            'overspend': over, 'overspend_rate': over / n if n else None,
            'underspend': under, 'underspend_rate': under / n if n else None,
            'ece_p_chosen_10_bins': ece(rows), 'confusion_matrix': matrix}


def summarize(rows, expected_repeats=3):
    first = [r for r in rows if r['repeat'] == 0]
    result = quality(first)
    result['by_band'] = {b: quality([r for r in first if r['band'] == b])
                         for b in ['trivial', 'easy', 'medium', 'hard'] if any(r['band'] == b for r in first)}
    result['by_split'] = {s: quality([r for r in first if r['split'] == s])
                          for s in ['tune', 'held'] if any(r['split'] == s for r in first)}
    success = [r for r in rows if r['status'] == 'ok']
    times = [r['latency_ms'] for r in success if r.get('latency_ms') is not None]
    result['timing'] = {'n': len(times), 'p50_ms': percentile(times, .5), 'p95_ms': percentile(times, .95),
                        'mean_ms': statistics.mean(times) if times else None,
                        'definition': 'Client wall clock around one system_one call, excluding the warmup; '
                                      'p95 uses linear interpolation. Null for replayed decisions.'}
    result['total_requests'] = len(rows)
    result['failed_requests'] = len(rows) - len(success)
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['id']].append(r)
    complete = [v for v in grouped.values() if len(v) == expected_repeats and all(r['status'] == 'ok' for r in v)]
    result['repeat_consistency'] = {'repeats': expected_repeats, 'complete_cases': len(complete),
                                    'same_tier_all_repeats': sum(len({r['predicted'] for r in v}) == 1
                                                                 for v in complete)}
    result['quality_repeat'] = 0
    return result
