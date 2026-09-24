"""The statistics `scripts/laya/PREREG.md` names, in the standard library, so every number in a
Laya-v2 result can be recomputed from the file that holds it with nothing installed.

    mcnemar_exact(a_ok, b_ok)        paired correctness of two deciders on the same asks
    bootstrap_ci(values)             a seeded percentile interval on a mean
    paired_bootstrap_ci(a, b)        the same on the per-ask difference b - a
    selection_optimism(k, n)         how far the best of k candidates flatters itself on n rows

`score.py` keeps its normal-approximation McNemar for the reports it already wrote; the
pre-registered test is the exact binomial one here.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from propose_cells import wilson_lcb  # noqa: E402,F401  re-exported: one Wilson bound in the estate

SEED = 20260923


def _binom_cdf(k, n):
    """P(X <= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n


def mcnemar_exact(a_ok, b_ok):
    """Exact McNemar on paired booleans. `p_b_better` is one-sided: B right where A is wrong more
    often than the reverse. No discordant pair is no evidence either way (p = 1)."""
    if len(a_ok) != len(b_ok):
        raise ValueError(f"McNemar needs paired decisions: {len(a_ok)} against {len(b_ok)}")
    a_only = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    b_only = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p_two_sided": 1.0, "p_b_better": 1.0}
    two = min(1.0, 2 * _binom_cdf(min(a_only, b_only), n))
    one = _binom_cdf(a_only, n)          # P(A-only this small or smaller) under H0
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n,
            "p_two_sided": round(two, 6), "p_b_better": round(one, 6)}


def _mean(xs):
    return sum(xs) / len(xs)


def _percentile(sorted_xs, q):
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def bootstrap_ci(values, stat=_mean, n=2000, seed=SEED, level=0.90):
    values = list(values)
    if not values:
        raise ValueError("a bootstrap interval needs at least one value")
    rng = random.Random(seed)
    k = len(values)
    draws = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    tail = (1 - level) / 2
    return round(_percentile(draws, tail), 6), round(_percentile(draws, 1 - tail), 6)


def paired_bootstrap_ci(a, b, n=2000, seed=SEED, level=0.90):
    if len(a) != len(b):
        raise ValueError(f"a paired interval needs paired values: {len(a)} against {len(b)}")
    return bootstrap_ci([y - x for x, y in zip(a, b)], n=n, seed=seed, level=level)


def _expected_max_normal(k):
    """E[max of k iid standard normals], by numerical integration of x * k * phi(x) * Phi(x)^(k-1)."""
    if k <= 1:
        return 0.0
    step, total, x = 0.001, 0.0, -8.0
    while x < 8.0:
        phi = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        big = 0.5 * (1 + math.erf(x / math.sqrt(2)))
        total += x * k * phi * big ** (k - 1) * step
        x += step
    return total


def selection_optimism(k, n, p=0.7):
    """How much the best of k equally good candidates overstates itself on n rows: the standard
    error of an accuracy near p times the expected maximum of k standard normals. It is the honest
    answer to "how many candidates can n rows tell apart" — a search whose winner beats the field by
    less than this has found noise."""
    if k <= 1:
        return 0.0
    return round(math.sqrt(p * (1 - p) / n) * _expected_max_normal(k), 4)
