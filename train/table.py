"""Recipe 2's measured table (PREREG-g1 Amendment 7): P(a cell solves | group), shrunk.

A group is a tuple of question answers (task, domain, stage, error state …). For each group the
table keeps how often each cell solved, and shrinks it toward the cell's prior:

    P(solve | group, cell) = (k + m · prior_cell) / (n + m)

For TwinRouterBench a row is labelled with the CHEAPEST tier that still lets its trajectory pass,
so tier t solves a row exactly when the label is ≤ t, and P(t solves | group) is the group's
cumulative label distribution. `TierTable` is that case; the prior is the fit rows' own cumulative
distribution, so an unseen group routes like the whole fit set.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

N_TIERS = 4


class TierTable:
    def __init__(self, m: float) -> None:
        if m < 0:
            raise ValueError("m is a pseudo-count; it cannot be negative")
        self.m = m
        self.counts: dict[Hashable, list[int]] = {}
        self.prior: list[float] = [1.0] * N_TIERS

    def fit(self, groups: Sequence[Hashable], labels: Sequence[int]) -> TierTable:
        if len(groups) != len(labels):
            raise ValueError("one group per label")
        if not labels:
            raise ValueError("a table needs at least one row")
        by_group: dict[Hashable, list[int]] = defaultdict(lambda: [0] * N_TIERS)
        total = [0] * N_TIERS
        for g, y in zip(groups, labels, strict=True):
            if not 0 <= y < N_TIERS:
                raise ValueError(f"tier {y} is outside 0..{N_TIERS - 1}")
            by_group[g][y] += 1
            total[y] += 1
        self.counts = dict(by_group)
        self.prior = _cumulative(total)
        return self

    def p_solve(self, group: Hashable) -> list[float]:
        """P(tier t lets the row pass), for t = 0..3; non-decreasing, and 1.0 at the top tier."""
        c = self.counts.get(group)
        if c is None:
            return list(self.prior)
        n = sum(c)
        k = [sum(c[: t + 1]) for t in range(N_TIERS)]
        return [(k[t] + self.m * self.prior[t]) / (n + self.m) if n + self.m > 0 else self.prior[t] for t in range(N_TIERS)]


def _cumulative(counts: Sequence[int]) -> list[float]:
    n = sum(counts)
    return [sum(counts[: t + 1]) / n for t in range(N_TIERS)]


def cheapest_passing(p: Sequence[float], tau: float) -> int:
    """The policy (train/policy.py's rule on tiers): the cheapest tier with P ≥ τ, else the top."""
    return next((t for t in range(N_TIERS) if p[t] >= tau), N_TIERS - 1)
