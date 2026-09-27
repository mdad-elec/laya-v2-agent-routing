"""The routing policy every g1 bar uses (PREREG-g1 Amendments 3, 5).

The cheapest candidate whose P(solve) ≥ τ, otherwise the most likely one. τ comes from the
19-point grid and is chosen by an objective computed on TUNE-side calibration rows only; the
caller supplies that objective, never a held score.
"""
from __future__ import annotations

from collections.abc import Callable

GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def cheapest_above(p: dict[str, float], cost: dict[str, float], tau: float) -> str:
    ok = [c for c in p if p[c] >= tau]
    return min(ok, key=lambda c: (cost[c], c)) if ok else max(p, key=lambda c: (p[c], c))


def choose_tau(objective: Callable[[float], float]) -> float:
    """The grid τ with the best objective; the smaller τ on a tie."""
    scores = [(objective(t), -t) for t in GRID]
    return -max(scores)[1]
