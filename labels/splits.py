"""Split rules beyond items (PREREG-g1 Amendments 1-2), all derived from `suites.base.split_of`.

- Held-out MODELS: `model_split(name)`. They are cut from the zero-shot PROBE checkpoint only;
  the scored checkpoint trains on the tune items of every model (Amendment 2).
- Trajectories: a multi-step source keys every step of one trajectory by the trajectory, so a
  trajectory never straddles the split (labels.twinrouterbench.trajectory_key).
"""
from __future__ import annotations

from suites.base import split_of


def model_split(name: str) -> str:
    return split_of(f"model/{name}")


def probe_trainable(name: str) -> bool:
    return model_split(name) == "tune"
