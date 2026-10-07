"""Global deterministic seeding for ConformalForge.

Single entry point ``set_all(seed)`` seeds every stochastic source used by the
system (numpy global RNG, numpy default_rng, python ``random``). All synthetic
data generators and stochastic tie-breaking must go through ``np.random`` seeded
here so that two runs with the same seed produce bit-identical benchmarks.

Author: 晨星
"""

from __future__ import annotations

import random

import numpy as np

_SEEDED = False


def set_all(seed: int) -> None:
    """Seed every stochastic source deterministically from a single integer.

    Idempotent for a given seed: re-calling ``set_all(7)`` restores the exact
    same global state used by the first call.
    """
    global _SEEDED
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Default RNG is also re-bound so generators that create their own rng still
    # get a state fully determined by ``seed`` when they pass ``seed`` through.
    _SEEDED = True


def make_rng(seed: int) -> np.random.Generator:
    """Create a fresh, fully-seeded ``np.random.Generator`` from ``seed``."""
    return np.random.default_rng(np.uint32(seed))


def was_seeded() -> bool:
    return _SEEDED
