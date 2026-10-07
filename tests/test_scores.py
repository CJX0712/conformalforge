import numpy as np

from conformalforge.conformal.scores import (
    aps_pi,
    conformal_quantile,
    lac_scores,
    quantile_level,
    raps_pi,
)


def _probs(n=20, k=4, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, k))
    return raw / raw.sum(axis=1, keepdims=True)


def test_quantile_level_plus_correction():
    lvl = quantile_level(0.1, 400, plus=True)
    assert lvl >= 0.9
    # Plus correction pushes level above 1-alpha.
    assert lvl > 0.9


def test_lac_scores_shape_and_range():
    p = _probs()
    y = np.random.default_rng(1).integers(0, 4, size=p.shape[0])
    s = lac_scores(p, y)
    assert s.shape == (p.shape[0],)
    assert np.all((s >= 0) & (s <= 1.0 + 1e-9))


def test_aps_cumulative_monotone():
    # Higher-probability classes have smaller cumulative pi.
    p = np.array([[0.7, 0.2, 0.1]])
    pi = aps_pi(p, np.random.default_rng(0))
    assert pi.shape == (1, 3)
    assert pi[0, 0] < pi[0, 1] < pi[0, 2]


def test_raps_penalty_increases_later_classes():
    p = np.array([[0.4, 0.3, 0.2, 0.1]])
    rng = np.random.default_rng(0)
    pi_aps = aps_pi(p, rng)
    pi_raps = raps_pi(p, rng, lam=0.5, k_reg=1)
    # RAPS penalises later (lower-prob) classes -> larger pi there.
    assert pi_raps[0, 3] > pi_aps[0, 3]


def test_conformal_quantile_deterministic():
    rng = np.random.default_rng(5)
    s = rng.random(300)
    q1 = conformal_quantile(s, 0.1)
    q2 = conformal_quantile(s.copy(), 0.1)
    assert q1 == q2
