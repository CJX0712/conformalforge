import numpy as np

from conformalforge.core.seed import make_rng, set_all


def test_set_all_reproducible():
    set_all(7)
    a = np.random.standard_normal(5)
    set_all(7)
    b = np.random.standard_normal(5)
    assert np.array_equal(a, b)


def test_make_rng_independent_streams():
    r1 = make_rng(3)
    r2 = make_rng(3)
    assert np.array_equal(r1.standard_normal(4), r2.standard_normal(4))
    r3 = make_rng(4)
    assert not np.array_equal(r1.standard_normal(4), r3.standard_normal(4))
