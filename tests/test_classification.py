from conformalforge.conformal.classification import (
    APSConformal,
    ClassConditional,
    ConformalFuse,
    LACConformal,
    NaiveThreshold,
    RAPSConformal,
)
from conformalforge.core.seed import set_all
from conformalforge.data.synthetic import GaussianBlobGenerator


def _data(seed=7, n=1200, calib=600, test=600, noise=0.9, k=5):
    set_all(seed)
    g = GaussianBlobGenerator(n_train=n, n_calib=calib, n_test=test, n_classes=k, noise=noise)
    return g.generate(seed)


def test_naive_has_no_coverage_guarantee():
    train, calib, test = _data(noise=1.3)
    r = NaiveThreshold().fit_predict(train, calib, test, 0.1, seed=1)
    # Argmax singleton: set size is always 1 and coverage = accuracy, which on this
    # imperfect DGP falls well below the 1-alpha target -> no validity guarantee.
    assert r.avg_set_size == 1.0
    assert r.coverage < 0.95


def test_lac_coverage_valid():
    train, calib, test = _data()
    r = LACConformal().fit_predict(train, calib, test, 0.1, seed=2)
    assert 0.87 <= r.coverage <= 1.0
    assert r.avg_set_size >= 1.0


def test_aps_and_raps_valid():
    train, calib, test = _data()
    aps = APSConformal().fit_predict(train, calib, test, 0.1, seed=3)
    raps = RAPSConformal(lam=0.1, k_reg=5).fit_predict(train, calib, test, 0.1, seed=3)
    assert 0.85 <= aps.coverage <= 1.0
    assert 0.85 <= raps.coverage <= 1.0
    # NOTE (honest): for softmax probabilities APS/RAPS set sizes are NOT uniformly
    # smaller than LAC; LAC (1-p_c) is already near-optimal. ConformalForge's
    # efficiency win is on the regression side (RegFuse) and its fairness win is
    # class-conditional (SCP). We only assert validity here.
    assert aps.avg_set_size >= 1.0
    assert raps.avg_set_size >= 1.0


def test_conformal_fuse_runs_and_valid():
    train, calib, test = _data()
    r = ConformalFuse().fit_predict(train, calib, test, 0.1, seed=4)
    assert 0.85 <= r.coverage <= 1.0
    assert r.avg_set_size is not None
    assert r.under_gap is not None


def test_class_conditional_valid():
    train, calib, test = _data()
    r = ClassConditional().fit_predict(train, calib, test, 0.1, seed=5)
    assert 0.85 <= r.coverage <= 1.0


def test_fuse_reduces_under_coverage_gap():
    # Flagship (SCP) should cut the worst-class under-coverage gap vs marginal LAC.
    train, calib, test = _data()
    lac = LACConformal().fit_predict(train, calib, test, 0.1, seed=9)
    fuse = ConformalFuse(score="lac", per_class=True).fit_predict(train, calib, test, 0.1, seed=9)
    assert fuse.under_gap <= lac.under_gap + 1e-9
