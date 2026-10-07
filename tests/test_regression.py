from conformalforge.conformal.regression import RegFuse, SplitConformal
from conformalforge.core.seed import set_all
from conformalforge.data.synthetic import HeteroscedasticRegGenerator


def _data(seed=7, n=800, calib=400, test=400, hetero=1.5):
    set_all(seed)
    g = HeteroscedasticRegGenerator(n_train=n, n_calib=calib, n_test=test, hetero=hetero)
    return g.generate(seed)


def test_split_conformal_valid():
    train, calib, test = _data()
    r = SplitConformal().fit_predict(train, calib, test, 0.1, seed=1)
    assert 0.87 <= r.coverage <= 1.0


def test_reg_fuse_valid_and_narrower():
    train, calib, test = _data(hetero=1.5)
    base = SplitConformal().fit_predict(train, calib, test, 0.1, seed=2)
    fuse = RegFuse(k=25).fit_predict(train, calib, test, 0.1, seed=2)
    assert 0.87 <= fuse.coverage <= 1.0
    # Under heteroscedastic noise, local-scale fuse should be <= baseline width.
    assert fuse.mean_interval_width <= base.mean_interval_width + 1e-9


def test_reg_fuse_homoscedastic_not_worse():
    train, calib, test = _data(hetero=0.0)
    fuse = RegFuse(k=25).fit_predict(train, calib, test, 0.1, seed=3)
    # Still valid; width may be slightly larger but must stay finite/valid.
    assert 0.85 <= fuse.coverage <= 1.0
    assert fuse.mean_interval_width > 0
