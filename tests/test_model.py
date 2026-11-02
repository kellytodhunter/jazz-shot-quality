import numpy as np
import pandas as pd

from conftest import make_shots
from shotquality import features, model


def test_zone_baseline_uses_cell_mean_and_falls_back():
    df = make_shots(400).assign(SHOT_ZONE_RANGE="8-16 ft.")
    df["SHOT_MADE_FLAG"] = (df.index % 4 == 0).astype(int)
    zb = model.ZoneBaseline(min_cell=50).fit(df)
    assert np.allclose(zb.predict_proba1(df), 0.25, atol=0.01)
    unseen = df.head(3).assign(SHOT_ZONE_AREA="Never Seen", SHOT_ZONE_RANGE="Nope")
    assert np.allclose(zb.predict_proba1(unseen), 0.25, atol=0.01)  # falls back to basic-zone rate


def test_ece_is_zero_for_perfect_calibration_and_large_for_miscalibration():
    p = np.repeat([0.2, 0.8], 5000)
    y = np.concatenate([np.arange(5000) % 5 == 0, np.arange(5000) % 5 != 0]).astype(int)
    assert model.expected_calibration_error(y, p, bins=2) < 1e-9
    assert model.expected_calibration_error(y, 1 - p, bins=2) > 0.5


def test_calibration_table_partitions_all_shots():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.7, 1000)
    tbl = model.calibration_table(rng.binomial(1, p), p, bins=10)
    assert tbl.n.sum() == 1000 and tbl.predicted.is_monotonic_increasing


def test_bootstrap_ci_brackets_the_mean_and_is_tight_for_constant_gap():
    rng = np.random.default_rng(2)
    games = rng.integers(0, 50, 2000)
    const = model.game_cluster_bootstrap(games, np.full(2000, 0.7), np.full(2000, 0.6))
    assert abs(const["mean"] - 0.1) < 1e-12 and const["hi"] - const["lo"] < 1e-9
    noisy = model.game_cluster_bootstrap(games, rng.normal(0.7, 0.3, 2000), rng.normal(0.6, 0.3, 2000))
    assert noisy["lo"] < noisy["mean"] < noisy["hi"]


def test_pointwise_log_loss_matches_closed_form():
    ll = model.pointwise_log_loss([1, 0], np.array([0.8, 0.3]))
    assert np.allclose(ll, [-np.log(0.8), -np.log(0.7)])


def test_gbm_learns_distance_signal_and_outputs_valid_probabilities():
    rng = np.random.default_rng(3)
    df = make_shots(6000, seed=3)
    f = features.add_features(df)
    f["SHOT_MADE_FLAG"] = rng.binomial(1, np.clip(0.75 - 0.02 * f["dist_ft"], 0.05, 0.95))
    m = model.ShotModel("gbm_geometry", {"max_iter": 60}).fit(f.iloc[:4000])
    p = m.predict(f.iloc[4000:])
    assert ((p > 0) & (p < 1)).all()
    assert np.corrcoef(p, -f.iloc[4000:]["dist_ft"])[0, 1] > 0.5
