import numpy as np
import pytest

from shotquality import shrinkage


def simulate(k=300, tau=0.05, seed=0):
    rng = np.random.default_rng(seed)
    se2 = (0.9 / np.sqrt(rng.integers(100, 900, k))) ** 2
    truth = rng.normal(-0.01, tau, k)
    return truth, rng.normal(truth, np.sqrt(se2)), se2


def test_prior_recovers_true_talent_spread():
    est = [shrinkage.fit_prior(*simulate(seed=s)[1:]).tau for s in range(20)]
    assert abs(np.mean(est) - 0.05) < 0.005


def test_prior_is_zero_when_variation_is_all_noise():
    _, obs, se2 = simulate(tau=0.0)
    assert shrinkage.fit_prior(obs, se2).tau < 0.01


def test_weights_in_unit_interval_and_grow_with_sample_size():
    prior = shrinkage.Prior(mu=0.0, tau2=0.0025)
    out = shrinkage.shrink([0.1, 0.1], [0.01, 0.0001], prior)
    assert ((out.weight >= 0) & (out.weight <= 1)).all()
    assert out.weight[1] > out.weight[0] and out.shrunk[0] < out.shrunk[1] < 0.1


def test_shrinkage_beats_raw_estimates_in_mean_squared_error():
    truth, obs, se2 = simulate(seed=4)
    shrunk = shrinkage.shrink(obs, se2, shrinkage.fit_prior(obs, se2)).shrunk
    assert np.mean((shrunk - truth) ** 2) < np.mean((obs - truth) ** 2)


def test_credible_intervals_have_close_to_nominal_coverage():
    hits = []
    for s in range(30):
        truth, obs, se2 = simulate(seed=100 + s)
        out = shrinkage.shrink(obs, se2, shrinkage.fit_prior(obs, se2), level=0.9)
        hits.append(((truth >= out.lo) & (truth <= out.hi)).mean())
    assert 0.86 < np.mean(hits) < 0.94


def test_zero_tau_collapses_everyone_to_league_mean():
    out = shrinkage.shrink([0.3, -0.2], [0.01, 0.01], shrinkage.Prior(mu=0.02, tau2=0.0))
    assert np.allclose(out.shrunk, 0.02)


def test_prior_needs_enough_players():
    with pytest.raises(ValueError):
        shrinkage.fit_prior([0.1, 0.2], [0.01, 0.01])
