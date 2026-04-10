import numpy as np

from hermes.backtest.calibration import centering_rate, expected_calibration_error, mae
from hermes.config import partial_pool_weight
from hermes.models.live import GammaState, gamma_poisson_update
from hermes.models.priors import fit_negative_binomial, generate_prior


def test_partial_pool_weight():
    assert partial_pool_weight(30, 15) == 30 / 45


def test_generate_prior_reasonable():
    prior = generate_prior(21, 19, 40, n_team_games=30, k_team=15, alpha=35, home_advantage=0.5)
    assert 39 <= prior.mean <= 41
    assert prior.variance > prior.mean


def test_gamma_poisson_update_weights_sum_to_one():
    post = gamma_poisson_update(GammaState(alpha=40, beta=10), observed_points=12, elapsed_minutes=3)
    assert np.isclose(post.prior_weight + post.data_weight, 1.0)


def test_calibration_metrics():
    probs = np.array([0.55, 0.60, 0.70, 0.80])
    outcomes = np.array([1, 1, 0, 1])
    assert 0 <= expected_calibration_error(probs, outcomes) <= 1
    assert mae(np.array([40, 42]), np.array([41, 39])) == 2.0
    assert 0 <= centering_rate(np.array([40, 41]), np.array([42, 40])) <= 1


def test_fit_negative_binomial_returns_positive_params():
    sample = np.array([34, 36, 40, 42, 39, 45, 37, 41, 38, 43])
    fit = fit_negative_binomial(sample)
    assert fit.mean > 0
    assert fit.alpha > 0
