from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

from hermes.config import partial_pool_weight


@dataclass(frozen=True)
class NBPrior:
    mean: float
    alpha: float

    @property
    def variance(self) -> float:
        return self.mean + (self.mean**2) / self.alpha


def _neg_ll(params: np.ndarray, data: np.ndarray) -> float:
    mean, alpha = params
    if mean <= 0 or alpha <= 0:
        return np.inf
    p = alpha / (alpha + mean)
    return float(-np.sum(nbinom.logpmf(data, alpha, p)))


def fit_negative_binomial(data: list[int] | np.ndarray) -> NBPrior:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must not be empty")
    initial = np.array([max(arr.mean(), 1.0), 40.0])
    result = minimize(_neg_ll, x0=initial, args=(arr,), bounds=((1e-6, None), (1e-6, None)))
    if not result.success:
        raise RuntimeError(f"NB fit failed: {result.message}")
    mean, alpha = result.x
    return NBPrior(mean=float(mean), alpha=float(alpha))


def generate_prior(
    team_a_off_q: float,
    team_b_off_q: float,
    league_mean: float,
    n_team_games: int,
    k_team: float,
    alpha: float,
    home_advantage: float = 0.0,
    quarter_adjustment: float = 0.0,
) -> NBPrior:
    """Generate team-matchup prior with partial pooling."""
    team_est = team_a_off_q + team_b_off_q
    w_team = partial_pool_weight(n_team_games, k_team)
    mean = (w_team * team_est) + ((1 - w_team) * league_mean)
    mean += home_advantage + quarter_adjustment
    return NBPrior(mean=mean, alpha=alpha)
