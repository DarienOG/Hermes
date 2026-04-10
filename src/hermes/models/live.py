from dataclasses import dataclass


@dataclass(frozen=True)
class GammaState:
    alpha: float
    beta: float


@dataclass(frozen=True)
class PosteriorSummary:
    alpha_post: float
    beta_post: float
    rate_mean: float
    prior_weight: float
    data_weight: float


def gamma_poisson_update(prior: GammaState, observed_points: int, elapsed_minutes: float) -> PosteriorSummary:
    if observed_points < 0:
        raise ValueError("observed_points must be >= 0")
    if elapsed_minutes < 0:
        raise ValueError("elapsed_minutes must be >= 0")

    alpha_post = prior.alpha + observed_points
    beta_post = prior.beta + elapsed_minutes
    rate_mean = alpha_post / beta_post
    prior_weight = prior.beta / beta_post
    data_weight = elapsed_minutes / beta_post

    return PosteriorSummary(
        alpha_post=alpha_post,
        beta_post=beta_post,
        rate_mean=rate_mean,
        prior_weight=prior_weight,
        data_weight=data_weight,
    )
