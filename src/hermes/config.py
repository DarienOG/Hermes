from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueConfig:
    prior_smoothing_k: int
    prior_alpha: float
    quarter_adjustments: bool
    min_team_games_for_weight: int
    data_tier: int


LEAGUE_CONFIG: dict[str, LeagueConfig] = {
    "acb": LeagueConfig(12, 35, True, 5, 1),
    "seriea": LeagueConfig(12, 35, True, 5, 1),
    "proa": LeagueConfig(15, 40, True, 8, 2),
    "vtb": LeagueConfig(18, 42, False, 8, 2),
    "bal": LeagueConfig(25, 50, False, 8, 3),
    "bcl_asia": LeagueConfig(25, 50, False, 8, 3),
}


def partial_pool_weight(n_obs: int, k: float) -> float:
    """Compute partial-pooling weight n / (n + k)."""
    if n_obs < 0:
        raise ValueError("n_obs must be >= 0")
    return n_obs / (n_obs + k)
