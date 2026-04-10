from hermes.backtest.calibration import centering_rate, expected_calibration_error, mae
from hermes.models.live import GammaState, gamma_poisson_update
from hermes.models.priors import generate_prior


def main() -> None:
    prior = generate_prior(
        team_a_off_q=20.5,
        team_b_off_q=19.8,
        league_mean=39.5,
        n_team_games=24,
        k_team=15,
        alpha=35,
        home_advantage=0.6,
        quarter_adjustment=-0.3,
    )
    update = gamma_poisson_update(GammaState(alpha=40, beta=10), observed_points=12, elapsed_minutes=3)
    print(f"Prior mean={prior.mean:.2f}, var={prior.variance:.2f}")
    print(f"Posterior rate={update.rate_mean:.3f}, prior_weight={update.prior_weight:.3f}")

    probs = [0.52, 0.61, 0.73, 0.57, 0.66]
    outcomes = [1, 1, 0, 1, 1]
    preds = [40.0, 41.2, 39.5, 42.0, 40.8]
    actuals = [41, 43, 35, 44, 42]

    print(f"ECE={expected_calibration_error(probs, outcomes):.4f}")
    print(f"MAE={mae(preds, actuals):.4f}")
    print(f"Centering over-rate={centering_rate(preds, actuals):.4f}")


if __name__ == "__main__":
    main()
