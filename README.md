# Hermes — Live Quarter Totals Engine

This repository contains a production-oriented v1 scaffold for the **International Basketball Live Quarter Totals** system described in `quarter-totals-dev-guide.docx`.

## What's implemented

- PostgreSQL schema aligned to the guide (`sql/schema.sql`)
- League configuration and prior-smoothing constants (`src/hermes/config.py`)
- Hierarchical prior generator with Negative Binomial parameterization (`src/hermes/models/priors.py`)
- Gamma-Poisson live updater (`src/hermes/models/live.py`)
- Calibration / validation metrics (ECE, MAE, centering, log-likelihood) (`src/hermes/backtest/calibration.py`)
- A minimal CLI runner for local validation (`src/hermes/main.py`)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python -m hermes.main
```

## Notes

- Scrapers and live feed integrations are intentionally not wired to external providers in this initial build.
- Core model math is implemented and unit-tested to provide an executable foundation for Week 1 development.
