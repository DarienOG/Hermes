# Week 1 Build Artifacts

This repository now includes the core Week 1 deliverable scaffolding from the guide:

1. **PostgreSQL schema** in `sql/schema.sql`.
2. **League configuration dictionary** in `src/hermes/config.py`.
3. **Negative Binomial fit + prior generation module** in `src/hermes/models/priors.py`.
4. **Gamma-Poisson updater** in `src/hermes/models/live.py`.
5. **Backtest validation metrics** in `src/hermes/backtest/calibration.py`.
6. **Coverage matrix template** in `coverage-matrix.csv`.

External tasks that require live credentials (BetsAPI, books, Playwright scraping targets) are prepared as templates and runtime modules but not executed in this offline build.
