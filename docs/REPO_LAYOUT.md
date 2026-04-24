# Repository Layout (Scaffold)

This project now uses a standard, GitHub-friendly scaffold so notebook code can be migrated incrementally.

## Current structure

- `src/augmented_neural_ode/` — reusable package code (currently minimal placeholder)
- `scripts/` — CLI entrypoints (`train.py`, `evaluate.py`, `plot.py`)
- `tests/` — lightweight tests for scaffold and future logic
- `data/` — local/processed data artifacts (git-ignored or placeholders)
- `results/` — experiment outputs and metrics
- `configs/` — run configs and hyperparameter presets
- `docs/` — project documentation (layout, roadmap, methodology)
- `legacy_original/` — all original root files preserved before refactoring

## Migration recommendation

1. Move shared notebook classes (dataset/model wrappers) into `src/augmented_neural_ode/`.
2. Call those modules from `scripts/train.py`, `scripts/evaluate.py`, and `scripts/plot.py`.
3. Save outputs under:
   - `results/<model>/<dataset>/<seed>/metrics.csv`
   - `results/<model>/<dataset>/<seed>/config.json`
4. Keep exploratory notebooks, but treat scripts as the reproducible source of truth.


## Preservation policy

During migration, original files are kept in `legacy_original/` rather than deleted, so the team can always roll back or cross-check notebook outputs.
