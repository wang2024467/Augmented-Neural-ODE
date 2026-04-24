# Augmented-Neural-ODE

This repository explores perturbation-response modeling with **GEARS**, **Linear baselines**, **Neural ODE**, and **Augmented Neural ODE (ANODE)** workflows.

## Project Status

For the current progress summary and roadmap, see:

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md)

## Repository Layout

A standard scaffold has been added to support migration from notebooks to reproducible scripts:

```text
.
├── src/augmented_neural_ode/   # reusable package code
├── scripts/                    # train/evaluate/plot entrypoints
├── tests/                      # automated checks
├── data/                       # data placeholders / local artifacts
├── results/                    # model outputs and metrics
├── configs/                    # run configs
├── docs/                       # extra project docs
├── legacy_original/            # original notebooks/models/files (moved to avoid loss)
├── PROJECT_STATUS.md
└── README.md
```

Detailed layout notes:

- [`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md)

## Quick start (scaffold)

```bash
python scripts/train.py --model node --dataset norman --seed 1
python scripts/evaluate.py --model node --dataset norman --seed 1
python scripts/plot.py --results-dir results --dataset norman --seed 1
```

> Note: the script entrypoints are scaffolds for now and will be wired to migrated notebook logic next.


## Legacy Files

All original repository files were moved into [`legacy_original/`](legacy_original) to avoid accidental loss during refactoring.
