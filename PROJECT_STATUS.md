# Project Status and Recommended Next Steps (2026-04-24)

## Overview

This document summarizes the current repository state and proposes a practical roadmap to move from notebook-centric experimentation to reproducible, paper-aligned experiments.

---

## 1) What is already implemented

Based on the current notebooks and model artifacts, the repository already includes:

- Data processing and dataset construction (`legacy_original/dataset_1.ipynb`, `legacy_original/preprocessing.py`)
- GEARS baseline training/inference workflows (`legacy_original/test_gears.ipynb`, `legacy_original/Untitled-1.ipynb`)
- Linear baseline implementation (`LinearPertModel`)
- Neural ODE implementation (`PerturbedNeuralODE`)
- Augmented Neural ODE implementation (`PerturbedAugmentedNODE`)
- Multi-metric evaluation and visualization (MSE/W2/KL/EMD/Pearson; violin plots and Top-N curves)
- Exported model artifacts:
  - `legacy_original/linear_model.pth`
  - `legacy_original/node_model.pth`
  - `legacy_original/anode_model.pt`
  - `legacy_original/perturbed_node_model.pt`

---

## 2) Highest-priority issues

### A. No unified reproducible entrypoint

- Core logic is spread across multiple notebooks.
- Training, evaluation, and plotting are not yet organized into one scriptable pipeline.

### B. Dependencies are not pinned

- No `requirements.txt` or `environment.yml` is currently provided.
- This blocks reliable reproducibility and CI setup.

### C. Results are not fully traceable

- Model files exist, but experiment metadata (seed/split/hyperparameters) is not consistently coupled with outputs.
- Model comparison outputs are not standardized into a uniform table format.

### D. Paper alignment is incomplete

- ODE/ANODE structures are implemented, but there is no explicit checklist proving parity with the target paper settings.

---

## 3) Recommended execution plan

### Step A (Day 1): Reproducible pipeline baseline

- Create script entrypoints:
  - `train.py`: train GEARS / Linear / NODE / ANODE
  - `eval.py`: compute MSE/W2/KL/EMD/Pearson in one place
  - `plot.py`: generate standardized comparison plots
- Add environment lock file:
  - `requirements.txt` **or** `environment.yml`
- Update README with a minimal end-to-end run path:
  - Data preparation → Training → Evaluation

### Step B (Day 2-3): Standardized experiment tracking

- Standardize output structure:
  - `results/<model>/<dataset>/<seed>/metrics.csv`
  - `results/<model>/<dataset>/<seed>/config.json`
- Fix random seeds and ensure all models use the exact same data split.
- Report mean ± std over multiple runs.

### Step C (Day 3-4): Paper parity checklist

Create a formal parity checklist and verify:

- Dataset and split policy
- Model depth/hidden size/solver/time grid
- Epochs, batch size, learning rate, early stopping
- Metric definitions (per-gene vs per-sample)

---

## 4) Immediate action items (start here)

1. Extract core training/evaluation blocks from `legacy_original/Untitled-1.ipynb` into scripts.
2. Save model comparison outputs into standardized CSV summaries.
3. Expand README so a new contributor can reproduce baseline results quickly in a fresh environment.

---

## 5) Paper-based alignment note

The repository includes `legacy_original/final_project_4243_ (2).pdf`, but this execution environment currently lacks working PDF text extraction tooling.

To proceed quickly with strict paper alignment:

- Share the pages containing experiment setup / hyperparameter tables / metric definitions.
- Then convert that directly into a line-by-line code alignment checklist with actionable patches.


---

## 6) Legacy file preservation

To prevent accidental loss during repository restructuring, original notebooks, model checkpoints, embeddings, and the project PDF are preserved under `legacy_original/`.
