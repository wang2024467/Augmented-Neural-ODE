#!/usr/bin/env python3
"""Evaluation entrypoint scaffold.

Planned responsibilities:
- load trained checkpoint
- run inference on evaluation split
- compute standardized metrics (MSE/W2/KL/EMD/Pearson)
- write metrics.csv and summary reports
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perturbation models")
    parser.add_argument("--model", choices=["gears", "linear", "node", "anode"], required=True)
    parser.add_argument("--dataset", default="norman")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", required=False, default="")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[TODO] Evaluation pipeline is not migrated from notebooks yet.")
    print(
        f"Requested model={args.model}, dataset={args.dataset}, seed={args.seed}, "
        f"checkpoint={args.checkpoint}, output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
