#!/usr/bin/env python3
"""Plotting entrypoint scaffold.

Planned responsibilities:
- read standardized result files from results/
- generate comparison plots across models and metrics
- save artifacts under results/figures/
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model comparison figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset", default="norman")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[TODO] Plotting pipeline is not migrated from notebooks yet.")
    print(f"Requested results_dir={args.results_dir}, dataset={args.dataset}, seed={args.seed}")


if __name__ == "__main__":
    main()
