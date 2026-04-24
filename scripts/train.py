#!/usr/bin/env python3
"""Training entrypoint scaffold.

Planned responsibilities:
- load dataset
- initialize selected model (gears/linear/node/anode)
- run training loop
- save checkpoints and config metadata
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train perturbation models")
    parser.add_argument("--model", choices=["gears", "linear", "node", "anode"], required=True)
    parser.add_argument("--dataset", default="norman")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[TODO] Training pipeline is not migrated from notebooks yet.")
    print(f"Requested model={args.model}, dataset={args.dataset}, seed={args.seed}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
