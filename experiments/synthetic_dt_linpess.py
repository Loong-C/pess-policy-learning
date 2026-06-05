from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.reproduce import outdir_for, run_contextual_nonadaptive, summarize


def main():
    parser = argparse.ArgumentParser(description="Run Section 7.1.2 contextual non-adaptive tree/linear experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--setting", type=int, default=None, help="Accepted for compatibility; maintained runner executes the paper grid.")
    parser.add_argument("--T", type=int, default=None, help="Accepted for compatibility; use --mode to select the maintained grid.")
    parser.add_argument("--beta", type=int, default=None, help="Accepted for compatibility; maintained runner evaluates all paper beta values.")
    parser.add_argument("--scenario", type=int, default=None, help="Accepted for compatibility; maintained runner evaluates all three scenarios.")
    args = parser.parse_args()

    outdir = outdir_for(args.mode)
    run_contextual_nonadaptive(args.mode, seed=args.seed)
    report = summarize(outdir)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
