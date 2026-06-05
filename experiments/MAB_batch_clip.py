from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.reproduce import outdir_for, run_mab, summarize


def main():
    parser = argparse.ArgumentParser(description="Run Section 7.1.1 MAB experiments including the clipped baseline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Accepted for CLI consistency; MAB is already vectorized and runs single-process.")
    parser.add_argument("--resume", action="store_true", help="Accepted for CLI consistency; MAB rewrites its aggregate output.")
    parser.add_argument("--setting", type=int, choices=[1, 2, 3], default=None, help="Accepted for old commands; the maintained runner executes all three paper settings.")
    args = parser.parse_args()

    outdir = outdir_for(args.mode)
    run_mab(args.mode, seed=args.seed)
    report = summarize(outdir)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
