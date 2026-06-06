from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.reproduce import outdir_for, run_ts_synthetic, summarize


def main():
    parser = argparse.ArgumentParser(description="Run Section 7.2.2 and 7.2.4 Thompson Sampling experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--protocol", choices=["published", "paper-spec"], default="published")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cv-only", action="store_true", help="Only run the Section 7.2.4 cross-validation experiment.")
    parser.add_argument("--setting", type=int, default=None, help="Accepted for compatibility; maintained runner includes setting 2.")
    parser.add_argument("--T", type=int, default=None, help="Accepted for compatibility; use --mode to select the maintained grid.")
    parser.add_argument("--beta", type=int, default=None, help="Accepted for compatibility; maintained runner evaluates all paper beta values.")
    parser.add_argument("--batch_size", type=int, default=None, help="Accepted for compatibility; maintained runner evaluates paper batch sizes.")
    args = parser.parse_args()

    outdir = outdir_for(args.mode, args.protocol)
    run_ts_synthetic(args.mode, args.protocol, seed=args.seed, cv_only=args.cv_only, jobs=args.jobs, resume=args.resume)
    if not args.cv_only:
        run_ts_synthetic(args.mode, args.protocol, seed=args.seed, cv_only=True, jobs=args.jobs, resume=args.resume)
    report = summarize(outdir)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
