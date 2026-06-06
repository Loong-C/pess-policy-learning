from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.reproduce import (
    PAPER_SPEC_REAL_DATASETS,
    PUBLISHED_REAL_DATASETS,
    outdir_for,
    run_real,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description="Run Section 7.3 OpenML real-data experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--protocol", choices=["published", "paper-spec"], default="published")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--data",
        type=int,
        default=None,
        help=(
            "Accepted for compatibility; full mode uses "
            f"{len(PUBLISHED_REAL_DATASETS)} published-code datasets or "
            f"{len(PAPER_SPEC_REAL_DATASETS)} appendix datasets."
        ),
    )
    parser.add_argument("--setting", type=int, default=None, help="Accepted for compatibility; maintained runner evaluates all paper exploration settings.")
    parser.add_argument("--beta_id", type=int, default=None, help="Accepted for compatibility; beta is selected by Algorithm 2 cross-validation.")
    parser.add_argument("--batch_size", type=int, default=None, help="Accepted for compatibility; maintained runner evaluates batch sizes 10 and 100.")
    parser.add_argument("--depth", type=int, default=5, help="Accepted for compatibility; maintained runner uses depth 5 as in the paper.")
    args = parser.parse_args()

    outdir = outdir_for(args.mode, args.protocol)
    run_real(args.mode, args.protocol, seed=args.seed, jobs=args.jobs, resume=args.resume)
    report = summarize(outdir)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
