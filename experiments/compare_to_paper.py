from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _summarize_ours(path: Path, metric: str, keys: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in keys + [metric] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = df.groupby(keys)[metric].agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": "ours_mean", "std": "ours_sd", "count": "ours_count"})
    out["ours_se"] = out["ours_sd"] / np.sqrt(out["ours_count"])
    return out


def _paper_se(df: pd.DataFrame) -> pd.Series:
    if "paper_se" in df.columns:
        return df["paper_se"]
    if {"paper_sd", "paper_count"}.issubset(df.columns):
        return df["paper_sd"] / np.sqrt(df["paper_count"])
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def compare_to_paper(ours_path: Path, paper_path: Path, metric: str, keys: list[str], margin: float, alpha: float) -> pd.DataFrame:
    ours = _summarize_ours(ours_path, metric, keys)
    paper = pd.read_csv(paper_path)
    required = keys + ["paper_mean"]
    missing = [col for col in required if col not in paper.columns]
    if missing:
        raise ValueError(f"{paper_path} is missing required columns: {missing}")

    merged = ours.merge(paper, on=keys, how="inner")
    if merged.empty:
        raise ValueError("No rows matched between our results and the paper reference keys.")

    paper_se = _paper_se(merged)
    merged["diff"] = merged["ours_mean"] - merged["paper_mean"]
    merged["combined_se"] = np.sqrt(merged["ours_se"] ** 2 + paper_se**2)
    merged["z"] = merged["diff"] / merged["combined_se"].replace(0, np.nan)
    merged["ci95_low"] = merged["diff"] - 1.96 * merged["combined_se"]
    merged["ci95_high"] = merged["diff"] + 1.96 * merged["combined_se"]

    # Two one-sided tests: equivalent iff -margin < ours-paper < margin at level alpha.
    merged["p_gt_minus_margin"] = 1.0 - stats.norm.cdf((merged["diff"] + margin) / merged["combined_se"])
    merged["p_lt_plus_margin"] = stats.norm.cdf((merged["diff"] - margin) / merged["combined_se"])
    merged["tost_p"] = merged[["p_gt_minus_margin", "p_lt_plus_margin"]].max(axis=1)
    merged["equivalent"] = merged["tost_p"] < alpha
    merged["consistent_by_ci"] = (merged["ci95_low"] <= margin) & (merged["ci95_high"] >= -margin)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Compare reproduction results with numeric paper references.")
    parser.add_argument("--ours", required=True, type=Path, help="CSV produced by experiments/reproduce.py.")
    parser.add_argument("--paper", required=True, type=Path, help="CSV with paper_mean and the same key columns. Optional: paper_se or paper_sd+paper_count.")
    parser.add_argument("--metric", required=True, help="Metric column in our CSV, for example value or rescaled_subopt.")
    parser.add_argument("--keys", required=True, help="Comma-separated key columns used to align rows.")
    parser.add_argument("--margin", required=True, type=float, help="Equivalence margin on the metric scale.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reproduction/paper_comparison.csv"))
    args = parser.parse_args()

    keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    result = compare_to_paper(args.ours, args.paper, args.metric, keys, args.margin, args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print(f"equivalent_rows={int(result['equivalent'].sum())}/{len(result)} at alpha={args.alpha}, margin={args.margin}")


if __name__ == "__main__":
    main()
