from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SYNTHETIC_MARGINS = [0.02, 0.05, 0.10]
MAB_MARGINS = [0.001, 0.0025, 0.005]
REAL_MARGINS = [0.01, 0.02, 0.05]


def _method_key(method: str, beta) -> str:
    if method in {"greedy", "CV_pess"}:
        return method
    return f"{method}_{float(beta):g}"


def _add_method_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["method_key"] = [
        _method_key(method, beta)
        for method, beta in zip(result["method"], result["beta"])
    ]
    return result


def _summarize_raw(
    raw: pd.DataFrame,
    keys: list[str],
    metric: str,
) -> pd.DataFrame:
    summary = raw.groupby(keys, dropna=False)[metric].agg(["mean", "std", "count"])
    summary = summary.reset_index().rename(
        columns={
            "mean": "ours_mean",
            "std": "ours_sd",
            "count": "ours_count",
        }
    )
    summary["ours_sd"] = summary["ours_sd"].fillna(0.0)
    summary["ours_se"] = summary["ours_sd"] / np.sqrt(summary["ours_count"])
    return summary


def _paper_se(reference: pd.DataFrame) -> pd.Series:
    if "paper_se" in reference:
        return reference["paper_se"].fillna(0.0)
    if {"paper_sd", "paper_count"}.issubset(reference.columns):
        return reference["paper_sd"] / np.sqrt(reference["paper_count"])
    return pd.Series(np.zeros(len(reference)), index=reference.index, dtype=float)


def _normal_tail_probability(z: pd.Series, upper: bool) -> pd.Series:
    values = stats.norm.sf(z) if upper else stats.norm.cdf(z)
    return pd.Series(values, index=z.index)


def add_equivalence_statistics(
    comparison: pd.DataFrame,
    margin: float,
    alpha: float,
) -> pd.DataFrame:
    result = comparison.copy()
    result["margin"] = margin
    result["alpha"] = alpha
    result["diff"] = result["ours_mean"] - result["paper_mean"]
    result["combined_se"] = np.sqrt(
        result["ours_se"].fillna(0.0) ** 2
        + result["paper_se_used"].fillna(0.0) ** 2
    )

    nonzero_se = result["combined_se"].replace(0.0, np.nan)
    lower_z = (result["diff"] + margin) / nonzero_se
    upper_z = (result["diff"] - margin) / nonzero_se
    result["p_gt_minus_margin"] = _normal_tail_probability(lower_z, upper=True)
    result["p_lt_plus_margin"] = _normal_tail_probability(upper_z, upper=False)

    zero_se = result["combined_se"] == 0.0
    result.loc[zero_se, "p_gt_minus_margin"] = np.where(
        result.loc[zero_se, "diff"] > -margin, 0.0, 1.0
    )
    result.loc[zero_se, "p_lt_plus_margin"] = np.where(
        result.loc[zero_se, "diff"] < margin, 0.0, 1.0
    )
    result["tost_p"] = result[
        ["p_gt_minus_margin", "p_lt_plus_margin"]
    ].max(axis=1)

    z_equivalence = stats.norm.ppf(1.0 - alpha)
    z_difference = stats.norm.ppf(1.0 - alpha / 2.0)
    result["ci90_low"] = result["diff"] - z_equivalence * result["combined_se"]
    result["ci90_high"] = result["diff"] + z_equivalence * result["combined_se"]
    result["ci95_low"] = result["diff"] - z_difference * result["combined_se"]
    result["ci95_high"] = result["diff"] + z_difference * result["combined_se"]
    result["equivalent"] = (result["ci90_low"] > -margin) & (
        result["ci90_high"] < margin
    )
    result["different"] = (result["ci95_low"] > margin) | (
        result["ci95_high"] < -margin
    )
    result["classification"] = np.select(
        [result["equivalent"], result["different"]],
        ["equivalent", "different"],
        default="inconclusive",
    )
    return result


def compare_cells(
    raw: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    metric: str,
    margin: float,
    alpha: float,
) -> pd.DataFrame:
    ours = _summarize_raw(raw, keys, metric)
    paper = reference.copy()
    paper["paper_se_used"] = _paper_se(paper)
    comparison = ours.merge(paper, on=keys, how="inner", validate="one_to_one")
    if comparison.empty:
        raise ValueError(f"No result cells matched reference keys {keys}.")
    return add_equivalence_statistics(comparison, margin, alpha)


def _figure_summary(
    figure: int,
    comparison: pd.DataFrame,
    reference_cells: int,
    ours_cells: int,
) -> dict:
    differences = comparison["diff"].to_numpy()
    equivalent = int(comparison["equivalent"].sum())
    different = int(comparison["different"].sum())
    return {
        "figure": figure,
        "margin": float(comparison["margin"].iloc[0]),
        "alpha": float(comparison["alpha"].iloc[0]),
        "matched_cells": len(comparison),
        "reference_cells": reference_cells,
        "ours_cells": ours_cells,
        "equivalent_cells": equivalent,
        "equivalent_rate": equivalent / len(comparison),
        "different_cells": different,
        "inconclusive_cells": len(comparison) - equivalent - different,
        "mean_diff": float(np.mean(differences)),
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences**2))),
        "max_abs_diff": float(np.max(np.abs(differences))),
        "all_cells_equivalent": equivalent == len(comparison),
    }


def _load_synthetic_inputs(root: Path, reference_root: Path):
    data_root = root / "data"
    tree = _add_method_key(
        pd.read_csv(data_root / "contextual_nonadaptive_results.csv")
    )
    ts = _add_method_key(pd.read_csv(data_root / "ts_synthetic_results.csv"))
    ts_cv = _add_method_key(pd.read_csv(data_root / "ts_cv_results.csv"))
    references = {
        figure: _add_method_key(
            pd.read_csv(reference_root / f"figure{figure}.csv")
        )
        for figure in range(5, 10)
    }
    return tree, ts, ts_cv, references


def _write_comparison(
    output_root: Path,
    figure: int,
    margin: float,
    comparison: pd.DataFrame,
) -> None:
    margin_label = str(margin).replace(".", "p")
    comparison.to_csv(
        output_root / "data" / f"figure{figure}_comparison_margin_{margin_label}.csv",
        index=False,
    )


def analyze_mab(
    root: Path,
    reference_root: Path,
    output_root: Path,
    alpha: float,
) -> list[dict]:
    raw = pd.read_csv(root / "data" / "mab_results.csv")
    reference = pd.read_csv(reference_root / "figure4_mab.csv")
    keys = ["setting_name", "T", "method"]
    ours_cells = len(raw.groupby(keys))
    summaries = []
    for margin in MAB_MARGINS:
        comparison = compare_cells(
            raw,
            reference,
            keys,
            "rescaled_subopt",
            margin,
            alpha,
        )
        _write_comparison(output_root, 4, margin, comparison)
        summaries.append(
            _figure_summary(4, comparison, len(reference), ours_cells)
        )
    return summaries


def analyze_synthetic(
    root: Path,
    reference_root: Path,
    output_root: Path,
    alpha: float,
) -> tuple[list[dict], dict[int, pd.DataFrame]]:
    tree, ts, ts_cv, references = _load_synthetic_inputs(root, reference_root)
    figure_inputs = {
        5: (
            tree,
            ["scenario", "decay", "T", "method_key"],
        ),
        6: (
            ts[ts["setting"] == 1],
            ["setting", "batch_size", "floor", "T", "method_key"],
        ),
        7: (
            ts[ts["setting"] == 2],
            ["setting", "batch_size", "floor", "T", "method_key"],
        ),
        8: (
            ts[ts["setting"] == 3],
            ["setting", "batch_size", "floor", "T", "method_key"],
        ),
    }

    fixed_figure9 = ts[
        (ts["setting"] == 2)
        & (ts["batch_size"] == 10)
        & (ts["method"] == "pess")
    ]
    cv_figure9 = ts_cv[
        (ts_cv["setting"] == 2)
        & (ts_cv["batch_size"] == 10)
        & (ts_cv["method"].isin(["greedy", "CV_pess"]))
    ]
    figure_inputs[9] = (
        pd.concat([fixed_figure9, cv_figure9], ignore_index=True),
        ["setting", "batch_size", "floor", "T", "method_key"],
    )

    summaries = []
    primary_comparisons = {}
    for figure, (raw, keys) in figure_inputs.items():
        reference = references[figure]
        ours_cells = len(raw.groupby(keys, dropna=False))
        for margin in SYNTHETIC_MARGINS:
            comparison = compare_cells(
                raw,
                reference,
                keys,
                "value",
                margin,
                alpha,
            )
            _write_comparison(output_root, figure, margin, comparison)
            summaries.append(
                _figure_summary(
                    figure,
                    comparison,
                    len(reference),
                    ours_cells,
                )
            )
            if np.isclose(margin, 0.05):
                primary_comparisons[figure] = comparison
    return summaries, primary_comparisons


def _paired_tost(diff: pd.Series, margin: float, alpha: float) -> dict:
    values = diff.dropna().to_numpy()
    count = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if count > 1 else 0.0
    se = sd / np.sqrt(count) if count else np.nan
    if count > 1 and se > 0:
        degrees = count - 1
        p_lower = float(stats.t.sf((mean + margin) / se, degrees))
        p_upper = float(stats.t.cdf((mean - margin) / se, degrees))
        critical = float(stats.t.ppf(1.0 - alpha, degrees))
    else:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        critical = 0.0
    ci_low = mean - critical * se
    ci_high = mean + critical * se
    tost_p = max(p_lower, p_upper)
    return {
        "count": count,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "ci90_low": ci_low,
        "ci90_high": ci_high,
        "tost_p": tost_p,
        "equivalent": tost_p < alpha,
    }


def analyze_real(
    root: Path,
    reference_root: Path,
    output_root: Path,
    alpha: float,
) -> tuple[list[dict], pd.DataFrame]:
    raw = pd.read_csv(root / "data" / "real_results.csv")
    reference = pd.read_csv(reference_root / "figure10_real.csv")
    keys = ["dataset", "batch_size", "floor"]
    comparison = raw.merge(reference, on=keys, how="inner")
    if comparison.empty:
        raise ValueError("No Figure 10 real-data points matched by dataset name.")
    comparison["diff"] = (
        comparison["improvement"] - comparison["paper_improvement"]
    )
    comparison["sign_agreement"] = (
        np.sign(comparison["improvement"])
        == np.sign(comparison["paper_improvement"])
    )
    comparison.to_csv(
        output_root / "data" / "figure10_real_comparison.csv",
        index=False,
    )

    summaries = []
    for margin in REAL_MARGINS:
        paired = _paired_tost(comparison["diff"], margin, alpha)
        within = int((comparison["diff"].abs() <= margin).sum())
        summaries.append(
            {
                "figure": 10,
                "margin": margin,
                "alpha": alpha,
                "matched_cells": len(comparison),
                "reference_cells": len(reference),
                "ours_cells": len(raw),
                "equivalent_cells": np.nan,
                "equivalent_rate": np.nan,
                "different_cells": np.nan,
                "inconclusive_cells": np.nan,
                "mean_diff": paired["mean_diff"],
                "mae": float(comparison["diff"].abs().mean()),
                "rmse": float(np.sqrt(np.mean(comparison["diff"] ** 2))),
                "max_abs_diff": float(comparison["diff"].abs().max()),
                "all_cells_equivalent": np.nan,
                "paired_tost_p": paired["tost_p"],
                "paired_mean_equivalent": paired["equivalent"],
                "paired_ci90_low": paired["ci90_low"],
                "paired_ci90_high": paired["ci90_high"],
                "descriptive_cells_within_margin": within,
                "descriptive_within_margin_rate": within / len(comparison),
                "sign_agreement_rate": float(comparison["sign_agreement"].mean()),
            }
        )

    panel_rows = []
    for (batch_size, floor), group in comparison.groupby(["batch_size", "floor"]):
        for margin in REAL_MARGINS:
            paired = _paired_tost(group["diff"], margin, alpha)
            panel_rows.append(
                {
                    "batch_size": batch_size,
                    "floor": floor,
                    "margin": margin,
                    **paired,
                    "mae": float(group["diff"].abs().mean()),
                    "sign_agreement_rate": float(group["sign_agreement"].mean()),
                }
            )
    pd.DataFrame(panel_rows).to_csv(
        output_root / "data" / "figure10_panel_statistics.csv",
        index=False,
    )
    return summaries, comparison


def _plot_synthetic_agreement(
    comparisons: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for figure, comparison in comparisons.items():
        ax.scatter(
            comparison["paper_mean"],
            comparison["ours_mean"],
            s=13,
            alpha=0.55,
            label=f"Figure {figure}",
        )
    limits = [
        min(frame["paper_mean"].min() for frame in comparisons.values()),
        max(frame["paper_mean"].max() for frame in comparisons.values()),
    ]
    limits[0] = min(
        limits[0], min(frame["ours_mean"].min() for frame in comparisons.values())
    )
    limits[1] = max(
        limits[1], max(frame["ours_mean"].max() for frame in comparisons.values())
    )
    ax.plot(limits, limits, color="black", linewidth=1.0)
    ax.fill_between(
        limits,
        np.asarray(limits) - 0.05,
        np.asarray(limits) + 0.05,
        color="#cccccc",
        alpha=0.25,
        label="±0.05",
    )
    ax.set(
        xlabel="Paper value",
        ylabel="Reproduced value",
        xlim=limits,
        ylim=limits,
        title="Synthetic contextual experiments: paper vs reproduction",
    )
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_equivalence_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    synthetic = summary[summary["figure"].between(5, 9)].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for figure, group in synthetic.groupby("figure"):
        ax.plot(
            group["margin"],
            group["equivalent_rate"],
            marker="o",
            label=f"Figure {figure}",
        )
    ax.set(
        xlabel="Equivalence margin",
        ylabel="Equivalent cell proportion",
        ylim=(-0.03, 1.03),
        title="TOST sensitivity at alpha=0.05",
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the complete Section 7 reproduction against vector references."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/reproduction/published/full"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("artifacts/reproduction/paper_reference"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    output_root = args.root
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)

    summaries = analyze_mab(
        args.root, args.reference_root, output_root, args.alpha
    )
    synthetic_summaries, comparisons = analyze_synthetic(
        args.root, args.reference_root, output_root, args.alpha
    )
    summaries.extend(synthetic_summaries)
    real_summaries, real_comparison = analyze_real(
        args.root, args.reference_root, output_root, args.alpha
    )
    summaries.extend(real_summaries)

    summary = pd.DataFrame(summaries).sort_values(["figure", "margin"])
    summary.to_csv(output_root / "data" / "equivalence_summary.csv", index=False)
    (output_root / "data" / "equivalence_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _plot_synthetic_agreement(
        comparisons,
        output_root / "figures" / "paper_vs_reproduction_synthetic.png",
    )
    _plot_equivalence_sensitivity(
        summary,
        output_root / "figures" / "equivalence_sensitivity.png",
    )

    print(summary.to_string(index=False))
    print(f"matched Figure 10 points: {len(real_comparison)}")


if __name__ == "__main__":
    main()
