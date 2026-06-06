# Reproduction workspace for Pessimistic Policy Learning

This repository is an attempted reproduction workspace for the paper [Policy learning "without" overlap: Pessimism and generalized empirical Bernstein's inequality](https://arxiv.org/abs/2212.09900).

The code has been repaired and audited against both Sections 6-7 and the
authors' public figure-generation code. The audit found material differences
between the displayed paper specification and the implementation underlying
some published figures, so the runner exposes two explicit protocols:

- `published` (default): reproduces the public implementation and published numerical figures.
- `paper-spec`: follows the displayed Algorithm 1/2 and the reward formulas written in Sections 6-7.

This distinction is intentional. It prevents a result from being called
"faithful" without saying whether that means faithful to the prose/equations or
to the published numerical assets.

The `paper-spec` path includes the following literal-specification repairs:

- Algorithm 1 PPL now uses the paper's MM update with the additive `Gamma_s` and `Gamma_p` penalty terms.
- Algorithm 2 cross-validation now uses consecutive folds and prefix-train/suffix-evaluate splits for adaptive data.
- Synthetic DGPs now match the reward formulas in Section 7.
- The MAB experiment uses `mu / sqrt(T)` and reports rescaled suboptimality.
- The Section 7.1.2 linear PEVI baseline is included.
- The real-data OpenML list now contains 33 datasets, including `skin-segmentation`.

## Environment

The working local environment is captured in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate pess-pl-legacy
python -m pip install -e .
```

On Windows, set these conda environment variables after activation:

```powershell
conda env config vars set R_HOME="$env:CONDA_PREFIX\Lib\R" PYTHONPATH="$PWD"
conda deactivate
conda activate pess-pl-legacy
```

This environment has been tested with Python dependencies, R `policytree`/`grf`, editable installation, and Python-to-R policy-tree calls.
It also includes `pytest`; the protocol regression suite runs with
`python -m pytest -q tests/test_protocols.py`.

## Data

OpenML datasets for Section 7.3 are cached under `data/openml/`, which is intentionally git-ignored. The committed manifest `notes/openml_cache_manifest.csv` records the 33 dataset names, OpenML IDs, targets, shapes, and class counts.

If an OpenML default target no longer behaves like a classification target, the real-data runner skips it and writes the reason to `real_skipped.csv`. In the current cache, `houses` has thousands of unique target values and is guarded this way.

## Running experiments

The maintained entry point is:

```bash
python experiments/reproduce.py --mode quick --protocol published --experiment all --seed 20260605
```

Available experiments are `mab`, `tree`, `ts`, `ts-cv`, `real`, and `all`. Slow experiments support chunk-level parallelism and resume:

```bash
python experiments/reproduce.py --mode full --protocol published --experiment tree --jobs 20 --resume --seed 20260605
python experiments/reproduce.py --mode full --protocol published --experiment ts --jobs 20 --resume --seed 20260605
python experiments/reproduce.py --mode full --protocol published --experiment real --jobs 2 --resume --seed 20260605
```

Completed chunks are written below
`artifacts/reproduction/<protocol>/<mode>/data/chunks/`; rerunning with
`--resume` skips chunks already on disk.

`quick` mode is a smoke-test mode that keeps runtime manageable. `full` mode keeps the paper-scale settings where feasible:

```bash
python experiments/reproduce.py --mode full --protocol published --experiment mab --seed 20260605
python experiments/reproduce.py --mode full --protocol published --experiment tree --jobs 20 --resume --seed 20260605
python experiments/reproduce.py --mode full --protocol published --experiment ts --jobs 20 --resume --seed 20260605
python experiments/reproduce.py --mode full --protocol published --experiment ts-cv --jobs 20 --resume --seed 20260605
python experiments/reproduce.py --mode full --protocol published --experiment real --jobs 2 --resume --seed 20260605
```

The old script names in `experiments/` remain as compatibility wrappers and delegate to `experiments/reproduce.py`.

## Current reproduction artifacts

Generated artifacts are stored in `artifacts/reproduction/`:

- `published/full/`: paper-scale published-protocol results.
- `published/quick/`: published-protocol smoke-test results.
- `paper-spec/`: literal paper-specification results when requested.
- `paper_reference/`: numerical references digitized from the paper's vector figures.
- `reproduction_report_zh.md`: Chinese report comparing the generated results with the paper and explaining statistical confidence.

The full Figure 5-10 grids are resumable multi-day CPU workloads. Runtime chunk
files are intentionally ignored; consolidated CSVs, figures, reports, and run
configurations are committed after completion.

## Statistical comparison with the paper

A statistically meaningful "successful reproduction" claim needs numeric paper references, not only visual inspection of the published figures. Use `experiments/compare_to_paper.py` once you have a CSV with `paper_mean` and matching key columns, optionally `paper_se` or `paper_sd` plus `paper_count`:

```bash
python experiments/compare_to_paper.py \
  --ours artifacts/reproduction/published/full/data/mab_results.csv \
  --paper artifacts/reproduction/paper_reference/figure4_mab.csv \
  --metric rescaled_subopt \
  --keys setting_name,T,method \
  --margin 0.005 \
  --alpha 0.05 \
  --out artifacts/reproduction/published/full/data/figure4_equivalence.csv
```

`experiments/extract_mab_paper_reference.py` extracts Figure 4 means and
uncertainty from the arXiv vector PDF after conversion with `pdftocairo -svg`.
`experiments/extract_contextual_paper_references.py` similarly extracts the
curve coordinates for Figures 5-9:

```bash
python experiments/extract_contextual_paper_references.py \
  --source-dir tmp/arxiv-2212.09900/figs \
  --figure all \
  --out-dir artifacts/reproduction/paper_reference
```

Figure 5's caption names six penalty values, but its vector paths and the
released experiment script contain three additional PPL values
(`0.0001`, `0.001`, and `0.01`). The `published` protocol follows those
numerical assets; the discrepancy is retained in the audit log.
The comparison script uses a two one-sided tests equivalence check. An
equivalence margin and significance level must be declared before interpreting
the result.

## Repository layout

- `algs/`: policy tree fitting, greedy policy learning, PPL, and cross-validation.
- `utils/`: data-generating processes, Thompson Sampling, and experiment collection helpers.
- `experiments/`: maintained reproduction CLI plus compatibility wrappers.
- `paper/`: local copy of the paper used for checking Sections 6 and 7.
- `notes/`: reproduction log and OpenML cache manifest.

## Reference

```bibtex
@article{jin2025policy,
  title={Policy learning" without" overlap: Pessimism and generalized empirical bernstein's inequality},
  author={Jin, Ying and Ren, Zhimei and Yang, Zhuoran and Wang, Zhaoran},
  journal={Annals of Statistics (accepted)},
  year={2025+}
}
```
