# Reproduction workspace for Pessimistic Policy Learning

This repository is an attempted reproduction workspace for the paper [Policy learning "without" overlap: Pessimism and generalized empirical Bernstein's inequality](https://arxiv.org/abs/2212.09900).

The code has been repaired to match the algorithms and experiments described in Sections 6 and 7 more closely:

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

## Data

OpenML datasets for Section 7.3 are cached under `data/openml/`, which is intentionally git-ignored. The committed manifest `notes/openml_cache_manifest.csv` records the 33 dataset names, OpenML IDs, targets, shapes, and class counts.

If an OpenML default target no longer behaves like a classification target, the real-data runner skips it and writes the reason to `real_skipped.csv`. In the current cache, `houses` has thousands of unique target values and is guarded this way.

## Running experiments

The maintained entry point is:

```bash
python experiments/reproduce.py --mode quick --experiment all --seed 20260605
```

Available experiments are `mab`, `tree`, `ts`, `ts-cv`, `real`, and `all`.

`quick` mode is a smoke-test mode that keeps runtime manageable. `full` mode keeps the paper-scale settings where feasible:

```bash
python experiments/reproduce.py --mode full --experiment mab --seed 20260605
python experiments/reproduce.py --mode full --experiment tree --seed 20260605
python experiments/reproduce.py --mode full --experiment ts --seed 20260605
python experiments/reproduce.py --mode full --experiment ts-cv --seed 20260605
python experiments/reproduce.py --mode full --experiment real --seed 20260605
```

The old script names in `experiments/` remain as compatibility wrappers and delegate to `experiments/reproduce.py`.

## Current reproduction artifacts

Generated artifacts are stored in `artifacts/reproduction/`:

- `full/`: paper-scale MAB results for Section 7.1.1.
- `quick/`: smoke-test results for MAB, contextual non-adaptive experiments, TS synthetic experiments, TS CV, and a real-data subset.
- `reproduction_report_zh.md`: Chinese report comparing the generated results with the paper and explaining statistical confidence.

The full Figure 5-10 grids were not all completed in this interaction because they require very large numbers of R `policytree` fits, especially for 33 OpenML datasets with 5-fold CV. The code paths and commands are present; the committed completed full-scale result is the MAB experiment.

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
