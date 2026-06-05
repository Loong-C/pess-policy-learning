<h1 align="center">
<p> Reproduction code for Pessimistic Policy Learning
</h1> 

This repository is being used as an attempted reproduction workspace for the paper [Policy learning "without" overlap: Pessimism and generalized empirical Bernstein's inequality](https://arxiv.org/abs/2212.09900).

The current goal is to document what can be reproduced, what environment and data are required, and where the checked code differs from the paper.


### Usage 
The original `install.sh` is not sufficient on this checkout because `setup.py` expects a missing `LICENSE` file and recent `rpy2` releases are incompatible with the current `numpy2ri.activate()` call in `algs/ptree.py`.

The working local reproduction environment is captured in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate pess-pl-legacy
```

On Windows, set these environment variables after activation so Python can find the local repository and R installation:

```powershell
conda env config vars set R_HOME="$env:CONDA_PREFIX\Lib\R" PYTHONPATH="$PWD"
conda deactivate
conda activate pess-pl-legacy
```

This environment has been smoke-tested with Python dependencies, R `policytree`/`grf`, `algs.ptree`, `algs.pess`, and a small policy-tree call.

OpenML datasets used by `experiments/real.py` have been downloaded to a local cache at `data/openml/`. The cache itself is git-ignored; `notes/openml_cache_manifest.csv` records the dataset names, OpenML IDs, targets, and shapes. The script currently lists 32 datasets, although the paper reports 33 real datasets.


Folder `experiments` contains scripts for reproducing the experiments:
- `MAB_batch.py`: experiments in Section 7.1.1 (PPL).
- `MAB_batch_clip.py`: experiments in Section 7.1.1 (with clipping).
- `synthetic_dt_linpess.py`: experiments in Section 7.1.2 (with linear PEVI).
- `synthetic_linear.py`: experiments in Section 7.2.1 (TS contextual bandit with well-specified exploration).
- `synthetic_opt.py`: experiments in Section 7.2.2 and 7.2.4 (cross validation) (TS contextual bandit with optimal overlap).
- `synthetic_miss.py`: experiments in Section 7.2.3 (TS contextual bandit with misspecified exploration).
- `real.py`: experiments in Section 7.3 (real datasets).
 
### Other files
Folder `utils` contains code for data generation and thopmson sampling. 

Folder `algs` contains the key algorithms for policy tree search, pessimistic policy learning, and cross validation.


#### Reference 

<a name="reference"></a>
```
@article{jin2025policy,
  title={Policy learning" without" overlap: Pessimism and generalized empirical bernstein's inequality},
  author={Jin, Ying and Ren, Zhimei and Yang, Zhuoran and Wang, Zhaoran},
  journal={Annals of Statistics (accepted)},
  year={2025+}
}
```
