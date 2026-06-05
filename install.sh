#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment.yml
conda run -n pess-pl-legacy python -m pip install -e .
