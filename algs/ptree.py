from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np

from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


pt = importr("policytree")


def _as_2d_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.atleast_2d(x), dtype=float)


def _as_1d_int(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1)


def _call_r(fun, *args, **kwargs):
    with localconverter(default_converter + numpy2ri.converter):
        return fun(*args, **kwargs)


def PL_aipw_score(xxs, yobs, wws, exs, muxs=None, min_propensity: float = 1e-12):
    """Return the AIPW score matrix Gamma_t(a)."""
    exs = np.asarray(exs, dtype=float)
    yobs = np.asarray(yobs, dtype=float).reshape(-1)
    wws = _as_1d_int(wws)
    t_count, arm_count = exs.shape

    if muxs is None:
        gamma = np.zeros((t_count, arm_count), dtype=float)
    else:
        gamma = np.asarray(muxs, dtype=float).copy()

    clipped_exs = np.clip(exs, min_propensity, None)
    for arm in range(arm_count):
        idx = wws == arm
        gamma[idx, arm] += (yobs[idx] - gamma[idx, arm]) / clipped_exs[idx, arm]
    return gamma


def fit_policy_tree(xxs, gamma, depth: int = 3):
    """Fit an R policytree::hybrid_policy_tree for an additive score matrix."""
    xxs = _as_2d_float(xxs)
    gamma = np.asarray(gamma, dtype=float)
    return _call_r(pt.hybrid_policy_tree, X=xxs, Gamma=gamma, depth=int(depth))


def predict_ptree(ptree, xtest):
    """Predict zero-based arm indices from an R policy tree."""
    xtest = _as_2d_float(xtest)
    pred = _call_r(pt.predict_policy_tree, ptree, xtest)
    return np.asarray(pred, dtype=int).reshape(-1) - 1


def eval_ptree(ptree, eval_data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluate a learned tree on full potential outcomes."""
    eval_xs = np.asarray(eval_data["xs"], dtype=float)
    eval_ys = np.asarray(eval_data["ys"], dtype=float)
    pred = predict_ptree(ptree, eval_xs)
    rewards = eval_ys[np.arange(eval_ys.shape[0]), pred]
    return pred, rewards, float(np.mean(rewards))


def PL_greedy(xxs, yobs, wws, exs, depth: int = 3, muxs=None):
    """Greedy empirical welfare maximization with AIPW scores."""
    gamma = PL_aipw_score(xxs, yobs, wws, exs, muxs)
    return fit_policy_tree(xxs, gamma, depth=depth)


def emp_eval_ptree(ptree, eval_xs, eval_yobs, eval_ws, eval_exs, eval_muxs=None):
    """Evaluate a learned policy on logged data with AIPW scores."""
    gamma = PL_aipw_score(eval_xs, eval_yobs, eval_ws, eval_exs, muxs=eval_muxs)
    pred = predict_ptree(ptree, eval_xs)
    selected = gamma[np.arange(gamma.shape[0]), pred]
    return float(np.mean(selected))
