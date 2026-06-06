from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from algs.ptree import (
    PL_aipw_score,
    PL_greedy,
    emp_eval_ptree,
    fit_policy_tree,
    predict_ptree,
)


@dataclass
class CVResult:
    beta: float
    beta_1se: float
    beta_lcb: float
    scores: List[List[float]]
    avg_scores: List[float]


def _selected(score_mat: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return score_mat[np.arange(score_mat.shape[0]), np.asarray(actions, dtype=int)]


def compute_variance_terms(wws, policy_actions, exs, min_propensity: float = 1e-12):
    """Compute the Vs and Vp terms used in Algorithm 1."""
    exs = np.asarray(exs, dtype=float)
    wws = np.asarray(wws, dtype=int).reshape(-1)
    policy_actions = np.asarray(policy_actions, dtype=int).reshape(-1)
    clipped_exs = np.clip(exs, min_propensity, None)

    match = wws == policy_actions
    chosen_exs = clipped_exs[np.arange(exs.shape[0]), policy_actions]
    vs_items = np.zeros(exs.shape[0], dtype=float)
    vs_items[match] = 1.0 / chosen_exs[match] ** 2
    vp_items = 1.0 / chosen_exs

    t_count = exs.shape[0]
    vs = float(np.sqrt(np.sum(vs_items)) / t_count)
    vp = float(np.sqrt(np.sum(vp_items)) / t_count)
    return {"Vs": vs, "Vp": vp, "Vs_items": vs_items, "Vp_items": vp_items}


def additive_penalty_items(wws, exs, min_propensity: float = 1e-12):
    """Return Gamma_s(t,a) and Gamma_p(t,a) from Algorithm 1."""
    exs = np.asarray(exs, dtype=float)
    wws = np.asarray(wws, dtype=int).reshape(-1)
    clipped_exs = np.clip(exs, min_propensity, None)
    gamma_s = np.zeros_like(clipped_exs, dtype=float)
    for arm in range(clipped_exs.shape[1]):
        idx = wws == arm
        gamma_s[idx, arm] = 1.0 / clipped_exs[idx, arm] ** 2
    gamma_p = 1.0 / clipped_exs
    return gamma_s, gamma_p


def ptree_aug_one(
    xxs,
    yobs,
    wws,
    what_0,
    exs,
    beta: float = 0.1,
    depth: int = 3,
    muxs=None,
    min_propensity: float = 1e-12,
):
    """One Algorithm-1 MM update for PPL."""
    t_count = np.asarray(exs).shape[0]
    gamma = PL_aipw_score(xxs, yobs, wws, exs, muxs=muxs, min_propensity=min_propensity)
    variance = compute_variance_terms(wws, what_0, exs, min_propensity=min_propensity)
    gamma_s, gamma_p = additive_penalty_items(wws, exs, min_propensity=min_propensity)

    bs = 0.0 if variance["Vs"] <= 0 else 1.0 / (2.0 * t_count * variance["Vs"])
    bp = 0.0 if variance["Vp"] <= 0 else 1.0 / (2.0 * t_count * variance["Vp"])
    penalized = gamma - beta * bs / t_count * gamma_s - beta * bp / t_count * gamma_p
    return fit_policy_tree(xxs, penalized, depth=depth)


def PL_pessimism(
    xxs,
    yobs,
    wws,
    exs,
    beta: float = 0.1,
    depth: int = 3,
    lower_bound: float | None = None,
    muxs=None,
    maxround: int = 50,
    verbose: bool = False,
    min_propensity: float = 1e-12,
):
    """Approximate PPL via Algorithm 1 from the paper.

    `lower_bound` is kept for backward compatibility and is interpreted as a
    numerical propensity floor only when explicitly provided.
    """
    if lower_bound is not None:
        min_propensity = max(min_propensity, float(lower_bound))

    current_tree = PL_greedy(xxs, yobs, wws, exs, depth=depth, muxs=muxs)
    fitted_actions = predict_ptree(current_tree, xxs)

    rounds = 0
    for rounds in range(1, maxround + 1):
        if verbose and (rounds <= 5 or rounds % 10 == 0):
            print(f"learning in round {rounds} ...")
        new_tree = ptree_aug_one(
            xxs,
            yobs,
            wws,
            what_0=fitted_actions,
            exs=exs,
            beta=beta,
            depth=depth,
            muxs=muxs,
            min_propensity=min_propensity,
        )
        new_actions = predict_ptree(new_tree, xxs)
        current_tree = new_tree
        if np.array_equal(new_actions, fitted_actions):
            fitted_actions = new_actions
            break
        fitted_actions = new_actions

    return current_tree, rounds


def _fold_indices(t_count: int, nfold: int) -> List[np.ndarray]:
    return [fold.astype(int) for fold in np.array_split(np.arange(t_count), nfold)]


def PPL_CV(
    xxs,
    yobs,
    wws,
    exs,
    beta_list: Sequence[float] = (0.1, 1, 5, 10, 15),
    Nfold: int = 5,
    depth: int = 3,
    lower_bound: float | None = None,
    muxs=None,
    maxround: int = 50,
    verbose: bool = False,
):
    """Algorithm 2 cross-validation for adaptive data."""
    t_count = np.asarray(exs).shape[0]
    if muxs is None:
        muxs = np.zeros_like(exs, dtype=float)

    folds = _fold_indices(t_count, Nfold)
    max_j = int(np.floor(3 * Nfold / 4))
    scores: List[List[float]] = [[] for _ in beta_list]

    for beta_idx, beta in enumerate(beta_list):
        for j in range(1, max_j + 1):
            train_idx = np.concatenate(folds[:j])
            eval_idx = np.concatenate(folds[j:])
            if len(train_idx) == 0 or len(eval_idx) == 0:
                continue
            tree, _ = PL_pessimism(
                xxs[train_idx],
                yobs[train_idx],
                wws[train_idx],
                exs[train_idx],
                beta=float(beta),
                depth=depth,
                lower_bound=lower_bound,
                muxs=muxs[train_idx],
                maxround=maxround,
                verbose=verbose,
            )
            score = emp_eval_ptree(
                tree,
                eval_xs=xxs[eval_idx],
                eval_yobs=yobs[eval_idx],
                eval_ws=wws[eval_idx],
                eval_exs=exs[eval_idx],
                eval_muxs=muxs[eval_idx],
            )
            scores[beta_idx].append(score)

    avg_scores = [float(np.mean(s)) if s else -np.inf for s in scores]
    std_scores = [float(np.std(s, ddof=1)) if len(s) > 1 else 0.0 for s in scores]
    opt_idx = int(np.argmax(avg_scores))
    lcb_scores = [
        avg - std / np.sqrt(len(s)) if s else -np.inf
        for avg, std, s in zip(avg_scores, std_scores, scores)
    ]
    lcb_idx = int(np.argmax(lcb_scores))

    best_score = avg_scores[opt_idx]
    best_se = std_scores[opt_idx] / np.sqrt(max(1, len(scores[opt_idx])))
    eligible = [i for i, avg in enumerate(avg_scores) if avg >= best_score - best_se]
    one_se_idx = int(max(eligible, key=lambda i: beta_list[i])) if eligible else opt_idx

    return (
        float(beta_list[opt_idx]),
        float(beta_list[one_se_idx]),
        float(beta_list[lcb_idx]),
        scores,
    )


def PPL_CV_v2(*args, **kwargs):
    return PPL_CV(*args, **kwargs)


def PPL_CV_v3(*args, **kwargs):
    return PPL_CV(*args, **kwargs)


def _published_variance_terms(wws, policy_actions, exs, lower_bound: float = 0.0001):
    """Variance proxy used by the public code that generated the paper figures."""
    exs = np.asarray(exs, dtype=float)
    wws = np.asarray(wws, dtype=int).reshape(-1)
    policy_actions = np.asarray(policy_actions, dtype=int).reshape(-1)
    t_count, arm_count = exs.shape

    vs_items = np.zeros((t_count, arm_count), dtype=float)
    vp_items = np.zeros((t_count, arm_count), dtype=float)
    vh_items = np.zeros((t_count, arm_count), dtype=float)
    for arm in range(arm_count):
        chosen = policy_actions == arm
        matched = chosen & (wws == arm)
        vs_items[matched, arm] = 1.0 / np.maximum(exs[matched, arm], 1e-12) ** 2
        propensity = np.maximum(exs[chosen, arm], lower_bound)
        vp_items[chosen, arm] = 1.0 / propensity
        vh_items[chosen, arm] = 1.0 / propensity**3

    vs = float(np.sqrt(vs_items.sum()) / t_count)
    vp = float(np.sqrt(vp_items.sum()) / t_count)
    vh = float(vh_items.sum() ** 0.25 / t_count)
    return {
        "Vs_items": vs_items,
        "Vp_items": vp_items,
        "Vh_items": vh_items,
        "Vs": vs,
        "Vp": vp,
        "Vh": vh,
        "maxV": max(vs, vp, vh),
    }


def _published_mm_update(
    xxs,
    yobs,
    wws,
    policy_actions,
    exs,
    beta: float,
    depth: int,
    lower_bound: float,
    muxs=None,
    gamma=None,
):
    """One update from the authors' released figure-generation implementation."""
    exs = np.asarray(exs, dtype=float)
    t_count = exs.shape[0]
    if t_count < 2:
        return PL_greedy(xxs, yobs, wws, exs, depth=depth, muxs=muxs)

    if gamma is None:
        gamma = PL_aipw_score(
            xxs,
            yobs,
            wws,
            exs,
            muxs=muxs,
            min_propensity=lower_bound,
        )
    else:
        gamma = np.asarray(gamma, dtype=float)
    variance = _published_variance_terms(wws, policy_actions, exs, lower_bound=lower_bound)
    max_v = variance["maxV"]
    if max_v <= 0:
        return fit_policy_tree(xxs, gamma, depth=depth)

    selected = _selected(gamma, policy_actions)
    observed = np.asarray(wws, dtype=int) == np.asarray(policy_actions, dtype=int)
    gamma_0 = np.where(observed, selected, 0.0)
    variance_items = variance["Vs_items"] + variance["Vp_items"]
    a0 = -float(gamma_0.sum()) / (t_count * (t_count - 1) * max_v)
    b0 = 1.0 / (2.0 * (t_count - 1) * max_v)
    majorizer = a0 * variance_items + b0 * variance_items**2
    return fit_policy_tree(xxs, gamma - float(beta) * majorizer, depth=depth)


def PL_pessimism_published(
    xxs,
    yobs,
    wws,
    exs,
    beta: float = 0.1,
    depth: int = 3,
    lower_bound: float = 0.0001,
    muxs=None,
    maxround: int = 50,
    verbose: bool = False,
    initial_tree=None,
    initial_actions=None,
    gamma=None,
):
    """Clean implementation of the public code used for the published figures.

    This intentionally remains separate from :func:`PL_pessimism`, which follows
    the displayed Algorithm 1 in Section 6.
    """
    current_tree = initial_tree
    if current_tree is None:
        current_tree = PL_greedy(xxs, yobs, wws, exs, depth=depth, muxs=muxs)
    fitted_actions = initial_actions
    if fitted_actions is None:
        fitted_actions = predict_ptree(current_tree, xxs)
    else:
        fitted_actions = np.asarray(fitted_actions, dtype=int)
    if gamma is None:
        gamma = PL_aipw_score(
            xxs,
            yobs,
            wws,
            exs,
            muxs=muxs,
            min_propensity=lower_bound,
        )
    t_count = len(fitted_actions)

    rounds = 0
    for rounds in range(1, maxround + 2):
        if verbose and (rounds <= 5 or rounds % 10 == 0):
            print(f"published-profile learning round {rounds} ...")
        new_tree = _published_mm_update(
            xxs,
            yobs,
            wws,
            fitted_actions,
            exs,
            beta=float(beta),
            depth=depth,
            lower_bound=lower_bound,
            muxs=muxs,
            gamma=gamma,
        )
        new_actions = predict_ptree(new_tree, xxs)
        current_tree = new_tree
        agreement = float(np.mean(new_actions == fitted_actions))
        fitted_actions = new_actions
        if agreement > 0.999 or rounds > maxround:
            break
    return current_tree, rounds


def PPL_CV_published(
    xxs,
    yobs,
    wws,
    exs,
    beta_list: Sequence[float] = (0.1, 1, 5, 10, 15),
    Nfold: int = 5,
    depth: int = 3,
    lower_bound: float = 0.0001,
    muxs=None,
    maxround: int = 50,
    verbose: bool = False,
):
    """Cross-validation behavior of the authors' released ``PPL_CV_v3``.

    Its two-prefix behavior for five folds differs from displayed Algorithm 2;
    the distinction is retained so published-figure and paper-spec runs can be
    audited independently.
    """
    xxs = np.asarray(xxs)
    yobs = np.asarray(yobs)
    wws = np.asarray(wws)
    exs = np.asarray(exs)
    if muxs is None:
        muxs = np.zeros_like(exs, dtype=float)
    else:
        muxs = np.asarray(muxs)

    folds = _fold_indices(len(xxs), Nfold)
    split_count = max(1, Nfold * 3 // 4 - 1)
    splits = []
    for fold_idx in range(split_count):
        boundary = int(folds[fold_idx][-1])
        train_idx = np.arange(0, boundary, dtype=int)
        eval_idx = np.arange(boundary, len(xxs), dtype=int)
        if len(train_idx) == 0 or len(eval_idx) == 0:
            continue
        train_tree = PL_greedy(
            xxs[train_idx],
            yobs[train_idx],
            wws[train_idx],
            exs[train_idx],
            depth=depth,
            muxs=muxs[train_idx],
        )
        train_actions = predict_ptree(train_tree, xxs[train_idx])
        train_gamma = PL_aipw_score(
            xxs[train_idx],
            yobs[train_idx],
            wws[train_idx],
            exs[train_idx],
            muxs=muxs[train_idx],
            min_propensity=lower_bound,
        )
        splits.append(
            (train_idx, eval_idx, train_tree, train_actions, train_gamma)
        )

    scores: List[List[float]] = [[] for _ in beta_list]
    for beta_idx, beta in enumerate(beta_list):
        for train_idx, eval_idx, train_tree, train_actions, train_gamma in splits:
            tree, _ = PL_pessimism_published(
                xxs[train_idx],
                yobs[train_idx],
                wws[train_idx],
                exs[train_idx],
                beta=float(beta),
                depth=depth,
                lower_bound=lower_bound,
                muxs=muxs[train_idx],
                maxround=maxround,
                verbose=verbose,
                initial_tree=train_tree,
                initial_actions=train_actions,
                gamma=train_gamma,
            )
            scores[beta_idx].append(
                emp_eval_ptree(
                    tree,
                    eval_xs=xxs[eval_idx],
                    eval_yobs=yobs[eval_idx],
                    eval_ws=wws[eval_idx],
                    eval_exs=exs[eval_idx],
                    eval_muxs=muxs[eval_idx],
                )
            )

    avg_scores = [float(np.mean(values)) if values else -np.inf for values in scores]
    std_scores = [float(np.std(values)) if values else np.inf for values in scores]
    opt_idx = int(np.argmax(avg_scores))
    lcb = [
        avg - std / np.sqrt(max(1, len(values)))
        for avg, std, values in zip(avg_scores, std_scores, scores)
    ]
    lcb_idx = int(np.argmax(lcb))
    ratios = [
        (avg - min(values)) / std if values and std > 0 else np.inf
        for avg, std, values in zip(avg_scores, std_scores, scores)
    ]
    eligible = [idx for idx, ratio in enumerate(ratios) if ratio <= 1]
    one_se_idx = max(eligible, key=lambda idx: beta_list[idx]) if eligible else lcb_idx
    return (
        float(beta_list[opt_idx]),
        float(beta_list[one_se_idx]),
        float(beta_list[lcb_idx]),
        scores,
    )
