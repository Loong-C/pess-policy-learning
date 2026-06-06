from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import seaborn as sns
from scipy import stats

from algs.pess import PPL_CV, PPL_CV_published, PL_pessimism, PL_pessimism_published
from algs.ptree import PL_aipw_score, PL_greedy, eval_ptree, predict_ptree
from utils.dgp import (
    MABandit,
    MultiLinear,
    MultiQuad,
    PublishedMultiLinear,
    PublishedMultiQuad,
    generate_bandit_data,
)
from utils.experiment import run_experiment, run_experiment_opt


ARTIFACT_ROOT = Path("artifacts/reproduction")
OPENML_CACHE = Path("data/openml")
FULL_CONTEXTUAL_T_VALUES = [500, 1000, 2000, 5000]
TREE_PESS_BETAS = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10]
TREE_LINEAR_BETAS = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10]
openml.config.cache_directory = str(OPENML_CACHE.resolve())

MAB_SETTINGS = {
    1: ("Optimal", np.array([0, 0.05, 0.01, 0, -0.01], dtype=float), np.array([0.07, 0.9, 0.01, 0.01, 0.01])),
    2: ("Suboptimal", np.array([0, 0.05, 0.04, 0.01, -0.01], dtype=float), np.array([0.07, 0.1, 0.8, 0.02, 0.01])),
    3: ("Uniform", np.array([0, 0.05, 0.03, 0.01, -0.01], dtype=float), np.array([0.2] * 5)),
}

REAL_DATASETS = [
    "waveform-5000",
    "Long",
    "cmc",
    "artificial-characters",
    "Click_prediction_small",
    "skin-segmentation",
    "allrep",
    "mfeat-morphological",
    "satellite_image",
    "jungle_chess_2pcs_endgame_elephant_elephant",
    "wilt",
    "Satellite",
    "ringnorm",
    "mammography",
    "delta_ailerons",
    "PhishingWebsites",
    "splice",
    "pendigits",
    "texture",
    "cardiotocography",
    "volcanoes-d4",
    "volcanoes-b3",
    "dis",
    "optdigits",
    "electricity",
    "kr-vs-kp",
    "bank-marketing",
    "satimage",
    "MagicTelescope",
    "houses",
    "eeg-eye-state",
    "car",
    "segment",
]


@dataclass
class ModeConfig:
    mode: str
    mab_nrep: int
    tree_nrep: int
    ts_nrep: int
    real_nrep: int
    t_eval: int
    num_mc: int


def mode_config(mode: str) -> ModeConfig:
    if mode == "full":
        return ModeConfig(mode, mab_nrep=1000, tree_nrep=200, ts_nrep=200, real_nrep=1, t_eval=100000, num_mc=1000)
    return ModeConfig(mode, mab_nrep=30, tree_nrep=1, ts_nrep=1, real_nrep=1, t_eval=1000, num_mc=100)


def outdir_for(mode: str, protocol: str) -> Path:
    outdir = ARTIFACT_ROOT / protocol / mode
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    return outdir


def _ppl_fit(protocol: str, *args, **kwargs):
    if protocol == "published":
        return PL_pessimism_published(*args, **kwargs)
    return PL_pessimism(*args, **kwargs)


def _ppl_cv(protocol: str, *args, **kwargs):
    if protocol == "published":
        return PPL_CV_published(*args, **kwargs)
    return PPL_CV(*args, **kwargs)


def _published_initialization(protocol, tree, xxs, yobs, wws, exs, muxs=None):
    if protocol != "published":
        return {}
    return {
        "initial_tree": tree,
        "initial_actions": predict_ptree(tree, xxs),
        "gamma": PL_aipw_score(
            xxs,
            yobs,
            wws,
            exs,
            muxs=muxs,
            min_propensity=0.0001,
        ),
    }


def _stable_seed(seed: int, *parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"|")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % (2**31 - 1)


def _chunk_path(chunk_dir: Path, task: dict) -> Path:
    return chunk_dir / f"{task['task_id']}.csv"


def _run_chunked(tasks, worker, chunk_dir: Path, combined_path: Path | None, jobs: int = 1, resume: bool = False) -> pd.DataFrame:
    """Run independent task chunks with optional process-level parallelism."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    error_path = chunk_dir / "errors.jsonl"
    if not resume:
        for path in chunk_dir.glob("*.csv"):
            path.unlink()
        error_path.unlink(missing_ok=True)

    pending = [task for task in tasks if not (resume and _chunk_path(chunk_dir, task).exists())]
    jobs = max(1, int(jobs))

    def write_result(task, result):
        df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        df.to_csv(_chunk_path(chunk_dir, task), index=False)

    def write_error(task, exc):
        record = {"task_id": task["task_id"], "error_type": type(exc).__name__, "error": str(exc)}
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if jobs == 1:
        for task in pending:
            try:
                write_result(task, worker(task))
            except Exception as exc:
                write_error(task, exc)
    elif pending:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_map = {executor.submit(worker, task): task for task in pending}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    write_result(task, future.result())
                except Exception as exc:
                    write_error(task, exc)

    frames = []
    for path in sorted(chunk_dir.glob("*.csv")):
        if path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined_path is not None:
        df.to_csv(combined_path, index=False)
    return df


def _safe_argmax(values: np.ndarray) -> int:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0
    cleaned = np.where(finite, values, -np.inf)
    return int(np.argmax(cleaned))


def _arm_mean_std(ys: np.ndarray, ws: np.ndarray, arm: int):
    obs = ys[ws == arm, arm]
    if len(obs) == 0:
        return -np.inf, np.inf
    mean = float(np.mean(obs))
    sd = float(np.std(obs, ddof=1)) if len(obs) > 1 else np.inf
    return mean, sd


def run_mab(mode: str, protocol: str, seed: int = 0) -> pd.DataFrame:
    cfg = mode_config(mode)
    outdir = outdir_for(mode, protocol)
    rng = np.random.default_rng(seed)
    beta_list = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15]
    t_list = [100, 500, 1000, 2000, 5000, 10000, 20000] if mode == "full" else [100, 500, 1000]
    rows = []

    for setting, (setting_name, abs_mu, ps) in MAB_SETTINGS.items():
        for T in t_list:
            true_mu = abs_mu / math.sqrt(T)
            dgp = MABandit(mu=true_mu, sigma=0.1, ps=ps)
            for rep in range(cfg.mab_nrep):
                if protocol == "published":
                    legacy_rng = np.random.RandomState(rep)
                    ys = np.column_stack(
                        [true_mu[arm] + legacy_rng.normal(size=T) * dgp.sigma for arm in range(5)]
                    )
                    ws = legacy_rng.choice(5, size=T, p=ps)
                else:
                    rep_seed = int(rng.integers(0, 2**31 - 1))
                    data = dgp.sample_data(T, seed=rep_seed)
                    ys = data["ys"]
                    ws = dgp.sample_arms(T, seed=rep_seed + 1)

                means = np.zeros(5)
                variance_proxy = np.zeros(5)
                for arm in range(5):
                    if protocol == "published":
                        observed = ys[ws == arm, arm]
                        means[arm] = float(np.mean(observed)) if len(observed) else np.nan
                    else:
                        means[arm], _ = _arm_mean_std(ys, ws, arm)
                    observed_rate = float(np.mean(ws == arm))
                    variance_proxy[arm] = max(
                        math.sqrt(T * observed_rate / ps[arm] ** 2) / T,
                        (T * observed_rate / ps[arm] ** 3) ** 0.25 / T,
                    )
                choose = np.argmax if protocol == "published" else _safe_argmax
                actions = {"greedy": int(choose(means))}
                for beta in beta_list:
                    actions[f"pess_{beta:g}"] = int(choose(means - beta * variance_proxy))

                clip_weight = np.minimum(5.0, 1.0 / ps)
                clip_scores = np.zeros((T, 5), dtype=float)
                for arm in range(5):
                    clip_scores[:, arm] = ys[:, arm] * (ws == arm) * clip_weight[arm]
                clip_mean = np.mean(clip_scores, axis=0)
                clip_sd = np.std(clip_scores, axis=0, ddof=1)
                for beta in beta_list:
                    actions[f"clip_{beta:g}"] = _safe_argmax(clip_mean - beta * clip_sd / math.sqrt(T))

                for method, action in actions.items():
                    rows.append(
                        {
                            "experiment": "mab",
                            "setting": setting,
                            "setting_name": setting_name,
                            "T": T,
                            "rep": rep,
                            "method": method,
                            "action": action,
                            "reward_abs_mu": abs_mu[action],
                            "rescaled_subopt": float(np.max(abs_mu) - abs_mu[action]),
                            "correct": int(action == int(np.argmax(abs_mu))),
                        }
                    )

    df = pd.DataFrame(rows)
    path = outdir / "data" / "mab_results.csv"
    df.to_csv(path, index=False)
    plot_mab(df, outdir)
    return df


def plot_mab(df: pd.DataFrame, outdir: Path):
    sns.set_theme(style="whitegrid")
    base = df[~df["method"].str.startswith("clip_")].copy()
    summary = base.groupby(["setting_name", "T", "method"], as_index=False).agg(
        mean=("rescaled_subopt", "mean"), sd=("rescaled_subopt", "std")
    )
    g = sns.relplot(
        data=summary,
        x="T",
        y="mean",
        hue="method",
        col="setting_name",
        kind="line",
        marker="o",
        facet_kws={"sharey": True},
        height=3.3,
        aspect=1.05,
    )
    for ax in g.axes.flat:
        ax.set_xscale("log")
    g.set_axis_labels("Sample size", "Rescaled suboptimality")
    g.figure.tight_layout()
    g.figure.savefig(outdir / "figures" / "figure4_mab.png", dpi=200)
    plt.close(g.figure)

    clip = df[df["method"].str.startswith(("pess_", "clip_"))].copy()
    clip["family"] = np.where(clip["method"].str.startswith("clip_"), "clip", "pess")
    clip["beta"] = clip["method"].str.extract(r"_(.+)$")[0]
    pivot = (
        clip.groupby(["setting_name", "T", "beta", "family"])["rescaled_subopt"]
        .mean()
        .unstack("family")
        .reset_index()
    )
    if {"clip", "pess"}.issubset(pivot.columns):
        g = sns.relplot(
            data=pivot,
            x="clip",
            y="pess",
            hue="beta",
            size="T",
            col="setting_name",
            kind="scatter",
            height=3.3,
            aspect=1.0,
        )
        g.set_axis_labels("Clipped method suboptimality", "PPL suboptimality")
        g.figure.tight_layout()
        g.figure.savefig(outdir / "figures" / "figure11_mab_clip.png", dpi=200)
        plt.close(g.figure)

    freq_methods = ["greedy", "pess_0.1", "pess_0.2", "pess_1", "pess_10"]
    freq = df[df["method"].isin(freq_methods)].copy()
    freq_summary = freq.groupby(["setting_name", "T", "method", "action"], as_index=False).size()
    totals = freq.groupby(["setting_name", "T", "method"], as_index=False).size().rename(columns={"size": "total"})
    freq_summary = freq_summary.merge(totals, on=["setting_name", "T", "method"])
    freq_summary["frequency"] = freq_summary["size"] / freq_summary["total"]
    g = sns.relplot(
        data=freq_summary,
        x="T",
        y="frequency",
        hue="action",
        col="method",
        row="setting_name",
        kind="line",
        marker="o",
        height=2.2,
        aspect=1.0,
    )
    for ax in g.axes.flat:
        ax.set_xscale("log")
    g.figure.tight_layout()
    g.figure.savefig(outdir / "figures" / "figure12_mab_frequency.png", dpi=200)
    plt.close(g.figure)


def _contextual_propensities(muxs, setting: int, decay: float | None):
    T, K = muxs.shape
    ps = np.zeros((T, K), dtype=float)
    opt = np.argmax(muxs, axis=1)
    for i in range(T):
        t = i + 1
        if setting == 1:
            low = 0.001 if decay is None else 0.1 * t ** (-decay)
            ps[i, :] = low
            ps[i, opt[i]] = 1.0 - low * (K - 1)
        else:
            high = muxs[i] >= np.median(muxs[i])
            if decay is None:
                low_prob = 0.001
                if setting == 2:
                    block = int(i / max(1, T / 5))
                    high_gets_good = block % 2 == 0
                else:
                    high_gets_good = setting != 3
                if high_gets_good:
                    high_prob = (1.0 - low_prob * np.sum(~high)) / np.sum(high)
                    ps[i, high] = high_prob
                    ps[i, ~high] = low_prob
                else:
                    ps[i, high] = low_prob
                    ps[i, ~high] = (1.0 - low_prob * np.sum(high)) / np.sum(~high)
            else:
                if setting == 2:
                    block = int(i / max(1, T / 5))
                    high_gets_good = block % 2 == 0
                else:
                    high_gets_good = False
                good = 0.2 * (1.0 - t ** (-decay))
                poor = 0.2 * t ** (-decay)
                if high_gets_good:
                    ps[i, high] = good
                    ps[i, ~high] = poor
                else:
                    ps[i, high] = poor
                    ps[i, ~high] = good
        ps[i, :] = ps[i, :] / ps[i, :].sum()
    return ps


def _linear_pevi_design(xs: np.ndarray, arms: np.ndarray, arm_count: int) -> np.ndarray:
    """Action-block features from Section 7.1.2: (1, x_1) for each arm."""
    xs = np.asarray(xs, dtype=float)
    arms = np.asarray(arms, dtype=int).reshape(-1)
    design = np.zeros((xs.shape[0], 2 * arm_count), dtype=float)
    rows = np.arange(xs.shape[0])
    design[rows, 2 * arms] = 1.0
    design[rows, 2 * arms + 1] = xs[:, 0]
    return design


def fit_linear_pevi(xs, yobs, wws, arm_count: int, ridge: float = 1.0):
    """Ridge fitted linear PEVI baseline from Jin et al. (2021)."""
    design = _linear_pevi_design(xs, wws, arm_count)
    gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
    theta = np.linalg.solve(gram, design.T @ np.asarray(yobs, dtype=float))
    inv_gram = np.linalg.pinv(gram)
    return theta, inv_gram


def eval_linear_pevi(model, eval_data, beta: float) -> tuple[np.ndarray, np.ndarray, float]:
    theta, inv_gram = model
    xs = np.asarray(eval_data["xs"], dtype=float)
    ys = np.asarray(eval_data["ys"], dtype=float)
    arm_count = ys.shape[1]
    scores = np.zeros((xs.shape[0], arm_count), dtype=float)
    uncertainty = np.zeros_like(scores)
    for arm in range(arm_count):
        feat = _linear_pevi_design(xs, np.full(xs.shape[0], arm, dtype=int), arm_count)
        scores[:, arm] = feat @ theta
        uncertainty[:, arm] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", feat, inv_gram, feat), 0.0))
    pred = np.argmax(scores - float(beta) * uncertainty, axis=1)
    rewards = ys[np.arange(ys.shape[0]), pred]
    return pred, rewards, float(np.mean(rewards))


def _run_contextual_task(task: dict) -> pd.DataFrame:
    cfg = ModeConfig(**task["cfg"])
    protocol = task["protocol"]
    seed = int(task["seed"])
    scenario = int(task["scenario"])
    T = int(task["T"])
    decay = task["decay"]
    decay_label = task["decay_label"]
    rep = int(task["rep"])
    dgp_class = PublishedMultiQuad if protocol == "published" else MultiQuad
    eval_data = dgp_class(2, 10, sigma=0).sample_data(
        cfg.t_eval,
        seed=_stable_seed(seed, protocol, "tree_eval", scenario, T, decay_label),
    )
    rows = []

    rep_seed = _stable_seed(seed, protocol, "tree", scenario, T, decay_label, rep)
    rep_rng = np.random.default_rng(rep_seed + 17)
    dgp = dgp_class(2, 10, sigma=0.1)
    data = dgp.sample_data(T, seed=rep_seed)
    xs, ys, muxs = data["xs"], data["ys"], data["muxs"]
    ps = _contextual_propensities(muxs, scenario, decay)
    ws = np.array([rep_rng.choice(10, p=ps[i]) for i in range(T)], dtype=int)
    yobs = ys[np.arange(T), ws]

    greedy = PL_greedy(xs, yobs, ws, ps, depth=5)
    _, _, rw_greedy = eval_ptree(greedy, eval_data)
    rows.append({"experiment": "tree", "scenario": scenario, "T": T, "decay": decay_label, "rep": rep, "method": "greedy", "beta": 0.0, "value": rw_greedy})
    lin_model = fit_linear_pevi(xs, yobs, ws, arm_count=10)
    initial = _published_initialization(
        protocol, greedy, xs, yobs, ws, ps
    )
    for beta in TREE_PESS_BETAS:
        tree, _ = _ppl_fit(
            protocol,
            xs,
            yobs,
            ws,
            ps,
            beta=beta,
            depth=5,
            **initial,
        )
        _, _, value = eval_ptree(tree, eval_data)
        rows.append({"experiment": "tree", "scenario": scenario, "T": T, "decay": decay_label, "rep": rep, "method": "pess", "beta": beta, "value": value})
    for beta in TREE_LINEAR_BETAS:
        _, _, lin_value = eval_linear_pevi(lin_model, eval_data, beta=beta)
        rows.append({"experiment": "tree", "scenario": scenario, "T": T, "decay": decay_label, "rep": rep, "method": "lin", "beta": beta, "value": lin_value})
    return pd.DataFrame(rows)


def run_contextual_nonadaptive(
    mode: str,
    protocol: str,
    seed: int = 0,
    jobs: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    cfg = mode_config(mode)
    outdir = outdir_for(mode, protocol)
    t_list = FULL_CONTEXTUAL_T_VALUES if mode == "full" else [300, 600]
    decays = [0.2, 0.4, 0.6, 0.8, None] if mode == "full" else [0.5, None]
    tasks = []
    for scenario in [1, 2, 3]:
        for T in t_list:
            for decay in decays:
                decay_label = "pure" if decay is None else str(decay)
                for rep in range(cfg.tree_nrep):
                    tasks.append(
                            {
                                "task_id": f"tree_s{scenario}_T{T}_d{decay_label}_r{rep:03d}",
                                "cfg": asdict(cfg),
                                "protocol": protocol,
                            "seed": seed,
                            "scenario": scenario,
                            "T": T,
                            "decay": decay,
                            "decay_label": decay_label,
                            "rep": rep,
                        }
                    )

    df = _run_chunked(
        tasks,
        _run_contextual_task,
        outdir / "data" / "chunks" / "contextual_nonadaptive",
        outdir / "data" / "contextual_nonadaptive_results.csv",
        jobs=jobs,
        resume=resume,
    )
    plot_value_grid(df, outdir / "figures" / "figure5_contextual_nonadaptive.png", row="scenario", col="decay")
    return df


def _dgp_for_ts(setting: int, protocol: str):
    linear_class = PublishedMultiLinear if protocol == "published" else MultiLinear
    quad_class = PublishedMultiQuad if protocol == "published" else MultiQuad
    if setting == 1:
        return linear_class(2, 10, sigma=0.1), linear_class(2, 10, sigma=0)
    return quad_class(2, 10, sigma=0.1), quad_class(2, 10, sigma=0)


def _ts_eval_seed(seed, protocol, setting, T, batch_size, floor_label):
    # The released fixed-beta and CV scripts reset to the same random stream.
    # Keep the historical False slot so this remains identical to fixed-beta
    # chunks generated before the shared-seed audit.
    return _stable_seed(
        seed,
        protocol,
        "ts_eval",
        setting,
        T,
        batch_size,
        floor_label,
        False,
    )


def _ts_rep_seed(seed, protocol, setting, T, batch_size, floor_label, rep):
    return _stable_seed(
        seed,
        protocol,
        "ts",
        setting,
        T,
        batch_size,
        floor_label,
        False,
        rep,
    )


def _run_ts_task(task: dict) -> pd.DataFrame:
    cfg = ModeConfig(**task["cfg"])
    protocol = task["protocol"]
    seed = int(task["seed"])
    setting = int(task["setting"])
    T = int(task["T"])
    batch_size = int(task["batch_size"])
    floor_label = task["floor_label"]
    floor_decay = task["floor_decay"]
    floor_start = task["floor_start"]
    cv_only = bool(task["cv_only"])
    rep = int(task["rep"])
    beta_default = [0.1, 0.2, 0.5, 1, 5, 10]
    beta_miss = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10]
    dgp, dgp_eval = _dgp_for_ts(setting, protocol)
    eval_data = dgp_eval.sample_data(
        cfg.t_eval,
        seed=_ts_eval_seed(seed, protocol, setting, T, batch_size, floor_label),
    )
    beta_list = beta_miss if setting == 3 else beta_default
    rows = []

    rep_seed = _ts_rep_seed(
        seed,
        protocol,
        setting,
        T,
        batch_size,
        floor_label,
        rep,
    )
    data = dgp.sample_data(T, seed=rep_seed)
    batch_sizes = [min(100, T)] + [batch_size] * int((T - min(100, T)) / batch_size)
    np.random.seed(rep_seed + 23)
    if setting == 2:
        logged = run_experiment_opt(
            data["xs"],
            data["ys"],
            "TS",
            dgp,
            batch_sizes=batch_sizes,
            num_mc=cfg.num_mc,
            if_floor=floor_decay is not None,
            floor_start=floor_start,
            floor_decay=floor_decay,
            ridge_mode="cv" if protocol == "published" else "fixed",
        )
    else:
        logged = run_experiment(
            data["xs"],
            data["ys"],
            "TS",
            dgp,
            batch_sizes=batch_sizes,
            num_mc=cfg.num_mc,
            if_floor=floor_decay is not None,
            floor_start=floor_start,
            floor_decay=floor_decay,
            ridge_mode="cv" if protocol == "published" else "fixed",
        )
    xs, yobs, ws, ps = logged["xs"], logged["yobs"], logged["ws"], logged["ps"]
    greedy = PL_greedy(xs, yobs, ws, ps, depth=5)
    _, _, rw_greedy = eval_ptree(greedy, eval_data)
    initial = _published_initialization(
        protocol, greedy, xs, yobs, ws, ps
    )
    experiment = "ts_cv" if cv_only else "ts"
    rows.append({"experiment": experiment, "setting": setting, "T": T, "batch_size": batch_size, "floor": floor_label, "rep": rep, "method": "greedy", "beta": 0.0, "value": rw_greedy})
    if cv_only:
        beta_cv, _, _, _ = _ppl_cv(protocol, xs, yobs, ws, ps, beta_list=beta_default, Nfold=5, depth=5)
        tree, _ = _ppl_fit(
            protocol,
            xs,
            yobs,
            ws,
            ps,
            beta=beta_cv,
            depth=5,
            **initial,
        )
        _, _, value = eval_ptree(tree, eval_data)
        rows.append({"experiment": "ts_cv", "setting": setting, "T": T, "batch_size": batch_size, "floor": floor_label, "rep": rep, "method": "CV_pess", "beta": beta_cv, "value": value})
    else:
        for beta in beta_list:
            tree, _ = _ppl_fit(
                protocol,
                xs,
                yobs,
                ws,
                ps,
                beta=beta,
                depth=5,
                **initial,
            )
            _, _, value = eval_ptree(tree, eval_data)
            rows.append({"experiment": "ts", "setting": setting, "T": T, "batch_size": batch_size, "floor": floor_label, "rep": rep, "method": "pess", "beta": beta, "value": value})
    return pd.DataFrame(rows)


def run_ts_synthetic(
    mode: str,
    protocol: str,
    seed: int = 0,
    cv_only: bool = False,
    jobs: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    cfg = mode_config(mode)
    outdir = outdir_for(mode, protocol)
    t_list = FULL_CONTEXTUAL_T_VALUES if mode == "full" else [300, 600]
    if protocol == "published":
        floor_settings = [("pure", 0.0, 0.001), ("0.2", 0.2, 0.1), ("0.5", 0.5, 0.1), ("0.8", 0.8, 0.1)]
    else:
        floor_settings = [("pure", None, None), ("0.2", 0.2, 0.1), ("0.5", 0.5, 0.1), ("0.8", 0.8, 0.1)]
    tasks = []
    settings = [2] if cv_only else [1, 2, 3]
    for setting in settings:
        for T in t_list:
            for batch_size in ([10] if cv_only else [10, 100]):
                for floor_label, floor_decay, floor_start in floor_settings:
                    prefix = "tscv" if cv_only else "ts"
                    for rep in range(cfg.ts_nrep):
                        tasks.append(
                            {
                                "task_id": f"{prefix}_s{setting}_T{T}_b{batch_size}_f{floor_label}_r{rep:03d}",
                                "cfg": asdict(cfg),
                                "protocol": protocol,
                                "seed": seed,
                                "setting": setting,
                                "T": T,
                                "batch_size": batch_size,
                                "floor_label": floor_label,
                                "floor_decay": floor_decay,
                                "floor_start": floor_start,
                                "cv_only": cv_only,
                                "rep": rep,
                            }
                        )

    name = "ts_cv_results.csv" if cv_only else "ts_synthetic_results.csv"
    df = _run_chunked(
        tasks,
        _run_ts_task,
        outdir / "data" / "chunks" / ("ts_cv" if cv_only else "ts_synthetic"),
        outdir / "data" / name,
        jobs=jobs,
        resume=resume,
    )
    if cv_only:
        plot_value_grid(df, outdir / "figures" / "figure9_ts_cv.png", row="floor", col="batch_size")
    else:
        figure_map = {
            1: "figure6_ts_well_specified.png",
            2: "figure7_ts_optimal_overlap.png",
            3: "figure8_ts_misspecified.png",
        }
        for setting, filename in figure_map.items():
            plot_value_grid(df[df["setting"] == setting], outdir / "figures" / filename, row="batch_size", col="floor")
    return df


def _load_openml_dataset(name: str):
    dataset = openml.datasets.get_dataset(name)
    return dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)


def _real_grid(mode: str, protocol: str):
    if mode == "full":
        beta_list = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15]
        if protocol == "published":
            settings = [("pure", 0.0, 0.001), ("0.8", 0.8, 0.5), ("0.5", 0.5, 0.5), ("0.2", 0.2, 0.5)]
        else:
            settings = [("pure", None, None), ("0.8", 0.8, 0.5), ("0.5", 0.5, 0.5), ("0.2", 0.2, 0.5)]
        datasets = REAL_DATASETS
        batch_values = [10, 100]
        max_train = None
        max_eval = None
    else:
        beta_list = [0.1, 1, 10]
        settings = [("pure", None, None), ("0.5", 0.5, 0.5)]
        datasets = ["cmc"]
        batch_values = [10]
        max_train = 600
        max_eval = 600
    return beta_list, settings, datasets, batch_values, max_train, max_eval


def _run_real_dataset_task(task: dict) -> pd.DataFrame:
    cfg = ModeConfig(**task["cfg"])
    mode = task["mode"]
    protocol = task["protocol"]
    seed = int(task["seed"])
    data_idx = int(task["dataset_index"])
    name = task["dataset"]
    beta_list, settings, _, batch_values, max_train, max_eval = _real_grid(mode, protocol)
    rows = []

    X, y, _, _ = _load_openml_dataset(name)
    for rep in range(cfg.real_nrep):
        try:
            bandit, _ = generate_bandit_data(
                X,
                y,
                noise_std=0.1,
                seed=seed + 1000 * rep + data_idx,
                max_arms=50,
            )
        except ValueError as exc:
            rows.append({"experiment": "real_skipped", "dataset_index": data_idx, "dataset": name, "reason": str(exc)})
            continue

        eval_limit = bandit["T_test"] if max_eval is None else min(max_eval, bandit["T_test"])
        train_limit = bandit["T"] if max_train is None else min(max_train, bandit["T"])
        eval_data = {
            "xs": bandit["xs_test"][:eval_limit],
            "ys": bandit["muxs_test"][:eval_limit],
            "muxs": bandit["muxs_test"][:eval_limit],
        }
        for batch_size in batch_values:
            explore = int(min(100, max(1, train_limit // 50)))
            usable = explore + batch_size * int((train_limit - explore) / batch_size)
            batch_sizes = [explore] + [batch_size] * int((usable - explore) / batch_size)
            for floor_label, floor_decay, floor_start in settings:
                np.random.seed(_stable_seed(seed, protocol, "real", data_idx, batch_size, floor_label, rep))
                logged = run_experiment(
                    bandit["xs"][:usable],
                    bandit["ys"][:usable],
                    "TS",
                    None,
                    batch_sizes=batch_sizes,
                    num_mc=cfg.num_mc,
                    if_floor=floor_decay is not None,
                    floor_start=floor_start,
                    floor_decay=floor_decay,
                    ridge_mode="cv" if protocol == "published" else "fixed",
                )
                xs, yobs, ws, ps = logged["xs"], logged["yobs"], logged["ws"], logged["ps"]
                greedy = PL_greedy(xs, yobs, ws, ps, depth=5)
                _, _, rw_greedy = eval_ptree(greedy, eval_data)
                initial = _published_initialization(
                    protocol, greedy, xs, yobs, ws, ps
                )
                beta_cv, _, _, _ = _ppl_cv(protocol, xs, yobs, ws, ps, beta_list=beta_list, Nfold=5, depth=5)
                tree, _ = _ppl_fit(
                    protocol,
                    xs,
                    yobs,
                    ws,
                    ps,
                    beta=beta_cv,
                    depth=5,
                    **initial,
                )
                _, _, rw_pess = eval_ptree(tree, eval_data)
                rows.append(
                    {
                        "experiment": "real",
                        "dataset_index": data_idx,
                        "dataset": name,
                        "batch_size": batch_size,
                        "floor": floor_label,
                        "rep": rep,
                        "greedy_value": rw_greedy,
                        "pess_value": rw_pess,
                        "improvement": rw_pess - rw_greedy,
                        "beta_cv": beta_cv,
                        "T_train": usable,
                        "T_eval": eval_limit,
                        "K": bandit["K"],
                        "p": bandit["p"],
                    }
                )
    return pd.DataFrame(rows)


def run_real(mode: str, protocol: str, seed: int = 0, jobs: int = 1, resume: bool = False) -> pd.DataFrame:
    cfg = mode_config(mode)
    outdir = outdir_for(mode, protocol)
    _, _, datasets, _, _, _ = _real_grid(mode, protocol)
    tasks = [
        {
            "task_id": f"real_d{idx}_{name.replace('/', '_')}",
            "cfg": asdict(cfg),
            "mode": mode,
            "protocol": protocol,
            "seed": seed,
            "dataset_index": idx,
            "dataset": name,
        }
        for idx, name in enumerate(datasets)
    ]

    combined = _run_chunked(
        tasks,
        _run_real_dataset_task,
        outdir / "data" / "chunks" / "real",
        None,
        jobs=jobs,
        resume=resume,
    )
    if combined.empty:
        df = pd.DataFrame()
        skipped = pd.DataFrame()
    else:
        df = combined[combined["experiment"] == "real"].copy()
        skipped = combined[combined["experiment"] == "real_skipped"].copy()
    df.to_csv(outdir / "data" / "real_results.csv", index=False)
    if not skipped.empty:
        skipped.drop_duplicates().to_csv(outdir / "data" / "real_skipped.csv", index=False)
    plot_real(df, outdir)
    return df


def plot_value_grid(df: pd.DataFrame, path: Path, row: str, col: str):
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["method_param"] = np.select(
        [
            plot_df["method"] == "greedy",
            plot_df["method"] == "CV_pess",
        ],
        [
            "greedy",
            "CV_pess",
        ],
        default=plot_df["method"] + "_" + plot_df["beta"].astype(str),
    )
    summary = plot_df.groupby([row, col, "T", "method_param"], as_index=False)["value"].mean()
    g = sns.relplot(data=summary, x="T", y="value", hue="method_param", row=row, col=col, kind="line", marker="o", height=2.2, aspect=1.0)
    g.figure.tight_layout()
    g.figure.savefig(path, dpi=200)
    plt.close(g.figure)


def plot_real(df: pd.DataFrame, outdir: Path):
    if df.empty:
        return
    g = sns.relplot(data=df, x="dataset_index", y="improvement", col="floor", row="batch_size", kind="scatter", height=2.6, aspect=1.2)
    for ax in g.axes.flat:
        ax.axhline(0, color="black", linewidth=0.8)
    g.set_axis_labels("Dataset index", "PPL - GPL value")
    g.figure.tight_layout()
    g.figure.savefig(outdir / "figures" / "figure10_real.png", dpi=200)
    plt.close(g.figure)


def _with_ci(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["se"]
    return summary


def _format_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def summarize(outdir: Path) -> Path:
    sections = []
    for csv_path in sorted((outdir / "data").glob("*.csv")):
        df = pd.read_csv(csv_path)
        sections.append(f"### {csv_path.name}\n\nrows: {len(df)}, columns: {', '.join(df.columns)}\n")
        if "value" in df.columns:
            group_cols = [c for c in ["experiment", "setting", "scenario", "method"] if c in df.columns]
            summary = df.groupby(group_cols)["value"].agg(["mean", "std", "count"]).reset_index()
            sections.append(_format_table(_with_ci(summary)))
        elif "rescaled_subopt" in df.columns:
            summary = df.groupby(["setting_name", "method"])["rescaled_subopt"].agg(["mean", "std", "count"]).reset_index()
            sections.append(_format_table(_with_ci(summary)))
        elif "improvement" in df.columns:
            summary = df.groupby(["floor", "batch_size"])["improvement"].agg(["mean", "std", "count"]).reset_index()
            summary = _with_ci(summary)
            pvals = []
            for _, group in df.groupby(["floor", "batch_size"]):
                spread = float(group["improvement"].std(ddof=1)) if len(group) > 1 else 0.0
                if len(group) > 1 and spread > 0:
                    pvals.append(float(stats.ttest_1samp(group["improvement"], 0.0, alternative="greater").pvalue))
                else:
                    pvals.append(np.nan)
            summary["p_value_gt0"] = pvals
            sections.append(_format_table(summary))
        sections.append("")

    report = outdir / "report_zh.md"
    report.write_text(
        "# 复现实验报告\n\n"
        "本报告由 `experiments/reproduce.py` 生成。`mean/std/count` 分别是重复实验均值、经验标准差和样本数；"
        "`ci95_low/ci95_high` 使用正态近似给出均值的 95% 置信区间。真实数据部分的 `p_value_gt0` "
        "是对 `PPL - GPL > 0` 的单侧 t 检验 p 值。快速模式样本数很小，只用于烟测；论文级结论应以 "
        "`--mode full` 的 N=200/N=1000 设置为准。\n\n"
        + "\n\n".join(sections),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Run paper reproduction experiments.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--protocol",
        choices=["published", "paper-spec"],
        default="published",
        help="Use figure-generating author behavior or the literal Section 6/7 specification.",
    )
    parser.add_argument("--experiment", choices=["mab", "tree", "ts", "ts-cv", "real", "all"], default="mab")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes for tree/TS/real chunked experiments.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed chunk CSVs and only run missing chunks.")
    args = parser.parse_args()

    outdir = outdir_for(args.mode, args.protocol)
    start = time.time()
    config_name = f"run_config_{args.experiment.replace('-', '_')}.json"
    (outdir / config_name).write_text(
        json.dumps(
            {
                "mode": args.mode,
                "protocol": args.protocol,
                "experiment": args.experiment,
                "seed": args.seed,
                "jobs": args.jobs,
                "resume": args.resume,
                "config": asdict(mode_config(args.mode)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.experiment in {"mab", "all"}:
        run_mab(args.mode, args.protocol, seed=args.seed)
    if args.experiment in {"tree", "all"}:
        run_contextual_nonadaptive(args.mode, args.protocol, seed=args.seed, jobs=args.jobs, resume=args.resume)
    if args.experiment in {"ts", "all"}:
        run_ts_synthetic(args.mode, args.protocol, seed=args.seed, jobs=args.jobs, resume=args.resume)
    if args.experiment in {"ts-cv", "all"}:
        run_ts_synthetic(args.mode, args.protocol, seed=args.seed, cv_only=True, jobs=args.jobs, resume=args.resume)
    if args.experiment in {"real", "all"}:
        run_real(args.mode, args.protocol, seed=args.seed, jobs=args.jobs, resume=args.resume)

    report = summarize(outdir)
    print(f"wrote {report}")
    print(f"elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
