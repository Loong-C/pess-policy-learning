from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _rng(seed=None):
    return np.random.default_rng(seed)


class MultiQuad:
    """Nonlinear contextual bandit from Sections 7.1.2, 7.2.2, and 7.2.3.

    mu(x, a_k) = 1 - alpha_k / 2 + alpha_k * x_1^2 / 2.
    """

    def __init__(self, p, K=2, sigma=0.1, alpha=None):
        self.p = int(p)
        self.K = int(K)
        self.sigma = float(sigma)
        self.alpha = np.linspace(-1, 1, num=self.K) if alpha is None else np.asarray(alpha, dtype=float)

    def mean(self, xs):
        xs = np.asarray(xs, dtype=float)
        muxs = np.zeros((xs.shape[0], self.K), dtype=float)
        for arm in range(self.K):
            muxs[:, arm] = 1.0 - self.alpha[arm] / 2.0 + self.alpha[arm] * xs[:, 0] ** 2 / 2.0
        return muxs

    def sample_data(self, n, seed=None):
        rng = _rng(seed)
        xs = rng.uniform(-2, 2, size=(int(n), self.p))
        muxs = self.mean(xs)
        ys = muxs + rng.normal(0, self.sigma, size=muxs.shape)
        return {"xs": xs, "ys": ys, "muxs": muxs}

    def compute_optimal(self, xs):
        return np.argmax(self.mean(xs), axis=1)


class TwoQuad(MultiQuad):
    def __init__(self, p, sigma=0.1, alpha=None):
        super().__init__(p=p, K=2, sigma=sigma, alpha=np.array([-1, 1]) if alpha is None else alpha)


class MultiLinear:
    """Well-specified TS setting from Section 7.2.1.

    mu(x, a_k) = 1 - alpha_k / 2 + x_1 / 2 - x_2.
    """

    def __init__(self, p, K=2, sigma=0.1, alpha=None):
        if p < 2:
            raise ValueError("MultiLinear requires p >= 2.")
        self.p = int(p)
        self.K = int(K)
        self.sigma = float(sigma)
        self.alpha = np.linspace(-1, 1, num=self.K) if alpha is None else np.asarray(alpha, dtype=float)

    def mean(self, xs):
        xs = np.asarray(xs, dtype=float)
        base = xs[:, 0] / 2.0 - xs[:, 1]
        muxs = np.zeros((xs.shape[0], self.K), dtype=float)
        for arm in range(self.K):
            muxs[:, arm] = 1.0 - self.alpha[arm] / 2.0 + base
        return muxs

    def sample_data(self, n, seed=None):
        rng = _rng(seed)
        xs = rng.uniform(-2, 2, size=(int(n), self.p))
        muxs = self.mean(xs)
        ys = muxs + rng.normal(0, self.sigma, size=muxs.shape)
        return {"xs": xs, "ys": ys, "muxs": muxs}

    def compute_optimal(self, xs):
        return np.argmax(self.mean(xs), axis=1)


class PublishedMultiQuad(MultiQuad):
    """Nonlinear reward model used by the authors' released experiment code."""

    def mean(self, xs):
        xs = np.asarray(xs, dtype=float)
        muxs = np.zeros((xs.shape[0], self.K), dtype=float)
        for arm in range(self.K):
            muxs[:, arm] = 1.0 - self.alpha[arm] + self.alpha[arm] * xs[:, 0] ** 2
        return muxs


class PublishedMultiLinear(MultiLinear):
    """Well-specified reward model underlying the published Figure 6 values."""

    def mean(self, xs):
        xs = np.asarray(xs, dtype=float)
        signal = xs[:, 0] / 2.0 - xs[:, 1]
        muxs = np.zeros((xs.shape[0], self.K), dtype=float)
        for arm in range(self.K):
            muxs[:, arm] = 1.0 - self.alpha[arm] / 2.0 + self.alpha[arm] * signal
        return muxs


class MABandit:
    """No-covariate five-arm bandit used in Section 7.1.1."""

    def __init__(self, mu=None, sigma=0.1, ps=None):
        self.sigma = float(sigma)
        self.mu = np.linspace(-1, 1, num=5) if mu is None else np.asarray(mu, dtype=float)
        self.ps = None if ps is None else np.asarray(ps, dtype=float)

    def set_ps(self, ps):
        self.ps = np.asarray(ps, dtype=float)

    def sample_data(self, n, seed=None):
        rng = _rng(seed)
        ys = self.mu.reshape(1, -1) + rng.normal(0, self.sigma, size=(int(n), len(self.mu)))
        return {"ys": ys}

    def sample_arms(self, n, seed=None):
        if self.ps is None:
            raise ValueError("Sampling probabilities `ps` must be set first.")
        return _rng(seed).choice(len(self.mu), size=int(n), p=self.ps)


def _preprocess_openml_features(X_train, X_eval):
    train_df = pd.DataFrame(X_train)
    eval_df = pd.DataFrame(X_eval)
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == "object" or str(train_df[c].dtype) == "category"]
    num_cols = [c for c in train_df.columns if c not in cat_cols]

    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot)]), cat_cols),
        ],
        sparse_threshold=0,
    )
    x_train = transformer.fit_transform(train_df)
    x_eval = transformer.transform(eval_df)
    return np.asarray(x_train, dtype=float), np.asarray(x_eval, dtype=float)


def generate_bandit_data(X, y, noise_std=0.1, test_size=0.5, seed=0, max_arms: int | None = None):
    """Classification-to-bandit transform used in Section 7.3."""
    classes = np.unique(y)
    if max_arms is not None and len(classes) > int(max_arms):
        raise ValueError(f"Expected a classification target with at most {max_arms} classes, got {len(classes)}.")

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    xs, xs_eval = _preprocess_openml_features(X_train, X_eval)

    label_to_arm = {label: idx for idx, label in enumerate(classes)}
    train_labels = np.array([label_to_arm[v] for v in y_train], dtype=int)
    eval_labels = np.array([label_to_arm[v] for v in y_eval], dtype=int)

    K = len(classes)
    muxs = np.eye(K, dtype=float)[train_labels]
    muxs_eval = np.eye(K, dtype=float)[eval_labels]
    rng = _rng(seed)
    ys = muxs + rng.normal(0, noise_std, size=muxs.shape)

    data = {
        "xs": xs,
        "ys": ys,
        "muxs": muxs,
        "xs_test": xs_eval,
        "ys_test": muxs_eval,
        "muxs_test": muxs_eval,
        "K": K,
        "p": xs.shape[1],
        "T": xs.shape[0],
        "T_test": xs_eval.shape[0],
        "classes": classes,
    }
    return data, muxs
