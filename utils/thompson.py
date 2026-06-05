from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from utils.compute import apply_floor


class LinTS:
    """Linear Thompson Sampling with Monte Carlo propensity estimates."""

    def __init__(self, K, p, DGP=None, num_mc=1000, prior_scale=10.0):
        self.K = int(K)
        self.p = int(p)
        self.num_mc = int(num_mc)
        self.DGP = DGP
        self.prior_scale = float(prior_scale)
        self.mu = [np.zeros(self.p + 1, dtype=float) for _ in range(self.K)]
        self.V = [np.eye(self.p + 1, dtype=float) * self.prior_scale for _ in range(self.K)]
        self.X = [[] for _ in range(self.K)]
        self.y = [[] for _ in range(self.K)]
        self.ps = None
        self.ws = None
        self.if_floor = False
        self.floor_decay = None
        self.floor_total = None

    def initialize_ps(self, T):
        self.ps = np.zeros((int(T), self.K), dtype=float)

    def initialize_w(self, T):
        self.ws = np.zeros(int(T), dtype=int)

    def set_floor(self, if_floor=False, floor_start=None, floor_decay=None, floor_total=None):
        self.if_floor = bool(if_floor)
        self.floor_decay = floor_decay
        self.floor_total = floor_total

    def _feature_with_intercept(self, x):
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones(x.shape[0]), x])

    def _update_arm(self, x, y, arm):
        if len(x) == 0:
            return
        self.X[arm].extend(np.asarray(x, dtype=float).tolist())
        self.y[arm].extend(np.asarray(y, dtype=float).tolist())
        X_arm = np.asarray(self.X[arm], dtype=float)
        y_arm = np.asarray(self.y[arm], dtype=float)

        if len(y_arm) < 2:
            self.mu[arm] = np.r_[float(np.mean(y_arm)), np.zeros(self.p)]
            self.V[arm] = np.eye(self.p + 1) * self.prior_scale
            return

        model = Ridge(alpha=1.0, fit_intercept=True).fit(X_arm, y_arm)
        self.mu[arm] = np.r_[model.intercept_, model.coef_]
        X_design = self._feature_with_intercept(X_arm)
        residual = y_arm - model.predict(X_arm)
        sigma2 = float(np.var(residual)) if len(residual) > 1 else 1.0
        gram = X_design.T @ X_design + np.eye(self.p + 1)
        self.V[arm] = sigma2 * np.linalg.pinv(gram) + 1e-8 * np.eye(self.p + 1)

    def update_TS(self, xss, wss, yss):
        wss = np.asarray(wss, dtype=int)
        for arm in range(self.K):
            idx = wss == arm
            self._update_arm(np.asarray(xss)[idx], np.asarray(yss)[idx], arm)

    def _floor_vector(self, current_T, add_floor=None, row=None):
        floor = np.zeros(self.K, dtype=float)
        if self.if_floor and self.floor_decay is not None:
            total = float(self.floor_total if self.floor_total is not None else current_T)
            floor[:] = total ** (-float(self.floor_decay)) / self.K
        if add_floor is not None and row is not None:
            floor = np.maximum(floor, np.asarray(add_floor[row], dtype=float))
        return floor

    def draw_TS_one_batch(self, xs, start, end, current_t=None, add_floor=None):
        xs = np.asarray(xs, dtype=float)
        if self.ps is None:
            self.initialize_ps(xs.shape[0])
        if self.ws is None:
            self.initialize_w(xs.shape[0])

        x_design = self._feature_with_intercept(xs)
        coeff = np.empty((self.K, self.num_mc, self.p + 1), dtype=float)
        for arm in range(self.K):
            cov = (self.V[arm] + self.V[arm].T) / 2.0
            coeff[arm] = np.random.multivariate_normal(self.mu[arm], cov, size=self.num_mc)
        draws = np.matmul(coeff, x_design.T)

        for row in range(int(start), int(end)):
            raw = np.bincount(np.argmax(draws[:, :, row], axis=0), minlength=self.K) / self.num_mc
            floor = self._floor_vector(xs.shape[0], add_floor=add_floor, row=row)
            self.ps[row, :] = apply_floor(raw, floor) if np.any(floor > 0) else raw
            if self.ps[row, :].sum() <= 0:
                self.ps[row, :] = np.ones(self.K) / self.K

        sampled = [np.random.choice(self.K, p=self.ps[row]) for row in range(int(start), int(end))]
        self.ws[np.arange(int(start), int(end))] = sampled
        return sampled, self.ps
