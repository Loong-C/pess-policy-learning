from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Optional

import numpy as np

from utils.thompson import LinTS


def _initial_actions(size, K):
    return np.asarray(np.arange(size) % K, dtype=int)


def run_experiment(
    xs,
    ys,
    bandit_model="TS",
    dgp_model=None,
    batch_sizes=None,
    num_mc=1000,
    record_idx=None,
    if_floor=False,
    floor_start=None,
    floor_decay=None,
):
    """Collect adaptive data with batched linear Thompson Sampling."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    T, K = ys.shape
    _, p = xs.shape
    if batch_sizes is None:
        batch_sizes = [min(100, T)] + [10] * max(0, (T - min(100, T)) // 10)
    if np.sum(batch_sizes) > T:
        raise ValueError("Sum of batch sizes exceeds available data.")
    T = int(np.sum(batch_sizes))
    xs = xs[:T]
    ys = ys[:T]

    if record_idx is None:
        record_idx = [len(batch_sizes) - 1]

    if bandit_model != "TS":
        raise ValueError("Only bandit_model='TS' is implemented.")

    agent = LinTS(K, p, dgp_model, num_mc=num_mc)
    agent.initialize_ps(T)
    agent.initialize_w(T)
    agent.set_floor(if_floor=if_floor, floor_decay=floor_decay, floor_total=T)

    batch_ends = np.cumsum(batch_sizes)
    initial_end = int(batch_ends[0])
    ws = np.zeros(T, dtype=int)
    yobs = np.zeros(T, dtype=float)
    ws[:initial_end] = _initial_actions(initial_end, K)
    yobs[:initial_end] = ys[np.arange(initial_end), ws[:initial_end]]
    agent.ws[:initial_end] = ws[:initial_end]
    agent.ps[:initial_end, :] = 1.0 / K
    agent.update_TS(xs[:initial_end], ws[:initial_end], yobs[:initial_end])

    record_agents = []
    for idx, (st, ed) in enumerate(zip(batch_ends[:-1], batch_ends[1:]), 1):
        st, ed = int(st), int(ed)
        sampled, _ = agent.draw_TS_one_batch(xs, st, ed, current_t=st)
        sampled = np.asarray(sampled, dtype=int)
        yobs[st:ed] = ys[np.arange(st, ed), sampled]
        agent.update_TS(xs[st:ed], sampled, yobs[st:ed])
        if idx in record_idx:
            record_agents.append(deepcopy(agent))

    return {"agents": record_agents, "yobs": yobs, "ws": agent.ws.copy(), "xs": xs, "ys": ys, "ps": agent.ps.copy()}


def run_experiment_opt(
    xs,
    ys,
    bandit_model="TS",
    dgp_model=None,
    batch_sizes=None,
    num_mc=1000,
    record_idx=None,
    add_floor_const=0.1,
    if_floor=False,
    floor_start=None,
    floor_decay=None,
):
    """TS collection with an additional 0.1 floor on the true optimal arm."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    T, K = ys.shape
    if batch_sizes is None:
        batch_sizes = [min(100, T)] + [10] * max(0, (T - min(100, T)) // 10)
    T_used = int(np.sum(batch_sizes))
    opt_w = np.asarray(dgp_model.compute_optimal(xs[:T_used]), dtype=int)
    add_floor = np.zeros((T_used, K), dtype=float)
    add_floor[np.arange(T_used), opt_w] = float(add_floor_const)

    xs_used = xs[:T_used]
    ys_used = ys[:T_used]
    _, p = xs_used.shape

    agent = LinTS(K, p, dgp_model, num_mc=num_mc)
    agent.initialize_ps(T_used)
    agent.initialize_w(T_used)
    agent.set_floor(if_floor=if_floor, floor_decay=floor_decay, floor_total=T_used)

    batch_ends = np.cumsum(batch_sizes)
    initial_end = int(batch_ends[0])
    ws = np.zeros(T_used, dtype=int)
    yobs = np.zeros(T_used, dtype=float)
    ws[:initial_end] = _initial_actions(initial_end, K)
    yobs[:initial_end] = ys_used[np.arange(initial_end), ws[:initial_end]]
    agent.ws[:initial_end] = ws[:initial_end]
    agent.ps[:initial_end, :] = 1.0 / K
    agent.update_TS(xs_used[:initial_end], ws[:initial_end], yobs[:initial_end])

    if record_idx is None:
        record_idx = [len(batch_sizes) - 1]
    record_agents = []
    for idx, (st, ed) in enumerate(zip(batch_ends[:-1], batch_ends[1:]), 1):
        st, ed = int(st), int(ed)
        sampled, _ = agent.draw_TS_one_batch(xs_used, st, ed, current_t=st, add_floor=add_floor)
        sampled = np.asarray(sampled, dtype=int)
        yobs[st:ed] = ys_used[np.arange(st, ed), sampled]
        agent.update_TS(xs_used[st:ed], sampled, yobs[st:ed])
        if idx in record_idx:
            record_agents.append(deepcopy(agent))

    return {"agents": record_agents, "yobs": yobs, "ws": agent.ws.copy(), "xs": xs_used, "ys": ys_used, "ps": agent.ps.copy()}
