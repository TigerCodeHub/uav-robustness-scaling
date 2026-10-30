"""Policy-level statistics for the (policy x fixed course) design.

The unit of analysis is the trained policy, scored by its success averaged over the six
fixed courses. Policies are independent replicates (not paired by seed). Each function
seeds its own generator (default 0), so every number is reproducible run to run.
"""

from __future__ import annotations

import numpy as np


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def unpaired_compare(a, b, n=20000, rng=None, alpha=0.05):
    """mean(b) - mean(a) for independent policy-level vectors: two-sided label-permutation
    p and percentile bootstrap CI (policies resampled within each group)."""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = b.mean() - a.mean()
    pool, na = np.concatenate([a, b]), a.size
    perm = np.empty(n)
    for k in range(n):
        x = rng.permutation(pool)
        perm[k] = x[na:].mean() - x[:na].mean()
    p = float((1 + (np.abs(perm) >= abs(obs) - 1e-12).sum()) / (n + 1))
    bs = np.array([rng.choice(b, b.size).mean() - rng.choice(a, a.size).mean() for _ in range(n)])
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(obs), "p_perm": p, "ci95": (float(lo), float(hi))}


def _slope(xs, groups):
    return float(np.polyfit(xs, [np.mean(g) for g in groups], 1)[0])


def trend_slope(xs, groups, n=10000, rng=None):
    """Least-squares slope of group means on xs (log2 width). CI: policies resampled within
    each group; p: group labels permuted across all policies."""
    rng = rng or np.random.default_rng(0)
    groups = [np.asarray(g, float) for g in groups]
    obs = _slope(xs, groups)
    bs = np.array([_slope(xs, [rng.choice(g, g.size) for g in groups]) for _ in range(n)])
    pool, cuts = np.concatenate(groups), np.cumsum([g.size for g in groups])[:-1]
    perm = np.array([_slope(xs, np.split(rng.permutation(pool), cuts)) for _ in range(n)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = float((1 + (np.abs(perm) >= abs(obs) - 1e-12).sum()) / (n + 1))
    return {"slope": obs, "ci95": (float(lo), float(hi)), "p_perm": p}


def interaction_slope(xs, ga, gb, n=10000, rng=None):
    """Slope over xs of the training benefit mean(gb_i) - mean(ga_i).

    p_boot tests slope = 0 with the bootstrap distribution centred on the estimate (the
    paper's test). p_perm shuffles the a/b label within each width, which tests the
    different null of no benefit at any width, and is reported for reference only."""
    rng = rng or np.random.default_rng(0)
    ga = [np.asarray(g, float) for g in ga]
    gb = [np.asarray(g, float) for g in gb]

    def s(A, B):
        return float(np.polyfit(xs, [B[i].mean() - A[i].mean() for i in range(len(xs))], 1)[0])
    obs = s(ga, gb)
    perm = np.empty(n)
    for k in range(n):
        A, B = [], []
        for x, y in zip(ga, gb):
            z = rng.permutation(np.concatenate([x, y]))
            A.append(z[:x.size]); B.append(z[x.size:])
        perm[k] = s(A, B)
    bs = np.array([s([rng.choice(g, g.size) for g in ga], [rng.choice(g, g.size) for g in gb])
                   for _ in range(n)])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = float((1 + (np.abs(perm) >= abs(obs) - 1e-12).sum()) / (n + 1))
    p_boot = float((1 + (np.abs(bs - obs) >= abs(obs) - 1e-12).sum()) / (n + 1))
    return {"slope": obs, "ci95": (float(lo), float(hi)), "p_perm": p, "p_boot": p_boot}


def retention(clean, pert, n=10000, rng=None):
    """Robustness = mean perturbed / mean clean over a cell's policies (ratio of cell
    means), with a policy bootstrap that keeps each policy's pair together."""
    rng = rng or np.random.default_rng(0)
    c, p = np.asarray(clean, float), np.asarray(pert, float)
    obs = p.mean() / c.mean()
    bs = np.empty(n)
    for k in range(n):
        i = rng.integers(0, c.size, c.size)
        bs[k] = p[i].mean() / c[i].mean()
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"retention": float(obs), "ci95": (float(lo), float(hi))}


def retention_trend(xs, cleans, perts, n=10000, rng=None):
    """Slope of cell robustness on xs. CI: policy bootstrap within width (pairs kept);
    p: width labels permuted across policies (each policy keeps its pair)."""
    rng = rng or np.random.default_rng(0)
    cleans = [np.asarray(c, float) for c in cleans]
    perts = [np.asarray(p, float) for p in perts]

    def s(C, Pp):
        return float(np.polyfit(xs, [p.mean() / c.mean() for c, p in zip(C, Pp)], 1)[0])
    obs = s(cleans, perts)
    bs = np.empty(n)
    for k in range(n):
        C, Pp = [], []
        for c, p in zip(cleans, perts):
            i = rng.integers(0, c.size, c.size)
            C.append(c[i]); Pp.append(p[i])
        bs[k] = s(C, Pp)
    allc, allp = np.concatenate(cleans), np.concatenate(perts)
    cuts = np.cumsum([c.size for c in cleans])[:-1]
    perm = np.empty(n)
    for k in range(n):
        j = rng.permutation(allc.size)
        perm[k] = s(np.split(allc[j], cuts), np.split(allp[j], cuts))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = float((1 + (np.abs(perm) >= abs(obs) - 1e-12).sum()) / (n + 1))
    return {"slope": obs, "ci95": (float(lo), float(hi)), "p_perm": p}
