"""Sanity checks for the policy-level statistics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import stats as sc  # noqa: E402


def test_holm():
    assert sc.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_unpaired_compare_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.05, 10)
    r = sc.unpaired_compare(a, a + 0.2, n=2000)
    assert r["p_perm"] < 0.01 and r["ci95"][0] > 0.1
    r0 = sc.unpaired_compare(a, rng.permutation(a), n=2000)
    assert r0["p_perm"] > 0.5


def test_statistics_are_deterministic():
    a, b = np.linspace(0.3, 0.6, 10), np.linspace(0.4, 0.8, 10)
    assert sc.unpaired_compare(a, b, n=500) == sc.unpaired_compare(a, b, n=500)


def test_trend_slope_recovers_a_linear_trend():
    xs = np.log2([0.25, 0.5, 1.0, 2.0])
    groups = [np.full(10, 0.1 * x + 0.5) + np.linspace(-0.01, 0.01, 10) for x in xs]
    r = sc.trend_slope(xs, groups, n=500)
    assert abs(r["slope"] - 0.1) < 1e-9


def test_retention_is_a_ratio_of_means():
    c, p = np.array([0.8, 0.6, 1.0]), np.array([0.4, 0.3, 0.5])
    assert abs(sc.retention(c, p, n=200)["retention"] - 0.5) < 1e-12


def test_interaction_slope_of_a_constant_benefit_is_zero():
    xs = np.log2([0.25, 0.5, 1.0, 2.0])
    ga = [np.linspace(0.3, 0.5, 10) for _ in xs]
    gb = [g + 0.1 for g in ga]
    assert abs(sc.interaction_slope(xs, ga, gb, n=200)["slope"]) < 1e-9
