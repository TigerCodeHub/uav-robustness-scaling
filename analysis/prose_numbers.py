#!/usr/bin/env python3
"""Numbers the paper quotes in its text rather than in a table.

    python analysis/prose_numbers.py RESULTS_DIR [OUT_TXT]

(A) Compounding, P1-clean at the four core widths: contrast = R_comb - R_lat * R_tilt,
    with R_x = mean perturbed / mean clean over the cell's policies. Negative means the
    combination costs more than the product of the single-perturbation robustnesses.
    95% CI and two-sided p from a policy bootstrap that keeps each policy's four
    evaluations together.
(B) Same-seed correlation: Pearson r across the 10 seeds between the policy-level
    success at adjacent widths, averaged over the adjacent pairs, per regime and
    condition (the reason policies are not paired by seed; the paper quotes the range of
    the headline-condition rows).
(C) Success/collision overlap: first episodes that both reached the goal and collided.
(D) Throughput: env-steps per second and wall clock to the 2M snapshot, median over
    policies (needs the *_d2000000_snap.json files written by train.py).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as P  # noqa: E402

LINES: list[str] = []


def out(s=""):
    LINES.append(s)
    print(s)


def compounding(root):
    out("(A) COMPOUNDING, six courses, P1-clean: contrast = R_comb - R_lat * R_tilt")
    B = 10000
    rng = np.random.default_rng(20261018)
    for w in P.CORE:
        d = {k: P.matrix(root, "r1", w, s).mean(axis=1)
             for k, s in (("clean", ""), ("lat", P.LAT), ("tilt", P.TILT), ("comb", P.COMB))}

        def contrast(i=None):
            g = (lambda v: v) if i is None else (lambda v: v[i])
            c = g(d["clean"]).mean()
            return g(d["comb"]).mean() / c - (g(d["lat"]).mean() / c) * (g(d["tilt"]).mean() / c)

        est = contrast()
        ts = np.array([contrast(rng.integers(0, 10, 10)) for _ in range(B)])
        lo, hi = np.quantile(ts, [0.025, 0.975])
        p = float((1 + (np.abs(ts - est) >= abs(est) - 1e-12).sum()) / (B + 1))
        c = d["clean"].mean()
        out(f"  w{w}: R_lat={d['lat'].mean()/c:.3f} R_tilt={d['tilt'].mean()/c:.3f} "
            f"product={d['lat'].mean()/c*d['tilt'].mean()/c:.3f} R_comb={d['comb'].mean()/c:.3f} "
            f"contrast={est:+.3f} 95%CI=({lo:+.3f},{hi:+.3f}) p_two={p:.4f}")


def seed_correlation(root):
    out("\n(B) SAME-SEED CORRELATION of policy-level success at adjacent widths")
    for reg in ("r1", "r2", "r3", "r4"):
        ws = sorted(P.WIDTHS[reg], key=float)
        for suf in (P.HEAD[reg], ""):
            v = [P.matrix(root, reg, w, suf).mean(axis=1) for w in ws]
            rs = [float(np.corrcoef(a, b)[0, 1]) for a, b in zip(v[:-1], v[1:])]
            out(f"  {P.NAME[reg]:9} {suf or 'clean':16} mean r={np.mean(rs):+.2f} "
                f"(pairs: {', '.join(f'{r:+.2f}' for r in rs)})")


def overlap(root):
    n = tot = 0
    for reg, w, s, c, suf in P.expected_cells():
        with open(P.summary_path(root, reg, c, w, s, suf)) as f:
            arm = json.load(f)["arms"]["ppo"]
        n += int(arm.get("n_success_collision_overlap", 0))
        tot += 1
    out(f"\n(C) success-and-collision on the same step: {n} episodes across {tot} evaluations")


def throughput(root):
    snaps = glob.glob(os.path.join(root, "r*", "*_d2000000_snap.json"))
    if not snaps:
        out("\n(D) throughput: no *_d2000000_snap.json files found")
        return
    sps, hours = [], []
    for p in snaps:
        with open(p) as f:
            j = json.load(f)
        sps.append(j["env_steps"] / j["wall_clock_s"])
        hours.append(j["wall_clock_s"] / 3600)
    out(f"\n(D) throughput over {len(snaps)} policies: median {np.median(sps):,.0f} env-steps/s, "
        f"median {np.median(hours):.2f} h to the 2M snapshot (incl. in-training evaluations)")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = sys.argv[1]
    P.check_complete(root)
    compounding(root)
    seed_correlation(root)
    overlap(root)
    throughput(root)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf8", newline="\n") as f:
            f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
