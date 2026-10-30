#!/usr/bin/env python3
"""Do the six-course effects hold on the ten held-out courses (9200-9209)?

    python analysis/heldout.py RESULTS_DIR [OUT_DIR]      (OUT_DIR default: outputs/tables)

The held-out courses were not used while the analysis was developed and are never pooled
with the six analysis courses. Every effect is re-estimated on them with the same
estimators as the main analysis (policy = success averaged over the courses, policies
independent, courses fixed), next to its six-course estimate, with the number of
held-out courses on which the per-course effect has the same sign.

Writes:
    tab_heldout_effects.tex  each effect: six-course estimate, held-out estimate, 95% CI, same sign
    tab_heldout_levels.tex   clean, perturbed, robustness (95% CI), boundary-breach rate per
                             regime under its own condition
    tab_heldout_p1x.tex      P1-clean under tilt and the combined condition
    heldout.txt              the same numbers in plain text, with p-values
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as P   # noqa: E402
import stats as sc  # noqa: E402

COND = {"lat": (P.LAT, "latency"), "tilt": (P.TILT, "tilt"), "comb": (P.COMB, "combined")}
TRAINED = {"lat": "r2", "tilt": "r3", "comb": "r4"}
WID = {"lat": P.CORE, "tilt": P.CORE, "comb": P.ALLW}
SERIES = [("lat", "r1", P.ALLW), ("tilt", "r1", P.CORE), ("comb", "r1", P.ALLW),
          ("lat", "r2", P.CORE), ("tilt", "r3", P.CORE), ("comb", "r4", P.ALLW)]
LINES: list[str] = []


def out(s=""):
    LINES.append(s)
    print(s)


def _p(pa):
    return f"{pa:.3f}" if pa >= 0.001 else "$<$0.001"


def ws_of(widths):
    return sorted(widths, key=float)


class Courses:
    """Policy-level and per-course views of one course set."""

    def __init__(self, root, courses):
        self.root, self.courses = root, courses

    def mat(self, reg, w, suf):
        return P.matrix(self.root, reg, w, suf, self.courses)

    def pols(self, reg, w, suf):
        return P.policies(self.root, reg, w, suf, self.courses)

    def course_means(self, reg, w, suf):
        """(n_courses,) mean success over the cell's policies, per course."""
        return self.mat(reg, w, suf).mean(axis=0)

    def breach(self, reg, w, suf):
        v = []
        for s in P.SEEDS:
            for c in self.courses:
                with open(P.outcomes_path(self.root, reg, c, w, s, suf)) as f:
                    v.append(float(np.mean([json.loads(line)["lateral_breach"] for line in f])))
        return float(np.mean(v))


def per_course_slope(xs, cells):
    """cells: list over widths of (n_courses,) -> (n_courses,) slopes on xs."""
    Y = np.stack(cells)                                  # (widths, courses)
    return np.array([np.polyfit(xs, Y[:, j], 1)[0] for j in range(Y.shape[1])])


def compounding(C, w, rng=None, n=10000):
    """R_comb - R_lat * R_tilt for P1-clean (ratio of cell means), policy bootstrap keeping each
    policy's four evaluations together; plus the per-course contrast."""
    rng = rng or np.random.default_rng(0)
    d = {k: C.pols("r1", w, s) for k, s in (("clean", ""), ("lat", P.LAT), ("tilt", P.TILT),
                                            ("comb", P.COMB))}

    def contrast(i=None):
        g = (lambda v: v) if i is None else (lambda v: v[i])
        c = g(d["clean"]).mean()
        return g(d["comb"]).mean() / c - (g(d["lat"]).mean() / c) * (g(d["tilt"]).mean() / c)

    est = contrast()
    k = d["clean"].size
    ts = np.array([contrast(rng.integers(0, k, k)) for _ in range(n)])
    lo, hi = np.quantile(ts, [0.025, 0.975])
    m = {k_: C.course_means("r1", w, s) for k_, s in (("clean", ""), ("lat", P.LAT),
                                                       ("tilt", P.TILT), ("comb", P.COMB))}
    pc = m["comb"] / m["clean"] - (m["lat"] / m["clean"]) * (m["tilt"] / m["clean"])
    return {"est": float(est), "ci95": (float(lo), float(hi))}, pc


def effects(six, held):
    """Rows: (group, label, six-course estimate, held-out result dict, per-course values or None)."""
    rows = []
    for cond, reg, W in SERIES:
        suf, xs = COND[cond][0], np.log2([float(w) for w in ws_of(W)])
        r6 = sc.trend_slope(xs, [six.pols(reg, w, suf) for w in ws_of(W)])
        rh = sc.trend_slope(xs, [held.pols(reg, w, suf) for w in ws_of(W)])
        pc = per_course_slope(xs, [held.course_means(reg, w, suf) for w in ws_of(W)])
        rows.append(("Size trend of perturbed success (per doubling)",
                     f"{P.NAME[reg]} / {COND[cond][1]}", r6["slope"],
                     {"est": rh["slope"], "ci95": rh["ci95"], "p": rh["p_perm"]}, pc))
    for cond, reg, W in SERIES:
        if reg == "r3":
            continue    # P3-tilt's w=0.25 robustness is not interpretable
        suf, xs = COND[cond][0], np.log2([float(w) for w in ws_of(W)])
        r6 = sc.retention_trend(xs, [six.pols(reg, w, "") for w in ws_of(W)],
                                [six.pols(reg, w, suf) for w in ws_of(W)])
        rh = sc.retention_trend(xs, [held.pols(reg, w, "") for w in ws_of(W)],
                                [held.pols(reg, w, suf) for w in ws_of(W)])
        pc = per_course_slope(xs, [held.course_means(reg, w, suf) / held.course_means(reg, w, "")
                                   for w in ws_of(W)])
        rows.append(("Size trend of robustness (per doubling)", f"{P.NAME[reg]} / {COND[cond][1]}",
                     r6["slope"], {"est": rh["slope"], "ci95": rh["ci95"], "p": rh["p_perm"]}, pc))
    for cond in ("lat", "tilt", "comb"):
        suf = COND[cond][0]
        for w in ("0.25", "2.0"):
            r6 = sc.unpaired_compare(six.pols("r1", w, suf), six.pols(TRAINED[cond], w, suf))
            rh = sc.unpaired_compare(held.pols("r1", w, suf), held.pols(TRAINED[cond], w, suf))
            pc = held.course_means(TRAINED[cond], w, suf) - held.course_means("r1", w, suf)
            rows.append(("Training benefit (trained $-$ P1-clean)",
                         f"{P.NAME[TRAINED[cond]]}, {COND[cond][1]}, $w={w}$", r6["mean_diff"],
                         {"est": rh["mean_diff"], "ci95": rh["ci95"], "p": rh["p_perm"]}, pc))
    for cond in ("lat", "tilt", "comb"):
        suf, W = COND[cond][0], ws_of(WID[cond])
        xs = np.log2([float(w) for w in W])
        r6 = sc.interaction_slope(xs, [six.pols("r1", w, suf) for w in W],
                                  [six.pols(TRAINED[cond], w, suf) for w in W])
        rh = sc.interaction_slope(xs, [held.pols("r1", w, suf) for w in W],
                                  [held.pols(TRAINED[cond], w, suf) for w in W])
        pc = per_course_slope(xs, [held.course_means(TRAINED[cond], w, suf) - held.course_means("r1", w, suf)
                                   for w in W])
        rows.append(("Size trend of training benefit (per doubling)",
                     f"{P.NAME[TRAINED[cond]]}, {COND[cond][1]}", r6["slope"],
                     {"est": rh["slope"], "ci95": rh["ci95"], "p": rh["p_boot"]}, pc))
    for w in P.CORE:
        r6, _ = compounding(six, w)
        rh, pc = compounding(held, w)
        rows.append(("Compounding: $R_{\\text{comb}}-R_{\\text{lat}}R_{\\text{tilt}}$, P1-clean",
                     f"$w={w}$", r6["est"], {**rh, "p": None}, pc))
    return rows


def plain(s):
    """LaTeX label -> plain text."""
    s = s.replace("\\text", "")
    return s.replace("$", "").replace("{", "").replace("}", "")


def same_sign(ref, pc):
    return f"{int(np.sum(np.sign(pc) == np.sign(ref)))}/{pc.size}"


def tab_effects(rows, write):
    lines, group = [], None
    for g, label, e6, rh, pc in rows:
        if g != group:
            if group is not None:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{g}}}}} \\\\")
            group = g
        lo, hi = rh["ci95"]
        lines.append(f"{label} & ${e6:+.3f}$ & ${rh['est']:+.3f}$ & $({lo:+.3f},{hi:+.3f})$ & "
                     f"{same_sign(e6, pc)} \\\\")
    write("tab_heldout_effects.tex", (
        "\\begin{tabular}{@{}l r r c c@{}}\n\\toprule\n"
        "Effect & six courses & held-out & 95\\% CI (held-out) & same sign \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))


def tab_levels(held, write):
    names = {"r1": "P1-clean under latency", "r2": "P2-lat under latency",
             "r3": "P3-tilt under tilt", "r4": "P4-comb under combined"}
    lines = []
    for reg in ("r1", "r2", "r3", "r4"):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{names[reg]}}}}} \\\\")
        for w in ws_of(P.WIDTHS[reg]):
            c, p = held.pols(reg, w, ""), held.pols(reg, w, P.HEAD[reg])
            br = held.breach(reg, w, P.HEAD[reg])
            if reg == "r3" and w == "0.25":
                rob = "n/i$^\\dagger$ & --"
            else:
                r = sc.retention(c, p)
                rob = f"{r['retention']:.2f} & $({r['ci95'][0]:.2f},{r['ci95'][1]:.2f})$"
            lines.append(f"{w} & {c.mean():.3f} & {p.mean():.3f} & {rob} & {100 * br:.1f} \\\\")
        lines.append("\\midrule")
    write("tab_heldout_levels.tex", (
        "\\begin{tabular}{@{}r c c c c r@{}}\n\\toprule\n"
        "$w$ & clean & perturbed & robustness & 95\\% CI & breach (\\%) \\\\\n\\midrule\n"
        + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n"))


def tab_p1x(held, write):
    lines = []
    for cond, label, W in (("tilt", "P1-clean under tilt", P.CORE),
                           ("comb", "P1-clean under combined", P.ALLW)):
        if lines:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{label}}}}} \\\\")
        for w in ws_of(W):
            c, p = held.pols("r1", w, ""), held.pols("r1", w, COND[cond][0])
            r = sc.retention(c, p)
            lines.append(f"{w} & {c.mean():.3f} & {p.mean():.3f} & {r['retention']:.2f} & "
                         f"$({r['ci95'][0]:.2f},{r['ci95'][1]:.2f})$ \\\\")
    write("tab_heldout_p1x.tex", (
        "\\begin{tabular}{@{}r c c c c@{}}\n\\toprule\n"
        "$w$ & clean & perturbed & robustness & 95\\% CI \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", "tables")
    os.makedirs(outdir, exist_ok=True)

    def write(name, text):
        with open(os.path.join(outdir, name), "w", encoding="utf8", newline="\n") as f:
            f.write(text)

    P.check_complete(root, P.ALL6)
    print(f"completeness: {P.check_complete(root, P.HELDOUT)} held-out evaluations present")
    six, held = Courses(root, P.ALL6), Courses(root, P.HELDOUT)
    rows = effects(six, held)
    tab_effects(rows, write)
    tab_levels(held, write)
    tab_p1x(held, write)

    out("HELD-OUT COURSES 9200-9209: six-course estimate | held-out estimate, 95% CI, p | "
        "held-out courses with the same sign")
    group = None
    for g, label, e6, rh, pc in rows:
        if g != group:
            out(f"\n{plain(g)}")
            group = g
        p = "" if rh["p"] is None else f" p={rh['p']:.4f}"
        out(f"  {plain(label):40} six={e6:+.3f}  held-out={rh['est']:+.3f} "
            f"({rh['ci95'][0]:+.3f},{rh['ci95'][1]:+.3f}){p}  same sign {same_sign(e6, pc)}  "
            f"per-course: {', '.join(f'{x:+.3f}' for x in pc)}")
    with open(os.path.join(outdir, "heldout.txt"), "w", encoding="utf8", newline="\n") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
