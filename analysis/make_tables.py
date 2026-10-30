#!/usr/bin/env python3
"""Regenerate the paper's tables and figure data on the six analysis courses.

    python analysis/make_tables.py RESULTS_DIR [OUT_DIR]      (OUT_DIR default: outputs/tables)

Writes LaTeX fragments:
    fig_main.tex       pgfplots series: perturbed success vs width (mean +- SEM)
    fig_robust.tex     pgfplots series: robustness vs width (95% CI)
    fig_benefit.tex    pgfplots series: matched training benefit vs width (95% CI)
    tab_trends.tex     size trends (slope per doubling of width) of success and robustness
    tab_benefit.tex    matched training benefit per width, and its size trend
    tab_retention.tex  clean, perturbed and robustness with CI for every series
    tab_failure.tex    how first episodes end (success / collision / breach / timeout)
    tab_sizes.tex      network sizes and counted compute (no data needed)

Aborts unless every expected evaluation is present (data.check_complete).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as P   # noqa: E402
import flops as fc  # noqa: E402
import stats as sc  # noqa: E402

COND = {"lat": (P.LAT, "latency"), "tilt": (P.TILT, "tilt"), "comb": (P.COMB, "combined")}
TRAINED = {"lat": "r2", "tilt": "r3", "comb": "r4"}
WID = {"lat": P.CORE, "tilt": P.CORE, "comb": P.ALLW}   # widths where the matched pair exists
# (condition, regime, widths): each trained regime next to P1-clean under the same condition
SERIES = [("lat", "r1", P.ALLW), ("lat", "r2", P.CORE), ("tilt", "r1", P.CORE), ("tilt", "r3", P.CORE),
          ("comb", "r1", P.ALLW), ("comb", "r4", P.ALLW)]


def _p(pa):
    return f"{pa:.3f}" if pa >= 0.001 else "$<$0.001"


def ws_of(widths):
    return sorted(widths, key=float)


class Tables:
    def __init__(self, root, out):
        self.root, self.out = root, out

    def write(self, name, text):
        with open(os.path.join(self.out, name), "w", encoding="utf8", newline="\n") as f:
            f.write(text)

    def pols(self, reg, w, suf):
        return P.policies(self.root, reg, w, suf)

    def fig_main(self):
        # Colour = test condition; solid filled = trained under it, dashed open = P1-clean.
        colour = {"lat": "blue", "tilt": "green!50!black", "comb": "red"}
        series, leg = [], []
        for cond, reg, W in SERIES:
            style = (f"{colour[cond]},mark=*" if reg != "r1"
                     else f"{colour[cond]},dashed,mark=o,mark options={{solid}}")
            coords = []
            for w in ws_of(W):
                p = self.pols(reg, w, COND[cond][0])
                coords.append(f"({w},{p.mean():.4f}) +- (0,{p.std(ddof=1) / np.sqrt(len(p)):.4f})")
            series.append(f"\\addplot[{style},error bars/.cd,y dir=both,y explicit] "
                          f"coordinates {{{' '.join(coords)}}};")
            leg.append(f"{P.NAME[reg]} / {COND[cond][1]}")
        self.write("fig_main.tex", "\n".join(series) + "\n\\legend{" + ",".join(leg) + "}\n")

    def fig_robust(self):
        """Robustness vs width with 95% policy-bootstrap CIs (asymmetric error bars). P3-tilt
        is omitted: its w=0.25 ratio (clean success ~0.2) is not interpretable."""
        colour = {"lat": "blue", "tilt": "green!50!black", "comb": "red"}
        series, leg = [], []
        for cond, reg, W in SERIES:
            if reg == "r3":
                continue
            style = (f"{colour[cond]},mark=*" if reg != "r1"
                     else f"{colour[cond]},dashed,mark=o,mark options={{solid}}")
            coords = []
            for w in ws_of(W):
                r = sc.retention(self.pols(reg, w, ""), self.pols(reg, w, COND[cond][0]))
                m, (lo, hi) = r["retention"], r["ci95"]
                coords.append(f"({w},{m:.4f}) += (0,{hi - m:.4f}) -= (0,{m - lo:.4f})")
            series.append(f"\\addplot[{style},error bars/.cd,y dir=both,y explicit] "
                          f"coordinates {{{' '.join(coords)}}};")
            leg.append(f"{P.NAME[reg]} / {COND[cond][1]}")
        self.write("fig_robust.tex", "\n".join(series) + "\n\\legend{" + ",".join(leg) + "}\n")

    def fig_benefit(self):
        """Matched training benefit vs width, one line per perturbation, with the same 95%
        policy-bootstrap CIs as tab_benefit."""
        colour = {"lat": "blue", "tilt": "green!50!black", "comb": "red"}
        series, leg = [], []
        for cond in ("lat", "tilt", "comb"):
            suf = COND[cond][0]
            coords = []
            for w in ws_of(WID[cond]):
                r = sc.unpaired_compare(self.pols("r1", w, suf), self.pols(TRAINED[cond], w, suf))
                m, (lo, hi) = r["mean_diff"], r["ci95"]
                coords.append(f"({w},{m:.4f}) += (0,{hi - m:.4f}) -= (0,{m - lo:.4f})")
            series.append(f"\\addplot[{colour[cond]},mark=*,error bars/.cd,y dir=both,y explicit] "
                          f"coordinates {{{' '.join(coords)}}};")
            leg.append(f"{P.NAME[TRAINED[cond]]} $-$ P1-clean ({COND[cond][1]})")
        self.write("fig_benefit.tex", "\n".join(series) + "\n\\legend{" + ",".join(leg) + "}\n")

    def tab_trends(self):
        xs_of = lambda W: np.log2([float(w) for w in ws_of(W)])  # noqa: E731
        ps, rs = [], []
        for cond, reg, W in SERIES:
            suf = COND[cond][0]
            ps.append(sc.trend_slope(xs_of(W), [self.pols(reg, w, suf) for w in ws_of(W)]))
            rs.append(sc.retention_trend(xs_of(W), [self.pols(reg, w, "") for w in ws_of(W)],
                                         [self.pols(reg, w, suf) for w in ws_of(W)]))
        pa = sc.holm([r["p_perm"] for r in ps])
        # P3-tilt's robustness at w=0.25 (clean success ~0.2) is not interpretable, so its
        # robustness slope is not reported or included in the Holm family.
        keep = [i for i, (_, reg, _) in enumerate(SERIES) if reg != "r3"]
        ra = [float("nan")] * len(rs)
        for i, v in zip(keep, sc.holm([rs[i]["p_perm"] for i in keep])):
            ra[i] = v
        f = lambda r: f"${r['slope']:+.3f}$ & $({r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f})$"  # noqa: E731
        lines = []
        for (cond, reg, W), p, a, r, b in zip(SERIES, ps, pa, rs, ra):
            rob = "-- & -- & --" if reg == "r3" else f"{f(r)} & {_p(b)}"
            lines.append(f"{P.NAME[reg]} / {COND[cond][1]} & {len(W)} & {f(p)} & {_p(a)} & {rob} \\\\")
        self.write("tab_trends.tex", (
            "\\begin{tabular}{lc rcc rcc}\n\\toprule\n"
            "& & \\multicolumn{3}{c}{perturbed success} & \\multicolumn{3}{c}{robustness} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Policy / test condition & widths & slope & 95\\% CI & $p_{\\text{Holm}}$ & slope & 95\\% CI & $p_{\\text{Holm}}$ \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))

    def tab_benefit(self):
        """Trained - P1-clean under the same condition; bold = Holm p < 0.05 within the
        condition; last rows = slope of the benefit per doubling of width."""
        res = {}
        for cond in ("lat", "tilt", "comb"):
            suf, W = COND[cond][0], ws_of(WID[cond])
            rows = [(w, sc.unpaired_compare(self.pols("r1", w, suf), self.pols(TRAINED[cond], w, suf)))
                    for w in W]
            adj = sc.holm([r["p_perm"] for _, r in rows])
            it = sc.interaction_slope(np.log2([float(w) for w in W]),
                                      [self.pols("r1", w, suf) for w in W],
                                      [self.pols(TRAINED[cond], w, suf) for w in W])
            res[cond] = ({w: (r, a) for (w, r), a in zip(rows, adj)}, it)
        lines = []
        for w in ws_of(P.ALLW):
            cells = []
            for cond in ("lat", "tilt", "comb"):
                if w not in res[cond][0]:
                    cells.append("-- & --"); continue
                r, a = res[cond][0][w]
                v = f"{r['mean_diff']:+.3f}"
                v = f"$\\mathbf{{{v}}}$" if a < 0.05 else f"${v}$"
                cells.append(f"{v} & $({r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f})$")
            lines.append(f"{w} & " + " & ".join(cells) + " \\\\")
        sl = [f"${res[c][1]['slope']:+.3f}$ & $({res[c][1]['ci95'][0]:+.3f},{res[c][1]['ci95'][1]:+.3f})$"
              for c in ("lat", "tilt", "comb")]
        lines += ["\\midrule", "slope & " + " & ".join(sl) + " \\\\",
                  "$p$ (slope) & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{_p(res[c][1]['p_boot'])}}}"
                                                 for c in ("lat", "tilt", "comb")) + " \\\\"]
        self.write("tab_benefit.tex", (
            "\\begin{tabular}{r cc cc cc}\n\\toprule\n"
            "& \\multicolumn{2}{c}{P2-lat $-$ P1-clean} & \\multicolumn{2}{c}{P3-tilt $-$ P1-clean} & "
            "\\multicolumn{2}{c}{P4-comb $-$ P1-clean} \\\\\n"
            "& \\multicolumn{2}{c}{(under latency)} & \\multicolumn{2}{c}{(under tilt)} & "
            "\\multicolumn{2}{c}{(under combined)} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "$w$ & benefit & 95\\% CI & benefit & 95\\% CI & benefit & 95\\% CI \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"))

    @staticmethod
    def _two_col(header, cols, blocks):
        """P1-clean series left, trained regimes right, as two top-aligned tables."""
        def sub(bl):
            return ("\\begin{tabular}[t]{" + cols + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
                    + "\n\\midrule\n".join(bl) + "\n\\bottomrule\n\\end{tabular}")
        left = [b for reg, b in blocks if reg == "r1"]
        right = [b for reg, b in blocks if reg != "r1"]
        return ("\\begin{tabular}{@{}c@{\\hspace{1.5em}}c@{}}\n" + sub(left) + "\n&\n" + sub(right)
                + "\n\\end{tabular}\n")

    def tab_retention(self):
        blocks = []
        for cond, reg, W in SERIES:
            suf = COND[cond][0]
            lines = [f"\\multicolumn{{5}}{{l}}{{\\textbf{{{P.NAME[reg]} under {COND[cond][1]}}}}} \\\\"]
            for w in ws_of(W):
                c, p = self.pols(reg, w, ""), self.pols(reg, w, suf)
                r = sc.retention(c, p)
                lines.append(f"{w} & {c.mean():.3f} & {p.mean():.3f} & {r['retention']:.2f} & "
                             f"$({r['ci95'][0]:.2f},{r['ci95'][1]:.2f})$ \\\\")
            blocks.append((reg, "\n".join(lines)))
        self.write("tab_retention.tex",
                   self._two_col("$w$ & clean & perturbed & robustness & 95\\% CI", "rcccc", blocks))

    def tab_failure(self):
        """First-episode outcomes per policy averaged over the 6 courses, then over the 10
        policies of a cell. A wall breach is logged by the env as a collision with
        lateral_breach = true. The success fraction must reproduce the summary's
        success_first_ep for every (policy, course)."""
        kinds = ("success", "collision", "breach", "timeout")
        series = [("r1", "", "clean"), ("r1", P.LAT, "latency"), ("r2", P.LAT, "latency"),
                  ("r1", P.TILT, "tilt"), ("r3", P.TILT, "tilt"),
                  ("r1", P.COMB, "combined"), ("r4", P.COMB, "combined")]

        def episode_kinds(path):
            first = {}
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["env"] in first:
                        continue
                    o = r["outcome"]
                    if o == "collision" and r.get("lateral_breach"):
                        o = "breach"
                    if o not in kinds:
                        sys.exit(f"unknown outcome {o!r} in {path}")
                    first[r["env"]] = o
            return first

        def cell(reg, suf, w):
            pol = []
            for s in P.SEEDS:
                per_course = []
                for c in P.ALL6:
                    k = episode_kinds(P.outcomes_path(self.root, reg, c, w, s, suf))
                    n = len(k)
                    fr = {x: sum(v == x for v in k.values()) / n for x in kinds}
                    ref = P.succ(P.summary_path(self.root, reg, c, w, s, suf))
                    if abs(fr["success"] - ref) > 5.1e-5 or n != P.N_EPISODES:
                        sys.exit(f"check failed {reg} w{w} s{s} c{c}{suf}: n={n} "
                                 f"success={fr['success']} summary={ref}")
                    per_course.append([fr[x] for x in kinds])
                pol.append(np.mean(per_course, axis=0))
            return np.mean(pol, axis=0)

        blocks = []
        for reg, suf, cond in series:
            lines = [f"\\multicolumn{{5}}{{l}}{{\\textbf{{{P.NAME[reg]} under {cond}}}}} \\\\"]
            for w in P.CORE:
                v = cell(reg, suf, w)
                lines.append(f"{w} & " + " & ".join(f"{x:.2f}" for x in v) + " \\\\")
            blocks.append((reg, "\n".join(lines)))
        self.write("tab_failure.tex",
                   self._two_col("$w$ & success & collision & breach & timeout", "rcccc", blocks))

    def tab_sizes(self):
        rows = []
        for w in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
            (c1, c2, c3), (t1, t2) = fc.scale(w)
            n, f = fc.count(w)
            rows.append(f"{w} & ({c1}, {c2}, {c3}) & ({t1}, {t2}) & {n:,} & {f / 1e6:.2f} \\\\")
        self.write("tab_sizes.tex", (
            "\\begin{tabular}{@{}r c c r r@{}}\n\\toprule\n"
            "$w$ & conv channels & trunk widths & $N$ & $F$ (MFLOP) \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", "tables")
    os.makedirs(out, exist_ok=True)
    print(f"completeness: {P.check_complete(root)} evaluations present")
    t = Tables(root, out)
    for fn in (t.fig_main, t.fig_robust, t.fig_benefit, t.tab_trends, t.tab_benefit,
               t.tab_retention, t.tab_failure, t.tab_sizes):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
