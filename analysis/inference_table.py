#!/usr/bin/env python3
"""Inference time per decision at each width (the paper's timing table).

    python analysis/inference_table.py [TIMING_DIR] [OUT_DIR]
    (defaults: paper_outputs/inference_timing, outputs/tables)

Formats the three timing runs written by scripts/inference_timing.py: cpu1.json (one
thread, pinned to one performance core), cpu4.json (four threads on four performance
cores) and cuda.json (GPU, including the input and action copies). The committed runs
were measured on an Intel i7-14700 with an NVIDIA RTX A1000 on 2026-10-28; times are
medians (and the 99th percentile for one thread) over 10,000 batch-1 decisions per width.
Timings are measurements, so re-running scripts/inference_timing.py gives similar, not
identical, numbers; this script only formats them.
"""

import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join("paper_outputs", "inference_timing")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", "tables")


def rows(name):
    with open(os.path.join(D, name)) as f:
        return {r["width"]: r for r in json.load(f)["rows"]}


cpu1, cpu4, gpu = rows("cpu1.json"), rows("cpu4.json"), rows("cuda.json")
lines = [r"\begin{tabular}{@{}r r c c c c@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{CPU, 1 thread} & CPU, 4 threads & GPU \\",
         r"\cmidrule(lr){3-4}",
         r"$w$ & $N$ & median & p99 & median & median \\", r"\midrule"]
for w in sorted(cpu1):
    lines.append(f"{w} & {cpu1[w]['params']:,} & {cpu1[w]['median_ms']:.3f} & {cpu1[w]['p99_ms']:.3f}"
                 f" & {cpu4[w]['median_ms']:.3f} & {gpu[w]['median_ms']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "tab_inference.tex"), "w", encoding="utf8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")

# Numbers the paper quotes in its text. 50 ms is one 20 Hz control step; 0.2 s is the mean
# command latency studied; "10x" bounds a ten-times-larger network by ten w=2.0 forward passes.
lo, hi = cpu1[min(cpu1)]["median_ms"], cpu1[max(cpu1)]["median_ms"]
ten_x = 10 * hi
g = [r["median_ms"] for r in gpu.values()]
text = [f"1 thread median: w={min(cpu1)} {lo:.2f} ms, w={max(cpu1)} {hi:.2f} ms (difference {hi - lo:+.2f} ms)",
        f"4 threads median: w={max(cpu4)} {cpu4[max(cpu4)]['median_ms']:.2f} ms",
        f"GPU median range: {min(g):.2f}-{max(g):.2f} ms",
        f"10x the w={max(cpu1)} 1-thread time: {ten_x:.1f} ms = {100 * ten_x / 50:.0f}% of a 50 ms control step, "
        f"{100 * ten_x / 200:.1f}% of the 0.2 s mean latency"]
with open(os.path.join(OUT, "inference_numbers.txt"), "w", encoding="utf8", newline="\n") as f:
    f.write("\n".join(text) + "\n")
print("\n".join(text))
print("wrote tab_inference.tex, inference_numbers.txt")
