"""Loading the evaluation results and checking that the design is complete.

RESULTS is the directory written by scripts/run_all.sh (or the extracted data archive):
    RESULTS/<reg>/evalmc/c<course>/w<width>_d2000000_s<seed>/summary_T_64x64<suffix>.json
    RESULTS/<reg>/evalmc/c<course>/w<width>_d2000000_s<seed>/outcomes_ppo_T<suffix>.jsonl

Regimes r1..r4 are P1-clean, P2-lat, P3-tilt, P4-comb. Suffixes: "" clean, "_lat200j100"
latency, "_att" tilt, "_lat200j100_att" combined.
"""

from __future__ import annotations

import json
import os

import numpy as np

DEV, FRESH = ["9000", "9001", "9002"], ["9100", "9101", "9102"]
ALL6 = DEV + FRESH                                   # the six analysis courses
HELDOUT = [str(c) for c in range(9200, 9210)]        # ten held-out courses (analysis/heldout.py)
ALLW = ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0"]
CORE = ["0.25", "0.5", "1.0", "2.0"]
SEEDS = range(10)
D2 = 2000000
N_EPISODES = 256
WIDTHS = {"r1": ALLW, "r2": CORE, "r3": CORE, "r4": ALLW}
NAME = {"r1": "P1-clean", "r2": "P2-lat", "r3": "P3-tilt", "r4": "P4-comb"}
LAT, TILT, COMB = "_lat200j100", "_att", "_lat200j100_att"
HEAD = {"r1": LAT, "r2": LAT, "r3": TILT, "r4": COMB}   # the condition each regime is about


def cell_dir(root, reg, c, w, s, d=D2):
    return os.path.join(root, reg, "evalmc", f"c{c}", f"w{w}_d{d}_s{s}")


def summary_path(root, reg, c, w, s, suf, d=D2):
    return os.path.join(cell_dir(root, reg, c, w, s, d), f"summary_T_64x64{suf}.json")


def outcomes_path(root, reg, c, w, s, suf, d=D2):
    return os.path.join(cell_dir(root, reg, c, w, s, d), f"outcomes_ppo_T{suf}.jsonl")


def succ(p):
    """success_first_ep of a summary file, or None if missing/malformed."""
    try:
        with open(p) as f:
            return json.load(f)["arms"]["ppo"]["success_first_ep"]
    except (OSError, KeyError, ValueError):
        return None


def expected_cells(courses=ALL6):
    """Every (reg, w, s, c, suffix) evaluation the paper uses on the given courses."""
    for reg in ("r1", "r2", "r3", "r4"):
        for w in WIDTHS[reg]:
            for s in SEEDS:
                for c in courses:
                    yield reg, w, s, c, HEAD[reg]
                    yield reg, w, s, c, ""
    # matched controls: P1-clean under combined (8 widths) and tilt (4 widths)
    for w in ALLW:
        for s in SEEDS:
            for c in courses:
                yield "r1", w, s, c, COMB
    for w in CORE:
        for s in SEEDS:
            for c in courses:
                yield "r1", w, s, c, TILT


def check_complete(root, courses=ALL6):
    """Abort unless every expected cell has a summary and N_EPISODES distinct first episodes."""
    bad, n_cells = [], 0
    for reg, w, s, c, suf in expected_cells(courses):
        n_cells += 1
        if succ(summary_path(root, reg, c, w, s, suf)) is None:
            bad.append(f"missing summary {reg} w{w} s{s} c{c} {suf or 'clean'}")
            continue
        op = outcomes_path(root, reg, c, w, s, suf)
        try:
            with open(op) as f:
                envs = [json.loads(line)["env"] for line in f]
        except (OSError, ValueError, KeyError):
            bad.append(f"missing/bad outcomes {op}")
            continue
        if len(envs) != N_EPISODES or len(set(envs)) != N_EPISODES:
            bad.append(f"{op}: {len(envs)} records, {len(set(envs))} distinct envs")
    if bad:
        raise SystemExit(f"INCOMPLETE: {len(bad)} of {n_cells} cells bad, e.g.\n  "
                         + "\n  ".join(bad[:10]))
    return n_cells


def matrix(root, reg, w, suf, courses=ALL6, d=D2):
    """(seeds, courses) success matrix, NaN where missing."""
    M = np.full((len(SEEDS), len(courses)), np.nan)
    for i, s in enumerate(SEEDS):
        for j, c in enumerate(courses):
            v = succ(summary_path(root, reg, c, w, s, suf, d))
            if v is not None:
                M[i, j] = v
    return M


def policies(root, reg, w, suf, courses=ALL6):
    """Policy-level success: each complete policy's mean over the courses."""
    M = matrix(root, reg, w, suf, courses)
    ok = ~np.isnan(M).any(axis=1)
    return M[ok].mean(axis=1)
