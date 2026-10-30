# Data manifest

The trained policies and every evaluation behind the paper "Model-Size Trends in Latency
and Attitude-Coupling Robustness of a Depth-Based UAV Policy" (Tiger Deng) are archived on
Zenodo as one file, `data_results.tar.gz`:

- DOI: [10.5281/zenodo.23784876](https://doi.org/10.5281/zenodo.23784876)
- License: CC-BY-4.0 (the code is MIT)
- Code: https://github.com/TigerCodeHub/uav-robustness-scaling (DOI:
  [10.5281/zenodo.23786688](https://doi.org/10.5281/zenodo.23786688), all versions)

The archive contains this file, `DATA_CHECKSUMS.txt` and `data/results/` (469 MB, 19,680
files). Extract it at the root of the code repository. `DATA_CHECKSUMS.txt` lists the
SHA-256 of every file under `data/results/`, and `scripts/reproduce_paper.sh` verifies them
before running the analysis.

```bash
tar -xzf data_results.tar.gz
sha256sum -c DATA_CHECKSUMS.txt
```

## Contents

| Files | Count | Size | What |
|---|---|---|---|
| `<reg>/<run>_d2000000.pt` | 240 | 115 MB | policy weights at 2M steps (torch state_dict; load with `gpuperception.ppo_net.infer_arch` + `build_policy`) |
| `<reg>/<run>_d2000000_snap.json` | 240 | 0.2 MB | env-steps, wall clock, size and learning rate at the snapshot |
| `<reg>/evalmc/c<course>/w<W>_d2000000_s<S>/summary_T_64x64<suffix>.json` | 9,600 | 38 MB | one evaluation: `arms.ppo.success_first_ep`, counts, settings, provenance |
| `<reg>/evalmc/c<course>/w<W>_d2000000_s<S>/outcomes_ppo_T<suffix>.jsonl` | 9,600 | 300 MB | one line per first episode: `env`, `outcome`, `lateral_breach`, `max_abs_y`, `steps` |

- `<reg>`: r1 = P1-clean, r2 = P2-lat, r3 = P3-tilt, r4 = P4-comb (240 policies:
  8 + 4 + 4 + 8 widths × 10 seeds)
- `<run>`: `<reg>_w<W>_s<S>` plus the training perturbation tag (`_lat200j100`, `_att`,
  `_lat200j100_att`; none for P1-clean)
- `<suffix>` (evaluation condition): empty = clean, `_lat200j100` = latency (mean 0.2 s,
  std 0.1 s), `_att` = tilt (cap 20°), `_lat200j100_att` = combined
- `<course>`: the six analysis courses 9000–9002, 9100–9102 (3,600 evaluations) and the
  ten held-out courses 9200–9209 (6,000 evaluations)
- `<W>`: 0.25 … 2.0; `<S>`: 0 … 9

Every policy is evaluated under its own condition and clean; P1-clean additionally under
combined (all widths) and tilt (the four core widths), on all 16 courses.

## How the archive was prepared

The files are the original outputs of the runs, copied unchanged except for the
evaluation summaries' `provenance` block: the machine hostname was removed, and paths in
`argv` and `ckpt` were rewritten from the original output directory to `data/results/`.
Training-time resume states, ONNX exports, logs, per-step trajectory records and
intermediate snapshots are not included.

The summaries were written by an earlier version of this code and carry fields this
repository no longer produces (for example OANE path-error components, `success_mean`
over all completed episodes, `tier`, and `audit_flags` for what `scripts/evaluate.py` now
writes as `env`). The analysis reads only `arms.ppo.success_first_ep` and the outcomes
files. Re-evaluating the archived `r2_w1.0_s0` checkpoint on course 9000 (clean and
latency) with this repository's `scripts/evaluate.py` reproduced both archived outcomes
files line for line.

The tarball is reproducible from the repository root (GNU tar):

```bash
tar --sort=name --mtime=2026-10-20T00:00:00Z --owner=0 --group=0 --numeric-owner \
  --mode=u+rw,go+r,go-w -cf - DATA_MANIFEST.md DATA_CHECKSUMS.txt data/results \
  | gzip -n -9 > data_results.tar.gz
```
