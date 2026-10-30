# UAV policy robustness vs. network size: Isaac Lab study

[![CI](https://github.com/TigerCodeHub/uav-robustness-scaling/actions/workflows/ci.yml/badge.svg)](https://github.com/TigerCodeHub/uav-robustness-scaling/actions/workflows/ci.yml)
[![Reproduce paper](https://github.com/TigerCodeHub/uav-robustness-scaling/actions/workflows/reproduce.yml/badge.svg)](https://github.com/TigerCodeHub/uav-robustness-scaling/actions/workflows/reproduce.yml)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23786688.svg)](https://doi.org/10.5281/zenodo.23786688)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23784876.svg)](https://doi.org/10.5281/zenodo.23784876)

Code, trained policies, and evaluation results for

> T. Deng, *Model-Size Trends in Latency and Attitude-Coupling Robustness of a Depth-Based
> UAV Policy* (manuscript).

A quadrotor policy flies a 40 m obstacle corridor in Isaac Sim 5.1 / Isaac Lab 2.3.2 from one
64×64 depth image and a 7-D state, outputting a body-frame velocity command. PPO policies at
eight network widths (5,759 to 323,783 parameters) are trained clean or under command latency
(mean 0.2 s), attitude-coupled tilt (up to 20°), or both, and tested clean and degraded on
unseen procedurally generated courses. Larger networks were more robust under every
degradation, even when trained clean; how much degradation training adds as networks grow
depends on the degradation (it nearly vanishes for tilt, changes little for latency, and grows
for both combined).

| Regime   | Trained under   | Widths `w`           | Evaluated under                         |
|----------|-----------------|----------------------|-----------------------------------------|
| P1-clean | nothing         | 0.25–2.0 (8)         | clean, latency, combined; tilt at 4 widths |
| P2-lat   | latency         | 0.25, 0.5, 1.0, 2.0  | clean, latency                          |
| P3-tilt  | tilt            | 0.25, 0.5, 1.0, 2.0  | clean, tilt                             |
| P4-comb  | latency + tilt  | 0.25–2.0 (8)         | clean, combined                         |

Ten seeds per cell (240 policies), 2M environment steps each. Every policy is evaluated
on six analysis courses (seeds 9000–9002, 9100–9102) and, separately, on ten held-out
courses (9200–9209) that were not used while the analysis was developed.

The trained policies and evaluation results are on Zenodo (dataset DOI:
[10.5281/zenodo.23784876](https://doi.org/10.5281/zenodo.23784876)); see
[`DATA_MANIFEST.md`](DATA_MANIFEST.md). Releases of this repository are archived on Zenodo
(code DOI: [10.5281/zenodo.23786688](https://doi.org/10.5281/zenodo.23786688), all
versions; v1.0.0 is [10.5281/zenodo.23786689](https://doi.org/10.5281/zenodo.23786689)).

## Repository layout

```
gpuperception/    simulation environment, perturbations, network (the package)
scripts/          train.py, evaluate.py, run_all.sh (whole design on one GPU), reproduce_paper.sh,
                  inference_timing.py
analysis/         regenerates every table, the figure data and the in-text numbers
paper_outputs/    the committed outputs of analysis/, as used in the paper
tests/            CPU-only unit tests
DATA_MANIFEST.md  the data archive (Zenodo): contents, layout, how it was prepared
DATA_CHECKSUMS.txt  SHA-256 of every file in the data archive
```

## Reproducing the paper's numbers from the archived results

The data archive is on Zenodo: [10.5281/zenodo.23784876](https://doi.org/10.5281/zenodo.23784876)
(`data_results.tar.gz`, 148,895,363 bytes, SHA-256
`491148827a9e9f10e1e16a7d2f5872ba46e329f73c07ddda271eccdb1ad335a5`; CC-BY-4.0).
`DATA_MANIFEST.md` describes its contents. Needs Python 3.10+ and numpy.

```bash
pip install -r requirements.txt
tar -xzf data_results.tar.gz          # extracted at the repository root -> data/results/
scripts/reproduce_paper.sh            # checksums, all analyses, and a diff against paper_outputs/
```

On Windows (Git Bash), GNU tar reads `C:` in a path as a remote host: extract with a
relative path or add `--force-local`.

| Paper item | Output (in `paper_outputs/`) | Script |
|---|---|---|
| Fig. perturbed success vs size | `tables/fig_main.tex` | `analysis/make_tables.py` |
| Fig. robustness vs size | `tables/fig_robust.tex` | `analysis/make_tables.py` |
| Fig. training benefit vs size | `tables/fig_benefit.tex` | `analysis/make_tables.py` |
| Size trends, training benefit, robustness detail | `tables/tab_trends.tex`, `tab_benefit.tex`, `tab_retention.tex` | `analysis/make_tables.py` |
| Failure modes | `tables/tab_failure.tex` | `analysis/make_tables.py` |
| Network sizes | `tables/tab_sizes.tex` | `analysis/make_tables.py` |
| Held-out courses | `tables/tab_heldout_effects.tex`, `tab_heldout_levels.tex`, `tab_heldout_p1x.tex`, `heldout.txt` | `analysis/heldout.py` |
| Compounding, seed correlations, throughput | `prose_numbers.txt` | `analysis/prose_numbers.py` |
| Inference time per decision (table and text) | `tables/tab_inference.tex`, `tables/inference_numbers.txt` | `analysis/inference_table.py` |

Every statistic seeds its own random generator, so the outputs are identical run to run.

Inference times are measurements, not computed from the data archive: the committed runs
(`paper_outputs/inference_timing/*.json`, Intel i7-14700 and NVIDIA RTX A1000) were produced
by `scripts/inference_timing.py` from the archived w=0.25…2.0 checkpoints (needs torch, no
simulator). Re-measuring gives similar but not identical numbers:

```bash
taskset -c 2 python scripts/inference_timing.py --threads 1 --out cpu1.json
taskset -c 2-5 python scripts/inference_timing.py --threads 4 --out cpu4.json
python scripts/inference_timing.py --device cuda --out cuda.json
```

## Re-running the experiments

Tested on an x86-64 Debian machine with one NVIDIA RTX A1000 (8 GB), driver 550,
Isaac Sim 5.1 and Isaac Lab 2.3.2 under Python 3.11. `requirements-isaac.txt` is
the exact environment.

```bash
pip install -r requirements-isaac.txt --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES          # after reading and accepting NVIDIA's Omniverse EULA
python -m gpuperception.env              # smoke check: builds a course and flies 20 steps
```

One policy, then one evaluation:

```bash
python scripts/train.py --width-mult 1.0 --seed 0 --cmd-latency 0.2 --cmd-latency-jitter 0.1 \
    --out-dir results/r2 --run-id r2_w1.0_s0
python scripts/evaluate.py --ckpt results/r2/r2_w1.0_s0_lat200j100_d2000000.pt --course 9000 \
    --cmd-latency 0.2 --cmd-latency-jitter 0.1 --out-dir results/r2/evalmc/c9000/w1.0_d2000000_s0
```

The whole design (resumable; pass `reg:width:seed` jobs to run a subset):

```bash
PYTHON=python scripts/run_all.sh results              # or: scripts/run_all.sh results r1:1.0:0 r4:1.0:0
scripts/reproduce_paper.sh results outputs_rerun
```

A training run takes about 25 minutes and an evaluation about 2 minutes on an RTX A1000,
so the full design (240 runs, 9,600 evaluations) is about 420 GPU-hours; split the job list across
GPUs, one process per GPU. GPU physics and rendering are not bitwise deterministic, so
re-trained policies reproduce the archived ones statistically, not exactly. Evaluation of a
fixed checkpoint is repeatable: the archived summaries were written by an earlier version of
`scripts/evaluate.py`, and re-evaluating the archived `r2_w1.0_s0` checkpoint on course 9000
(clean and latency) with this version reproduced both archived outcomes files line for line.
The repository's `ScenarioSpec` also reproduces the archived `spec_hash` and `course_hash` of
all 16 courses.

**Known behaviour.** On the first step of every episode the policy sees the depth frame
rendered before the reset: the camera is not re-rendered at reset (found by reading the
code, not measured). This holds in training and evaluation alike, and in the archived runs.

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs the tests on every push (`.github/workflows/ci.yml`). On every push to
`main` and every pull request, `.github/workflows/reproduce.yml` resolves the data DOI,
downloads the archive from Zenodo, verifies it and runs `scripts/reproduce_paper.sh`.

## License and citation

MIT (see `LICENSE`). Please cite the paper; `CITATION.cff` has the details.
