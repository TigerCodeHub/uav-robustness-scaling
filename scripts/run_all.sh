#!/usr/bin/env bash
# Train and evaluate the paper's full design (240 policies) on one GPU, sequentially.
#
#   r1 = P1-clean  widths 0.25..2.0 (8)   evals: clean, latency, combined; tilt at the 4 core widths
#   r2 = P2-lat    widths 0.25 0.5 1.0 2.0  evals: clean, latency
#   r3 = P3-tilt   widths 0.25 0.5 1.0 2.0  evals: clean, tilt
#   r4 = P4-comb   widths 0.25..2.0 (8)   evals: clean, combined
# 10 seeds each; every 2M-step policy is evaluated on the six analysis courses (9000-9002,
# 9100-9102) and the ten held-out courses (9200-9209).
#
# Resumable: a run whose 2M checkpoint exists is not retrained, and an eval whose summary
# exists is skipped. Output layout (read by analysis/):
#   OUT/<reg>/<reg>_w<W>_s<S>[_<tag>]_d2000000.pt
#   OUT/<reg>/evalmc/c<course>/w<W>_d2000000_s<S>/summary_T_64x64<suffix>.json
#
# Usage:  scripts/run_all.sh [OUT_DIR] [JOB ...]      JOB = <reg>:<width>:<seed>, e.g. r1:1.0:0
#   PYTHON=/path/to/isaaclab/python   (default: python)
# Each training run takes ~25 min and each evaluation ~2 min on an RTX A1000; the full design
# (240 runs, 9,600 evaluations) is ~420 GPU-hours, so run a subset of JOBs or split the job list across GPUs
# (one process per GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OMNI_KIT_ACCEPT_EULA:?read and accept the NVIDIA Omniverse EULA, then export OMNI_KIT_ACCEPT_EULA=YES}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
OUT="${1:-results}"; shift || true
mkdir -p "$OUT/logs"

D=2000000
COURSES="9000 9001 9002 9100 9101 9102 9200 9201 9202 9203 9204 9205 9206 9207 9208 9209"
LAT="--cmd-latency 0.2 --cmd-latency-jitter 0.1"
ATT="--attitude-coupling --attitude-max-deg 20"
ALLW="2.0 1.0 0.5 0.25 1.75 1.5 1.25 0.75"; CORE="2.0 1.0 0.5 0.25"

is_core() { case " $CORE " in *" $1 "*) return 0 ;; esac; return 1; }

ev() { # ckpt course out_dir suffix [flags...]
  local ck=$1 c=$2 od=$3 suf=$4; shift 4
  [ -f "$od/summary_T_64x64$suf.json" ] && return 0
  mkdir -p "$od"
  timeout -s KILL 30m "$PY" scripts/evaluate.py --ckpt "$ck" --course "$c" --out-dir "$od" "$@" \
    > "$od/eval$suf.log" 2>&1 || echo "FAIL eval $ck c$c $suf"
}

run_job() { # reg width seed
  local REG=$1 W=$2 S=$3 TRAIN_FLAGS HEAD_SUF HEAD_FLAGS
  case $REG in
    r1) TRAIN_FLAGS="";          HEAD_SUF="_lat200j100";     HEAD_FLAGS="$LAT" ;;
    r2) TRAIN_FLAGS="$LAT";      HEAD_SUF="_lat200j100";     HEAD_FLAGS="$LAT" ;;
    r3) TRAIN_FLAGS="$ATT";      HEAD_SUF="_att";            HEAD_FLAGS="$ATT" ;;
    r4) TRAIN_FLAGS="$LAT $ATT"; HEAD_SUF="_lat200j100_att"; HEAD_FLAGS="$LAT $ATT" ;;
    *) echo "unknown regime $REG"; return 1 ;;
  esac
  local RID="${REG}_w${W}_s${S}" RD="$OUT/$REG"
  mkdir -p "$RD"
  local CK; CK=$(ls "$RD/${RID}"*_d${D}.pt 2>/dev/null | head -1 || true)
  if [ -z "$CK" ]; then
    echo "=== train $RID $(date -u)"
    # shellcheck disable=SC2086
    timeout -s KILL 5h "$PY" scripts/train.py --width-mult "$W" --seed "$S" $TRAIN_FLAGS \
      --out-dir "$RD" --run-id "$RID" > "$OUT/logs/train_${RID}.log" 2>&1 \
      || { echo "FAIL train $RID"; return 1; }
    CK=$(ls "$RD/${RID}"*_d${D}.pt | head -1)
  fi
  for C in $COURSES; do
    local OD="$RD/evalmc/c${C}/w${W}_d${D}_s${S}"
    # shellcheck disable=SC2086
    ev "$CK" "$C" "$OD" "$HEAD_SUF" $HEAD_FLAGS
    ev "$CK" "$C" "$OD" ""
    if [ "$REG" = r1 ]; then
      # matched controls: P1-clean under the other regimes' conditions
      # shellcheck disable=SC2086
      ev "$CK" "$C" "$OD" "_lat200j100_att" $LAT $ATT
      # shellcheck disable=SC2086
      is_core "$W" && ev "$CK" "$C" "$OD" "_att" $ATT
    fi
  done
  echo "DONE $RID $(date -u)"
}

JOBS="$*"
if [ -z "$JOBS" ]; then
  for W in $ALLW; do for S in 0 1 2 3 4 5 6 7 8 9; do JOBS="$JOBS r4:$W:$S r1:$W:$S"; done; done
  for W in $CORE; do for S in 0 1 2 3 4 5 6 7 8 9; do JOBS="$JOBS r2:$W:$S r3:$W:$S"; done; done
fi
for J in $JOBS; do
  IFS=: read -r REG W S <<< "$J"
  run_job "$REG" "$W" "$S" || true
done
