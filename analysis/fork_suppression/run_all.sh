#!/usr/bin/env bash
# Orchestrate the fork-suppression offline analysis end-to-end (plan §9 order).
#
# Stages:
#   1. data_gen  : reconstruct trajectories + teacher contexts from saved chat_completions (CPU).
#   2. forwards  : the ONLY NPU stage — per-position student/teacher top-k (needs CANN + torch_npu).
#   3. run.py    : P3 gate FIRST, then M1-M4 + placebo; M3 across steps.
#   4. plots.py  : 4 PDFs.
#
# Env: data_gen/run/plots are CPU-only (no CANN needed; point PYTHON_BIN at a CPU interpreter).
#      forwards NEEDS the CANN toolkit and a working torch_npu — npu_env.sh sets that up.
#
# Required:
#   RUN_DIR         training run dir holding chat_completions/<step>.jsonl
#   STAGED_PARQUET  staged dataset parquet the run trained on (patch_generation/train_verl.parquet)
#   BASE_MODEL      HF dir of the base model, which is also the teacher
# Optional:
#   STUDENT_CKPT (analysis/fork_suppression/merged_ckpts/sr_gs30)  merged student checkpoint
#   RLLM_ROOT     rllm checkout; data_gen needs it for the engine's tokenization primitives
#   PYTHON_BIN (python3)   ASCEND_DEV (0)   MEM_FRAC (0.30)
#
# Usage:
#   bash run_all.sh gate      # N=20 P3 gate (stop if it fails)
#   bash run_all.sh full 100  # N=100 all figs
#   bash run_all.sh full 300  # N>=300 final
#   bash run_all.sh m3        # fork-rate vs step (base/gs30/gs50/gs70)
set -u

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SWE_LADDER_ROOT:-$(cd -- "${HERE}/../.." && pwd)}"
cd "${REPO}"

ROOT=analysis/fork_suppression
PY="${PYTHON_BIN:-python3}"
STUDENT_GS30="${STUDENT_CKPT:-$ROOT/merged_ckpts/sr_gs30}"

missing=()
for var in RUN_DIR STAGED_PARQUET BASE_MODEL; do
  [[ -n "${!var:-}" ]] || missing+=("${var}")
done
if (( ${#missing[@]} )); then
  echo "ERROR: set ${missing[*]} — see the header of $0" >&2
  exit 2
fi
PARQUET="${STAGED_PARQUET}"
BASE="${BASE_MODEL}"

# swe_ladder resolves from this repo; rllm (engine primitives, data_gen only) from its checkout.
CPU_ENV="PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=${REPO}${RLLM_ROOT:+:${RLLM_ROOT}}"
mkdir -p $ROOT/out

datagen() {  # $1=chat_completions jsonl  $2=limit  $3=out.pkl
  env $CPU_ENV $PY $ROOT/data_gen.py \
    --chat-completions "$1" --parquet "$PARQUET" --model-path "$BASE" \
    --out "$3" --limit "$2"
}

forwards() {  # $1=data.pkl  $2=student  $3=out.pkl  ($4=limit optional)
  # NPU stage. Pure STATIC CANN env via npu_env.sh — do NOT source setenv.bash
  # (it runs a filesystem-wide `find /`). Isolate to one device (ASCEND_DEV, default 0) and
  # hard-cap HBM to 30% so a forward spike can never OOM a co-resident training job.
  ASCEND_DEV="${ASCEND_DEV:-0}" bash $ROOT/npu_env.sh $PY $ROOT/forwards.py \
    --data "$1" --student "$2" --teacher "$BASE" --out "$3" --mem-fraction "${MEM_FRAC:-0.30}" ${4:+--limit $4}
}

case "${1:-}" in
  gate)
    datagen "$RUN_DIR/chat_completions/30.jsonl" 20 $ROOT/out/data_gs30_n20.pkl
    forwards $ROOT/out/data_gs30_n20.pkl "$STUDENT_GS30" $ROOT/out/fwd_gs30_n20.pkl
    env $CPU_ENV $PY $ROOT/run.py p3gate --forwards $ROOT/out/fwd_gs30_n20.pkl
    ;;
  full)
    N="${2:-100}"
    datagen "$RUN_DIR/chat_completions/30.jsonl" "$N" $ROOT/out/data_gs30_n$N.pkl
    forwards $ROOT/out/data_gs30_n$N.pkl "$STUDENT_GS30" $ROOT/out/fwd_gs30_n$N.pkl
    env $CPU_ENV $PY $ROOT/run.py analyze --forwards $ROOT/out/fwd_gs30_n$N.pkl \
      --out $ROOT/results.json || { echo "P3 gate failed"; exit 2; }
    env $CPU_ENV $PY $ROOT/plots.py --results $ROOT/results.json \
      --forwards $ROOT/out/fwd_gs30_n$N.pkl --outdir $ROOT/figs
    ;;
  m3)
    # fork-rate vs step: base + gs30/gs50/gs70 (student varies; same chat_completions source per step)
    echo "M3 requires merged student ckpts per step + a base condition; see REPORT.md M3 section."
    ;;
  *)
    echo "usage: bash run_all.sh {gate|full [N]|m3}"; exit 1;;
esac
