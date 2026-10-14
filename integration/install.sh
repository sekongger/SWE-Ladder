#!/usr/bin/env bash
# Graft SWE-Ladder onto the upstream training stack.
#
# SWE-Ladder is an overlay, not a fork. Code we wrote ships as files in this
# repository; the handful of upstream files we had to change ship as patches
# under integration/patches/. This script does both halves:
#
#   1. applies the three patches to the upstream checkouts (rllm, R2E-Gym, verl)
#   2. installs swe_ladder/ as the `rllm.opd` package and training/ into
#      examples/swe/, which is where the launchers expect to live (they derive
#      the workspace root from their own path: ${SCRIPT_DIR}/../../../..)
#
# Re-running is safe: already-applied patches are detected and skipped, and the
# installed files are replaced.
#
# Usage:
#   bash integration/install.sh --workspace /path/to/workspace
#   bash integration/install.sh --workspace /path/to/workspace --check
#   bash integration/install.sh --rllm /path/rllm --r2egym /path/R2E-Gym
#
# Options:
#   --workspace DIR   tree holding DeepSWE_RL/rllm and R2E-Gym
#                     (default: $WORKSPACE_ROOT, else the current directory)
#   --rllm DIR        override the rllm checkout    (default <workspace>/DeepSWE_RL/rllm)
#   --r2egym DIR      override the R2E-Gym checkout (default <workspace>/R2E-Gym)
#   --verl DIR        override the verl checkout    (default <rllm>/verl)
#   --mode copy|link  how swe_ladder/ and training/ land in the rllm tree.
#                     `link` symlinks them back to this repo, so edits here take
#                     effect without re-installing. Default: copy.
#   --skip-patches    install our files only, leave upstream files untouched
#   --force           overwrite files that are neither ours nor backed up
#   --check           report what each step would do; change nothing
#   -h, --help        this text
#
# Env: PYTHON_BIN (python3) is used only for the post-install smoke check.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${HERE}/.." && pwd)"

# The upstream commits these patches were cut against. Any other commit may still
# work (git apply is content-addressed, not commit-addressed), but it is not what
# we tested; the script says so instead of failing silently.
SWE_MASTER_URL="https://github.com/RUCAIBox/SWE-Master"
SWE_MASTER_PIN="fb864a7d397b2c287ba6e38152da2228efc98e03"
VERL_URL="https://github.com/verl-project/verl"
VERL_PIN="d4399a673ae60748271af009d38e9a3dca86086b"

RECEIPT_NAME=".swe_ladder_install"
BACKUP_SUFFIX=".pre-swe-ladder"

WORKSPACE=""
RLLM=""
R2EGYM=""
VERL=""
MODE="copy"
SKIP_PATCHES=0
FORCE=0
CHECK=0
PY="${PYTHON_BIN:-python3}"

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "[install] $*"; }
plan() { echo "[install] would $*"; }

while (($#)); do
  case "$1" in
    --workspace) WORKSPACE="${2:?}"; shift 2;;
    --rllm)      RLLM="${2:?}";      shift 2;;
    --r2egym)    R2EGYM="${2:?}";    shift 2;;
    --verl)      VERL="${2:?}";      shift 2;;
    --mode)      MODE="${2:?}";      shift 2;;
    --skip-patches) SKIP_PATCHES=1;  shift;;
    --force)     FORCE=1;            shift;;
    --check)     CHECK=1;            shift;;
    -h|--help)   usage; exit 0;;
    *) die "unknown argument: $1 (try --help)";;
  esac
done

[[ "${MODE}" == copy || "${MODE}" == link ]] || die "--mode must be copy or link, got ${MODE}"

WORKSPACE="${WORKSPACE:-${WORKSPACE_ROOT:-$PWD}}"
[[ -d "${WORKSPACE}" ]] || die "workspace not found: ${WORKSPACE}"
WORKSPACE="$(cd -- "${WORKSPACE}" && pwd)"
RLLM="${RLLM:-${WORKSPACE}/DeepSWE_RL/rllm}"
R2EGYM="${R2EGYM:-${WORKSPACE}/R2E-Gym}"
VERL="${VERL:-${RLLM}/verl}"

command -v git >/dev/null || die "git is required (the patches are applied with \`git apply\`)"

missing=()
[[ -d "${RLLM}/rllm" ]]       || missing+=("rllm: ${RLLM}")
[[ -d "${R2EGYM}/src" ]]      || missing+=("R2E-Gym: ${R2EGYM}")
[[ -d "${VERL}/verl" ]]       || missing+=("verl: ${VERL}")
if ((${#missing[@]})); then
  {
    echo "ERROR: upstream checkout(s) not found:"
    printf '  %s\n' "${missing[@]}"
    echo
    echo "Expected layout (override with --rllm/--r2egym/--verl):"
    echo "  <workspace>/DeepSWE_RL/rllm      <- rllm + verl, from SWE-Master"
    echo "  <workspace>/DeepSWE_RL/rllm/verl <- verl (its own git repo)"
    echo "  <workspace>/R2E-Gym              <- the agent scaffold / sandbox"
    echo
    echo "Get them with:"
    echo "  git clone ${SWE_MASTER_URL} <workspace> && git -C <workspace> checkout ${SWE_MASTER_PIN}"
    echo "  git clone ${VERL_URL} <workspace>/DeepSWE_RL/rllm/verl && \\"
    echo "    git -C <workspace>/DeepSWE_RL/rllm/verl checkout ${VERL_PIN}"
  } >&2
  exit 1
fi
RLLM="$(cd -- "${RLLM}" && pwd)"
R2EGYM="$(cd -- "${R2EGYM}" && pwd)"
VERL="$(cd -- "${VERL}" && pwd)"

say "repo       ${REPO}"
say "rllm       ${RLLM}"
say "R2E-Gym    ${R2EGYM}"
say "verl       ${VERL}"
say "mode       ${MODE}$( ((CHECK)) && echo "  (--check: nothing will be written)")"

# ---------------------------------------------------------------- 1. patches

# Report whether the target is sitting on the pin we cut the patch against. This
# is advisory only: the patch either applies or it does not, and `git apply`
# decides that by content.
pin_note() {  # $1=dir  $2=pin  $3=label
  local head
  head="$(git -C "$1" rev-parse HEAD 2>/dev/null || true)"
  if [[ -z "${head}" ]]; then
    say "  ${3}: not a git checkout — applying by content only"
  elif [[ "${head}" == "$2" ]]; then
    say "  ${3}: at the tested pin ${2:0:12}"
  else
    say "  ${3}: HEAD ${head:0:12} != tested pin ${2:0:12} (patch will still be tried)"
  fi
}

apply_patch() {  # $1=target dir  $2=patch file  $3=label  $4=pin
  local dir="$1" patch="$2" label="$3" pin="$4"
  [[ -f "${patch}" ]] || die "missing patch: ${patch}"
  pin_note "${dir}" "${pin}" "${label}"

  if git -C "${dir}" apply --reverse --check "${patch}" 2>/dev/null; then
    say "  ${label}: already applied — skipping"
    return 0
  fi
  if git -C "${dir}" apply --check "${patch}" 2>/dev/null; then
    if ((CHECK)); then plan "apply ${label} patch cleanly"; return 0; fi
    git -C "${dir}" apply "${patch}"
    say "  ${label}: applied"
    return 0
  fi
  # Partially applied, or a tree that has drifted from the pin. A 3-way merge can
  # often still place the changes, but only inside a real git repo with the blobs.
  if git -C "${dir}" apply --3way --check "${patch}" 2>/dev/null; then
    if ((CHECK)); then plan "apply ${label} patch via 3-way merge (tree has drifted)"; return 0; fi
    git -C "${dir}" apply --3way "${patch}"
    say "  ${label}: applied via 3-way merge — review the result with \`git -C ${dir} diff\`"
    return 0
  fi
  {
    echo "ERROR: ${label} patch does not apply to ${dir}"
    echo "  patch: ${patch}"
    echo "  tested against: ${pin}"
    echo "  Most likely the checkout is at a different commit, or a previous run"
    echo "  applied part of it. Reset the upstream tree and retry:"
    echo "    git -C ${dir} checkout ${pin} && git -C ${dir} checkout -- ."
    echo "  Diagnose a specific hunk with: git -C ${dir} apply -v --check ${patch}"
  } >&2
  exit 1
}

if ((SKIP_PATCHES)); then
  say "patches: skipped (--skip-patches)"
else
  say "applying patches"
  apply_patch "${RLLM}"   "${HERE}/patches/rllm-swe-ladder.patch"   "rllm"    "${SWE_MASTER_PIN}"
  apply_patch "${R2EGYM}" "${HERE}/patches/r2egym-swe-ladder.patch" "R2E-Gym" "${SWE_MASTER_PIN}"
  apply_patch "${VERL}"   "${HERE}/patches/verl-swe-ladder.patch"   "verl"    "${VERL_PIN}"
fi

# ------------------------------------------------------- 2. our files, grafted

OPD_DIR="${RLLM}/rllm/opd"
SWE_EX="${RLLM}/examples/swe"

# Was this tree installed by a previous run of this script? Recorded now, because
# install_opd() removes the receipt along with the old directory.
WAS_OURS=0
[[ -e "${OPD_DIR}/${RECEIPT_NAME}" || -L "${OPD_DIR}" ]] && WAS_OURS=1
ours() { ((WAS_OURS)); }

copy_tree() {  # $1=source dir  $2=dest dir (created; contents merged, not cleared)
  mkdir -p "$2"
  tar -C "$1" --exclude='__pycache__' --exclude='*.pyc' -cf - . | tar -C "$2" -xf -
}

install_opd() {
  if [[ -L "${OPD_DIR}" ]]; then
    ((CHECK)) && { plan "replace the existing symlink ${OPD_DIR}"; return 0; }
    rm -f "${OPD_DIR}"
  elif [[ -d "${OPD_DIR}" ]]; then
    if ! ours && ((!FORCE)); then
      die "${OPD_DIR} exists and was not installed by this script. Move it aside, or pass --force."
    fi
    ((CHECK)) && { plan "replace the existing directory ${OPD_DIR}"; return 0; }
    rm -rf "${OPD_DIR}"
  elif ((CHECK)); then
    plan "install swe_ladder/ -> ${OPD_DIR} (${MODE})"
    return 0
  fi

  mkdir -p "$(dirname -- "${OPD_DIR}")"
  if [[ "${MODE}" == link ]]; then
    ln -s "${REPO}/swe_ladder" "${OPD_DIR}"
    say "  linked swe_ladder/ -> rllm/opd"
  else
    copy_tree "${REPO}/swe_ladder" "${OPD_DIR}"
    say "  copied swe_ladder/ -> rllm/opd"
  fi
}

install_training() {
  if [[ "${MODE}" == link ]]; then
    install_training_links
    return 0
  fi
  # Upstream's examples/swe/ holds five files of its own; ours have distinct
  # names, so this is a merge. Any collision that does appear is backed up once.
  local rel dest
  while IFS= read -r rel; do
    dest="${SWE_EX}/${rel}"
    [[ -e "${dest}" ]] || continue
    [[ -L "${dest}" ]] && continue                      # a previous --mode link
    cmp -s "${REPO}/training/${rel}" "${dest}" && continue
    if [[ -e "${dest}${BACKUP_SUFFIX}" ]]; then continue; fi   # backed up already
    if ((!FORCE)) && ! ours; then
      die "${dest} exists and differs from ours, and no install receipt was found. Pass --force to back it up and overwrite."
    fi
    if ((CHECK)); then plan "back up ${dest} -> ${dest}${BACKUP_SUFFIX}"; else
      cp -p "${dest}" "${dest}${BACKUP_SUFFIX}"
      say "  backed up ${rel} -> ${rel}${BACKUP_SUFFIX}"
    fi
  done < <(cd -- "${REPO}/training" && find . -type f -not -path '*/__pycache__/*' -printf '%P\n')

  if ((CHECK)); then plan "install training/ -> ${SWE_EX} (${MODE})"; return 0; fi
  mkdir -p "${SWE_EX}"
  copy_tree "${REPO}/training" "${SWE_EX}"
  say "  copied training/ -> examples/swe/"
}

install_training_links() {
  # One symlink per top-level entry of training/, so upstream's own files in
  # examples/swe/ survive. A *real* directory at the destination has to be moved
  # out of the way first: `ln -sfn dir dest` would otherwise put the link
  # *inside* it (examples/swe/configs/configs).
  local entry name dest
  ((CHECK)) || mkdir -p "${SWE_EX}"
  for entry in "${REPO}/training"/*; do
    name="$(basename -- "${entry}")"
    dest="${SWE_EX}/${name}"
    if [[ -e "${dest}" && ! -L "${dest}" ]]; then
      if ours; then
        # Our own copy-mode install from an earlier run: replace it outright.
        ((CHECK)) && { plan "replace the copied ${name} with a symlink"; continue; }
        rm -rf "${dest}"
      else
        if ((!FORCE)); then
          die "${dest} exists and is not a symlink. Move it aside, or pass --force to back it up as ${name}${BACKUP_SUFFIX}."
        fi
        if ((CHECK)); then plan "move ${dest} aside to ${dest}${BACKUP_SUFFIX}, then link"; continue; fi
        rm -rf "${dest}${BACKUP_SUFFIX}"
        mv -- "${dest}" "${dest}${BACKUP_SUFFIX}"
        say "  moved ${name} aside -> ${name}${BACKUP_SUFFIX}"
      fi
    fi
    ((CHECK)) && continue
    ln -sfn "${entry}" "${dest}"
  done
  if ((CHECK)); then plan "link training/* -> ${SWE_EX}"; return 0; fi
  say "  linked training/* -> examples/swe/"
}

say "installing SWE-Ladder files"
install_opd
install_training

if ((!CHECK)) && [[ "${MODE}" == copy ]]; then
  cat >"${OPD_DIR}/${RECEIPT_NAME}" <<EOF
# Written by SWE-Ladder integration/install.sh. Delete this file and the
# directory is no longer recognised as ours (re-install then needs --force).
source=${REPO}
mode=${MODE}
swe_master_pin=${SWE_MASTER_PIN}
verl_pin=${VERL_PIN}
installed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
fi

# ----------------------------------------------------------------- 3. verify

if ((CHECK)); then
  say "--check complete: no changes were made"
  exit 0
fi

say "verifying"
for f in rllm/opd/stage_classifier.py rllm/opd/staged_privileged.py rllm/opd/topk_loss.py \
         examples/swe/run_npu_train_from_yaml.sh examples/swe/configs/smoke_stage_routing.yaml; do
  [[ -e "${RLLM}/${f}" ]] || die "expected ${RLLM}/${f} after install"
done
say "  files in place"

# The stage router is the one component every experiment depends on, so verify it
# by running it, not by importing it. rllm/__init__.py is empty upstream and
# swe_ladder imports lazily, so this needs no torch and no training env.
if ! command -v "${PY}" >/dev/null; then
  say "  smoke check skipped: ${PY} not found (set PYTHON_BIN)"
else
  PYTHONPATH="${RLLM}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONNOUSERSITE=1 "${PY}" - <<'PY' || die "stage-router smoke check failed"
from rllm.opd.stage_classifier import StageStateMachine

STEPS = [
    '<function=execute_bash>\n<parameter=cmd>grep -rn "parse_date" src</parameter>\n</function>',
    '<function=file_editor>\n<parameter=command>create</parameter>\n<parameter=path>/testbed/reproduce_issue.py</parameter>\n</function>',
    '<function=execute_bash>\n<parameter=cmd>python /testbed/reproduce_issue.py</parameter>\n</function>',
    '<function=file_editor>\n<parameter=command>str_replace</parameter>\n<parameter=path>/testbed/src/dates.py</parameter>\n</function>',
]
EXPECT = ["causal_localization", "localized_issue_reproduction",
          "localized_issue_reproduction", "patch_generation"]

machine = StageStateMachine(backtrack_threshold=3)
got = [machine.update(step) for step in STEPS]
assert got == EXPECT, f"stage router returned {got}, expected {EXPECT}"
print("[install]   rllm.opd import OK; stage router labels explore/create-repro/run-repro/edit as "
      "LOC/REPRO/REPRO/PATCH")
PY
fi

cat <<EOF

[install] done.

Next:
  1. Site paths:  cp ${REPO}/training/configs/site.env.example ${REPO}/training/configs/site.env
                  \$EDITOR ${REPO}/training/configs/site.env
  2. Render a config (configs ship with {{PLACEHOLDER}} slots, never site paths):
       ${REPO}/scripts/tools/render_config.py \\
         ${REPO}/training/configs/smoke_stage_routing.yaml -o /tmp/run.yaml
  3. Launch:
       bash ${SWE_EX}/run_npu_train_from_yaml.sh --dry-run /tmp/run.yaml   # inspect the env
       bash ${SWE_EX}/run_npu_train_from_yaml.sh /tmp/run.yaml

See ${REPO}/docs/TRAINING.md for the full ladder schedule and
${REPO}/training/configs/README.md for the per-run config table.
EOF
