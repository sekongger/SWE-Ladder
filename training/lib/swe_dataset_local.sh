#!/bin/bash

summarize_dataset() {
  local dataset_path="$1"
  python3 - "$dataset_path" <<'PY'
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
try:
    import polars as pl
except Exception as exc:
    print(f"[dataset] failed to import polars: {exc}")
    raise SystemExit(0)

if not dataset_path.exists():
    print(f"[dataset] missing: {dataset_path}")
    raise SystemExit(0)

df = pl.read_parquet(dataset_path)
print(f"[dataset] path: {dataset_path}")
print(f"[dataset] rows: {df.height}")
print(f"[dataset] columns: {df.columns}")
summary_limit = 8

images = []
instance_ids = []
if "extra_info" in df.columns:
    for row in df["extra_info"].to_list():
        if isinstance(row, dict):
            image = row.get("docker_image")
            if image:
                images.append(image)
            instance_id = row.get("instance_id")
            if instance_id:
                instance_ids.append(instance_id)

if images:
    unique_images = list(dict.fromkeys(images))
    print(f"[dataset] unique docker images: {len(unique_images)}")
    for image in unique_images[:summary_limit]:
        print(f"[dataset] docker_image: {image}")
    if len(unique_images) > summary_limit:
        print(f"[dataset] docker_image: ... ({len(unique_images) - summary_limit} more)")

if instance_ids:
    print(f"[dataset] unique instance_ids: {len(set(instance_ids))}")
    print(f"[dataset] instance_ids preview: {instance_ids[:summary_limit]}")
    if len(instance_ids) > summary_limit:
        print(f"[dataset] instance_ids preview: ... ({len(instance_ids) - summary_limit} more rows)")
PY
}

ensure_pyarrow_readable_parquet() {
  local dataset_path="$1"
  python3 - "$dataset_path" <<'PY'
import shutil
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
if not dataset_path.exists():
    print(f"[dataset] missing: {dataset_path}")
    raise SystemExit(1)

def _pyarrow_check(path: Path) -> tuple[bool, str]:
    try:
        import pyarrow.parquet as pq
        pq.read_table(path)
        return True, "ok"
    except Exception as exc:
        return False, repr(exc)

ok, msg = _pyarrow_check(dataset_path)
if ok:
    print(f"[dataset] pyarrow check passed: {dataset_path}")
    raise SystemExit(0)

print(f"[dataset] pyarrow check failed: {dataset_path}")
print(f"[dataset] pyarrow error: {msg}")
print("[dataset] trying polars rewrite for compatibility ...")

try:
    import polars as pl
except Exception as exc:
    print(f"[dataset] failed to import polars for rewrite: {exc}")
    raise SystemExit(1)

backup_path = dataset_path.with_suffix(dataset_path.suffix + ".bak")
tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".rewrite_tmp")

df = pl.read_parquet(dataset_path)
df.write_parquet(tmp_path, compression="zstd")
shutil.copy2(dataset_path, backup_path)
tmp_path.replace(dataset_path)

ok, msg = _pyarrow_check(dataset_path)
if not ok:
    print(f"[dataset] rewrite failed, pyarrow still cannot read: {msg}")
    raise SystemExit(1)

print(f"[dataset] rewrite succeeded: {dataset_path}")
print(f"[dataset] backup saved: {backup_path}")
PY
}

prepull_dataset_images() {
  local dataset_path="$1"
  local -a images=()
  local image
  local strict_prepull="${STRICT_DOCKER_PREPULL:-0}"
  if ! command -v docker >/dev/null 2>&1; then
    echo "[docker] docker command not found, skip prepull"
    return 0
  fi
  mapfile -t images < <(python3 - "$dataset_path" <<'PY'
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
try:
    import polars as pl
except Exception:
    raise SystemExit(0)

if not dataset_path.exists():
    raise SystemExit(0)

df = pl.read_parquet(dataset_path)
seen = set()
if "extra_info" in df.columns:
    for row in df["extra_info"].to_list():
        if isinstance(row, dict):
            image = row.get("docker_image")
            if image and image not in seen:
                seen.add(image)
                print(image)
PY
)
  if [[ "${#images[@]}" -eq 0 ]]; then
    echo "[docker] no docker images found in dataset metadata"
    return 0
  fi
  for image in "${images[@]}"; do
    echo "[docker] ensure image: ${image}"
    if docker image inspect "${image}" >/dev/null 2>&1; then
      continue
    fi
    if ! docker pull "${image}"; then
      echo "[docker] WARNING: failed to pull ${image}; training will retry at runtime if possible"
      if [[ "${strict_prepull}" == "1" && "${R2EGYM_DOCKER_MODE:-local}" == "local" ]]; then
        echo "[docker] ERROR: strict local prepull is enabled and docker pull failed for ${image}"
        exit 1
      fi
    fi
  done
}

filter_dataset_to_local_images() {
  local input_dataset="$1"
  local output_dataset="$2"
  python3 - "$input_dataset" "$output_dataset" <<'PY'
import subprocess
import sys
from pathlib import Path

import polars as pl

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.exists():
    print(f"[dataset] missing: {src}")
    raise SystemExit(1)

df = pl.read_parquet(src)
if "extra_info" not in df.columns:
    print(f"[dataset] no extra_info column in {src}, skip local-image filtering")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dst)
    raise SystemExit(0)

available_images = set()
keep_mask = []
for row in df["extra_info"].to_list():
    image = None
    if isinstance(row, dict):
        image = row.get("docker_image")
    if not image:
        keep_mask.append(False)
        continue
    if image not in available_images:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            available_images.add(image)
    keep_mask.append(image in available_images)

filtered = df.filter(pl.Series("keep", keep_mask))
dst.parent.mkdir(parents=True, exist_ok=True)
filtered.write_parquet(dst)
print(f"[dataset] local-image filtered src: {src}")
print(f"[dataset] local-image filtered dst: {dst}")
print(f"[dataset] rows kept: {filtered.height}/{df.height}")
kept_images = sorted(available_images)
print(f"[dataset] kept docker images: {len(kept_images)}")
for image in kept_images[:8]:
    print(f"[dataset] kept docker_image: {image}")
if len(kept_images) > 8:
    print(f"[dataset] kept docker_image: ... ({len(kept_images) - 8} more)")
PY
}

dataset_row_count() {
  local dataset_path="$1"
  python3 - "$dataset_path" <<'PY'
import sys
from pathlib import Path
import polars as pl

path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    print(pl.read_parquet(path).height)
PY
}

assert_local_image_arch_compatibility() {
  local dataset_path="$1"
  if [[ "${R2EGYM_DOCKER_MODE:-local}" != "local" ]]; then
    return 0
  fi
  if [[ "${ALLOW_CROSS_ARCH_LOCAL_IMAGES:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "$(uname -m)" != "aarch64" ]]; then
    return 0
  fi

  python3 - "$dataset_path" <<'PY'
import subprocess
import sys
from pathlib import Path

import polars as pl

src = Path(sys.argv[1])
if not src.exists():
    raise SystemExit(0)

df = pl.read_parquet(src)
if "extra_info" not in df.columns:
    raise SystemExit(0)

images = []
for row in df["extra_info"].to_list():
    if isinstance(row, dict):
        image = row.get("docker_image")
        if image and image not in images:
            images.append(image)

incompatible: list[tuple[str, str]] = []
for image in images[:32]:
    try:
        arch = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Architecture}}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        continue
    if arch in {"amd64", "x86_64"} or "x86_64" in image:
        incompatible.append((image, arch or "unknown"))

if incompatible:
    print("[dataset] ERROR: local docker mode on aarch64 cannot run x86_64/amd64 SWE images.")
    for image, arch in incompatible[:8]:
        print(f"[dataset] incompatible image: {image} (arch={arch})")
    print("[dataset] Use a remote x86 Docker host, e.g. set R2EGYM_DOCKER_MODE=remote and R2EGYM_DOCKER_HOST=tcp://<x86-host>:2375")
    raise SystemExit(1)
PY
}

normalize_divisible_value() {
  local name="$1"
  local value="$2"
  local divisor="$3"
  local normalized
  if (( divisor <= 0 )); then
    echo "ERROR: divisor for ${name} must be > 0, got ${divisor}"
    exit 1
  fi
  if (( value <= 0 )); then
    echo "ERROR: ${name} must be > 0, got ${value}"
    exit 1
  fi
  if (( value % divisor == 0 )); then
    printf '%s' "${value}"
    return 0
  fi
  if (( value < divisor )); then
    # Small smoke-test / triage batches are valid even when they do not fill every data-parallel rank.
    printf '%s' "${value}"
    return 0
  fi
  normalized=$(( value / divisor * divisor ))
  if (( normalized == 0 )); then
    normalized="${divisor}"
  fi
  echo "Adjust ${name} from ${value} to ${normalized} so it is divisible by TRAIN_N_GPUS_PER_NODE=${divisor}." >&2
  printf '%s' "${normalized}"
}

gcd_int() {
  local a="$1"
  local b="$2"
  local tmp
  while (( b != 0 )); do
    tmp=$(( a % b ))
    a="${b}"
    b="${tmp}"
  done
  printf '%s' "${a}"
}

normalize_actor_ppo_mini_batch_size() {
  local value="$1"
  local rollout_n="$2"
  local training_world_size="$3"
  local ulysses_sp_size="$4"
  local effective_dp
  local gcd_value
  local quantum
  local normalized

  if (( value <= 0 || rollout_n <= 0 || training_world_size <= 0 || ulysses_sp_size <= 0 )); then
    echo "ERROR: invalid PPO mini-batch normalization inputs value=${value} rollout_n=${rollout_n} training_world_size=${training_world_size} ulysses_sp_size=${ulysses_sp_size}." >&2
    exit 1
  fi

  effective_dp=$(( training_world_size / ulysses_sp_size ))
  if (( effective_dp <= 1 )); then
    printf '%s' "${value}"
    return 0
  fi

  gcd_value="$(gcd_int "${effective_dp}" "${rollout_n}")"
  quantum=$(( effective_dp / gcd_value ))
  normalized="${value}"

  if (( normalized < quantum )); then
    normalized="${quantum}"
  elif (( normalized % quantum != 0 )); then
    normalized=$(( (normalized + quantum - 1) / quantum * quantum ))
  fi

  if (( normalized != value )); then
    echo "Adjust PPO_MINI_BATCH_SIZE from ${value} to ${normalized} so PPO_MINI_BATCH_SIZE*ROLLOUT_N is divisible by effective actor data parallel size ${effective_dp}." >&2
  fi

  printf '%s' "${normalized}"
}

normalize_train_batch_size_for_ppo() {
  local train_batch_size="$1"
  local ppo_mini_batch_size="$2"
  local normalized="${train_batch_size}"

  if (( train_batch_size <= 0 || ppo_mini_batch_size <= 0 )); then
    echo "ERROR: invalid train/ppo mini batch sizes train_batch_size=${train_batch_size} ppo_mini_batch_size=${ppo_mini_batch_size}." >&2
    exit 1
  fi

  if (( normalized < ppo_mini_batch_size )); then
    normalized="${ppo_mini_batch_size}"
  elif (( normalized % ppo_mini_batch_size != 0 )); then
    normalized=$(( (normalized + ppo_mini_batch_size - 1) / ppo_mini_batch_size * ppo_mini_batch_size ))
  fi

  if (( normalized != train_batch_size )); then
    echo "Adjust TRAIN_BATCH_SIZE from ${train_batch_size} to ${normalized} so it is divisible by PPO_MINI_BATCH_SIZE=${ppo_mini_batch_size}." >&2
  fi

  printf '%s' "${normalized}"
}

prepare_local_image_filtered_datasets() {
  local train_src="$1"
  local val_src="$2"
  local filtered_root="$3"
  local filtered_train="${filtered_root}/train_local_images.parquet"
  local filtered_val="${filtered_root}/val_local_images.parquet"
  local filtered_rows

  filter_dataset_to_local_images "${train_src}" "${filtered_train}"
  if [[ "${val_src}" == "${train_src}" ]]; then
    filtered_val="${filtered_train}"
  else
    filter_dataset_to_local_images "${val_src}" "${filtered_val}"
  fi

  filtered_rows="$(dataset_row_count "${filtered_train}")"
  echo "[dataset] filtered train rows: ${filtered_rows}"
  if [[ "${filtered_rows}" -le 0 ]]; then
    echo "[dataset] ERROR: no local docker images match the dataset."
    exit 1
  fi

  TRAIN_DATASET="${filtered_train}"
  VAL_DATASET="${filtered_val}"

  assert_local_image_arch_compatibility "${TRAIN_DATASET}"

  if [[ "${filtered_rows}" -lt "${TRAIN_N_GPUS_PER_NODE}" ]]; then
    TRAIN_N_GPUS_PER_NODE="${filtered_rows}"
  fi
  if [[ "${TRAIN_N_GPUS_PER_NODE}" -lt 1 ]]; then
    TRAIN_N_GPUS_PER_NODE=1
  fi

  if [[ "${filtered_rows}" -lt "${TRAIN_BATCH_SIZE}" ]]; then
    TRAIN_BATCH_SIZE="${filtered_rows}"
  fi
  if [[ "${filtered_rows}" -lt "${VAL_BATCH_SIZE}" ]]; then
    VAL_BATCH_SIZE="${filtered_rows}"
  fi
  if [[ "${filtered_rows}" -lt "${PPO_MINI_BATCH_SIZE}" ]]; then
    PPO_MINI_BATCH_SIZE="${filtered_rows}"
  fi

  TRAIN_BATCH_SIZE="$(normalize_divisible_value TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}" "${TRAIN_N_GPUS_PER_NODE}")"
  VAL_BATCH_SIZE="$(normalize_divisible_value VAL_BATCH_SIZE "${VAL_BATCH_SIZE}" "${TRAIN_N_GPUS_PER_NODE}")"
  PPO_MINI_BATCH_SIZE="$(normalize_divisible_value PPO_MINI_BATCH_SIZE "${PPO_MINI_BATCH_SIZE}" "${TRAIN_N_GPUS_PER_NODE}")"

  if [[ "${ROLLOUT_TP_SIZE}" -gt "${TRAIN_N_GPUS_PER_NODE}" ]]; then
    ROLLOUT_TP_SIZE="${TRAIN_N_GPUS_PER_NODE}"
  fi

  echo "[dataset] using filtered TRAIN_DATASET: ${TRAIN_DATASET}"
  echo "[dataset] adjusted TRAIN_N_GPUS_PER_NODE: ${TRAIN_N_GPUS_PER_NODE}"
  echo "[dataset] adjusted TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
  echo "[dataset] adjusted PPO_MINI_BATCH_SIZE: ${PPO_MINI_BATCH_SIZE}"
  echo "[dataset] adjusted ROLLOUT_TP_SIZE: ${ROLLOUT_TP_SIZE}"
}
