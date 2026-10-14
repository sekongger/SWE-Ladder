"""Convert the staged JSONL datasets into the verl parquet files training reads.

One row per (instance, stage). `extra_info` carries everything the rollout and the
teacher need: the Docker image to run, the student messages, the per-stage
privileged evidence (`teacher_pi`) and the pre-rendered `teacher_messages`.

  python3 scripts/data/build_staged_verl_dataset.py \\
      --stages-root stages --stage all --splits train,valid \\
      --output-root "$DATA_ROOT/my_family" \\
      --swesmith-data-root ... --swerebench-data-root ... \\
      --r2e-gym-data-root ... --swe-gym-data-root ...

The `--*-data-root` flags are how a row gets an executable Docker image: each
upstream dataset names its images differently, so the image is matched per source
by `instance_id`. Rows with no match keep `--generic-image`, which can only run
the non-executing stages. See `data/README.md`.

Needs the rllm checkout importable (`PYTHONPATH=<rllm>:<r2egym>/src`): the student
system prompt is baked into every row and comes from `rllm.agents.system_prompts`.
"""
from __future__ import annotations

import argparse
import collections
import copy
import importlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

# This repository's `swe_ladder/` is installed into the training tree as
# `rllm.opd`; the two names are the same module, so either resolution is correct.
# The import is deferred to run time (see `_load_staged_dataset_api`) so that
# `--help` works on a plain Python: `staged_dataset` reaches into rllm for the
# student system prompt it bakes into every row.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SUPPORTED_SPLITS: tuple[str, ...] = ()
SUPPORTED_STAGES: tuple[str, ...] = ()
build_teacher_messages = None
load_staged_opd_dataset = None


def _load_staged_dataset_api() -> None:
    """Bind the four staged-dataset symbols, or exit naming the thing to install."""
    global SUPPORTED_SPLITS, SUPPORTED_STAGES, build_teacher_messages, load_staged_opd_dataset
    missing = None
    for module_name in ("swe_ladder.staged_dataset", "rllm.opd.staged_dataset"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = missing or exc.name
            continue
        SUPPORTED_SPLITS = module.SUPPORTED_SPLITS
        SUPPORTED_STAGES = module.SUPPORTED_STAGES
        build_teacher_messages = module.build_teacher_messages
        load_staged_opd_dataset = module.load_staged_opd_dataset
        return
    raise SystemExit(
        f"cannot import staged_dataset: no module named {missing!r}. This builder bakes the "
        "student system prompt into every row and so needs the rllm checkout importable: run it "
        "with PYTHONPATH=<rllm>:<r2egym>/src, or inside the environment scripts/setup/ builds."
    )


DEFAULT_GENERIC_IMAGE = "skongger/r2egym-scaffold-base:latest"

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm may be absent in minimal data envs.
    tqdm = None


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", ""))
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _codebase_access(sample: dict[str, Any]) -> dict[str, Any]:
    student_input = sample.get("student_input") or {}
    access = student_input.get("codebase_access") or {}
    if not isinstance(access, dict):
        return {}
    return access


def _repo_leaf(repo: str) -> str:
    return repo.rstrip("/").split("/")[-1]


def _plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [value]
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _plain_list(value) if str(item)]


def _looks_like_test_id(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text in {"[", "]"}:
        return False
    if re.fullmatch(r"\[\d+%]", text):
        return False
    if text.lower().startswith(("collected ", "passed", "failed", "error")):
        return False
    return "::" in text or ".py" in text or text.startswith("test_")


def _test_id_list(value: Any) -> list[str]:
    """Keep executable pytest-style ids and drop progress/log tokens."""

    return [item.strip() for item in _string_list(value) if _looks_like_test_id(item)]


def _first_text(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _reverse_unified_diff(patch: str) -> str:
    """Return a repair-oriented diff for SWE-smith bug-introducing patches."""

    if not patch:
        return patch

    def _reverse_hunk_header(line: str) -> str:
        match = re.match(
            r"^@@ -(?P<old>\d+(?:,\d+)?) \+(?P<new>\d+(?:,\d+)?)(?P<rest> @@.*)$",
            line,
        )
        if not match:
            return line
        return f"@@ -{match.group('new')} +{match.group('old')}{match.group('rest')}"

    lines = patch.splitlines(keepends=True)
    reversed_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        newline = "\n" if line.endswith("\n") else ""
        raw = line[:-1] if newline else line

        if raw.startswith("--- ") and i + 1 < len(lines):
            next_line = lines[i + 1]
            next_newline = "\n" if next_line.endswith("\n") else ""
            next_raw = next_line[:-1] if next_newline else next_line
            if next_raw.startswith("+++ "):
                reversed_lines.append("--- " + next_raw[4:] + newline)
                reversed_lines.append("+++ " + raw[4:] + next_newline)
                i += 2
                continue

        if raw.startswith("@@ "):
            reversed_lines.append(_reverse_hunk_header(raw) + newline)
        elif raw.startswith("-") and not raw.startswith("---"):
            reversed_lines.append("+" + raw[1:] + newline)
        elif raw.startswith("+") and not raw.startswith("+++"):
            reversed_lines.append("-" + raw[1:] + newline)
        else:
            reversed_lines.append(line)
        i += 1

    return "".join(reversed_lines)


def _offline_swerebench_install_config(value: Any) -> Any:
    """Avoid network-only requirements lookups while preserving eval-script settings."""

    if not isinstance(value, dict):
        return value
    config = copy.deepcopy(value)
    if config.get("packages") in {"requirements.txt", "environment.yml"}:
        config["packages"] = ""
    config["reqs_path"] = None
    config["env_yml_path"] = None
    return config


def _resolve_swesmith_data_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "data").is_dir():
        return root / "data"
    return root


def _read_parquet_records(path: str | Path, *, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Read selected parquet columns, using polars for parquet files pyarrow cannot decode."""

    try:
        return pd.read_parquet(path, columns=list(columns)).to_dict("records")
    except Exception as pandas_error:
        try:
            import polars as pl
        except ImportError:
            raise pandas_error
        return pl.read_parquet(path, columns=list(columns)).to_dicts()


def _read_parquet_records_available(path: str | Path, *, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Read only columns present in a metadata parquet file."""

    requested = list(dict.fromkeys(columns))
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        selected = [column for column in requested if column in available]
    except Exception:
        selected = requested
    if not selected:
        return []
    try:
        return pd.read_parquet(path, columns=selected).to_dict("records")
    except Exception as pandas_error:
        try:
            import polars as pl
        except ImportError:
            raise pandas_error
        try:
            return pl.read_parquet(path, columns=selected).to_dicts()
        except Exception:
            frame = pl.read_parquet(path)
            selected = [column for column in selected if column in frame.columns]
            return frame.select(selected).to_dicts() if selected else []


def _metadata_files(path: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(path)
    if root.is_file():
        if root.suffix == ".jsonl":
            return [root], []
        if root.suffix == ".parquet":
            return [], [root]
    jsonl_files: list[Path] = []
    canonical = root / "canonical" / "tasks.jsonl"
    if canonical.exists():
        jsonl_files.append(canonical)
    elif (root / "tasks.jsonl").exists():
        jsonl_files.append(root / "tasks.jsonl")
    parquet_root = root / "data" if (root / "data").is_dir() else root
    parquet_files = sorted(parquet_root.glob("*.parquet"))
    return jsonl_files, parquet_files


def _is_invalid_swe_gym_version(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or text in {"swe_gym", "swegym", "swe-gym"}


def _path_variants(path: Path) -> set[Path]:
    variants = {path}
    try:
        variants.add(path.resolve())
    except OSError:
        pass
    return variants


def _candidate_swe_gym_original_roots(primary_path: str | Path) -> list[Path]:
    """Find local raw SWE-Gym metadata roots used to restore dropped fields."""

    primary = Path(primary_path)
    primary_variants = _path_variants(primary)
    bases: list[Path] = []
    if primary.is_file():
        bases.append(primary.parent)
        bases.extend(primary.parent.parents)
    else:
        bases.append(primary)
        bases.extend(primary.parents)
    bases.append(Path.cwd())
    bases.extend(Path.cwd().parents)
    bases.extend(Path(__file__).resolve().parents)

    candidates: list[Path] = []
    for base in bases:
        for relative in ("SWE-Gym/data", "SWE-Gym", "swe-gym-subset"):
            candidate = base / relative
            if not candidate.exists():
                continue
            if _path_variants(candidate) & primary_variants:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _merge_swe_gym_original_metadata(
    metadata: dict[str, dict[str, Any]],
    *,
    primary_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Restore SWE-Gym fields, especially version, from raw HF metadata."""

    if not metadata:
        return metadata
    if not any(_is_invalid_swe_gym_version(item.get("version")) for item in metadata.values()):
        return metadata

    for candidate in _candidate_swe_gym_original_roots(primary_path):
        try:
            original = _load_gym_metadata(candidate, source_dataset="swe_gym", merge_swe_gym_original=False)
        except (FileNotFoundError, OSError, ValueError):
            continue
        restored = 0
        for task_id, item in metadata.items():
            original_item = original.get(task_id)
            if not original_item:
                continue
            merged = dict(item)
            for key, value in original_item.items():
                if key == "source_dataset":
                    continue
                if key == "version":
                    if _is_invalid_swe_gym_version(merged.get("version")) and not _is_invalid_swe_gym_version(value):
                        merged["version"] = str(value)
                        restored += 1
                    continue
                if not merged.get(key) and value not in (None, "", [], {}):
                    merged[key] = value
            metadata[task_id] = merged
        if not any(_is_invalid_swe_gym_version(item.get("version")) for item in metadata.values()):
            break
        if restored:
            break
    return metadata


def _docker_api_base_url(docker_host: str) -> str:
    host = docker_host.strip().rstrip("/")
    if not host:
        return ""
    if host.startswith("tcp://"):
        return "http://" + host[len("tcp://") :]
    if host.startswith(("http://", "https://")):
        return host
    if "://" not in host:
        return "http://" + host
    raise ValueError(f"Unsupported Docker host for image inference: {docker_host!r}")


def _image_tags_from_docker_api_payload(payload: Any) -> list[str]:
    tags: list[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("tags"), list):
            payload = [{"RepoTags": payload["tags"]}]
        elif isinstance(payload.get("RepoTags"), list):
            payload = [payload]
        else:
            payload = []
    if not isinstance(payload, list):
        return tags
    for image in payload:
        if isinstance(image, str):
            tags.append(image)
            continue
        if not isinstance(image, dict):
            continue
        for tag in image.get("RepoTags") or []:
            tag = str(tag)
            if tag and tag != "<none>:<none>":
                tags.append(tag)
    return list(dict.fromkeys(tags))


def load_docker_image_tags_from_host(docker_host: str, *, timeout: float = 30.0) -> list[str]:
    """Read Docker image tags through the HTTP Docker API; no docker CLI is required."""

    base_url = _docker_api_base_url(docker_host)
    if not base_url:
        return []
    with urllib.request.urlopen(f"{base_url}/images/json?all=1", timeout=timeout) as response:
        payload = json.loads(response.read())
    return _image_tags_from_docker_api_payload(payload)


def load_docker_image_tags_from_file(path: str | Path) -> list[str]:
    """Load Docker image tags from a text file or a saved /images/json response."""

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        return _image_tags_from_docker_api_payload(json.loads(text))
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def infer_swe_gym_docker_image(
    *,
    task_id: str,
    repo: str,
    image_tags: Sequence[str],
) -> str:
    """Infer SumanthRH/SWE-Gym-Subset runtime image from downloaded Docker image tags."""

    task_id = str(task_id or "").strip()
    repo = str(repo or "").strip()
    if not task_id or not image_tags:
        return ""

    owner = ""
    if "/" in repo:
        owner = repo.split("/", 1)[0]
    elif "__" in task_id:
        owner = task_id.split("__", 1)[0]

    issue_id = task_id.split("__", 1)[1] if "__" in task_id else task_id
    candidates: list[str] = []
    if owner and issue_id:
        # SWE-Gym-Subset images downloaded on the Docker host use this naming:
        # xingyaoww/sweb.eval.x86_64.<owner>_s_<repo-or-issue>:latest
        candidates.append(f"xingyaoww/sweb.eval.x86_64.{owner}_s_{issue_id}:latest")
        candidates.append(f"swebench/sweb.eval.x86_64.{owner}_1776_{issue_id}:latest")
        candidates.append(f"swerebench/sweb.eval.x86_64.{owner}_1776_{issue_id}:latest")
    if "__" in task_id:
        candidates.append(f"xingyaoww/sweb.eval.x86_64.{task_id.replace('__', '_s_')}:latest")

    tag_by_lower = {str(tag).lower(): str(tag) for tag in image_tags if str(tag)}
    for candidate in dict.fromkeys(candidates):
        image = tag_by_lower.get(candidate.lower())
        if image:
            return image

    suffixes = []
    if owner and issue_id:
        suffixes.extend(
            [
                f".{owner}_s_{issue_id}:latest",
                f".{owner}_1776_{issue_id}:latest",
                f".{owner}_s_{issue_id}",
                f".{owner}_1776_{issue_id}",
            ]
        )
    matches = [
        str(tag)
        for tag in image_tags
        if any(str(tag).lower().endswith(suffix.lower()) for suffix in suffixes)
    ]
    matches = list(dict.fromkeys(matches))
    return matches[0] if len(matches) == 1 else ""


def enrich_swe_gym_metadata_with_docker_images(
    metadata: dict[str, dict[str, Any]],
    *,
    docker_host: str | None = None,
    image_list_path: str | Path | None = None,
    image_tags: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Attach missing SWE-Gym docker_image/image_name fields from host image tags."""

    tags: list[str] = []
    if image_tags:
        tags.extend(str(tag) for tag in image_tags)
    if image_list_path is not None:
        tags.extend(load_docker_image_tags_from_file(image_list_path))
    if docker_host:
        tags.extend(load_docker_image_tags_from_host(docker_host))
    tags = list(dict.fromkeys(tag for tag in tags if tag))
    if not tags:
        return metadata

    enriched: dict[str, dict[str, Any]] = {}
    for task_id, item in metadata.items():
        copied = dict(item)
        existing_image = str(copied.get("docker_image") or copied.get("image_name") or "")
        if not existing_image or existing_image == DEFAULT_GENERIC_IMAGE:
            inferred_image = infer_swe_gym_docker_image(
                task_id=task_id,
                repo=str(copied.get("repo") or ""),
                image_tags=tags,
            )
            if inferred_image:
                copied["docker_image"] = inferred_image
                copied["image_name"] = inferred_image
        enriched[task_id] = copied
    return enriched


def _iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_swesmith_image_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load instance_id -> SWE-smith image metadata from local SWE-smith parquet files."""

    data_root = _resolve_swesmith_data_root(path)
    parquet_files = sorted(data_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No SWE-smith parquet files found under {data_root}")

    columns = ["instance_id", "image_name", "repo", "FAIL_TO_PASS", "PASS_TO_PASS"]
    metadata: dict[str, dict[str, Any]] = {}
    iterator = parquet_files
    if tqdm is not None:
        iterator = tqdm(parquet_files, desc="Loading SWE-smith image metadata")
    for parquet_file in iterator:
        for row in _read_parquet_records(parquet_file, columns=columns):
            instance_id = str(row.get("instance_id") or "")
            image_name = str(row.get("image_name") or "")
            if not instance_id or not image_name:
                continue
            metadata[instance_id] = {
                "image_name": image_name,
                "repo": str(row.get("repo") or ""),
                "FAIL_TO_PASS": _test_id_list(row.get("FAIL_TO_PASS")),
                "PASS_TO_PASS": _test_id_list(row.get("PASS_TO_PASS")),
            }
    return metadata


GYM_METADATA_COLUMNS = [
    "task_id",
    "instance_id",
    "id",
    "source_dataset",
    "repo",
    "repo_name",
    "base_commit",
    "commit_hash",
    "docker_image",
    "image_name",
    "patch",
    "test_patch",
    "problem_statement",
    "prompt",
    "hints_text",
    "version",
    "eval_script",
    "FAIL_TO_PASS",
    "FAIL_TO_FAIL",
    "PASS_TO_PASS",
    "PASS_TO_FAIL",
    "fail_to_pass",
    "pass_to_pass",
    "fail_to_pass_test_code",
    "fail_to_pass_test_ids",
    "fail_to_pass_test_names",
    "parsed_commit",
    "parsed_commit_content",
    "execution_result_content",
    "expected_output_json",
    "requirements",
    "environment",
    "install_config",
]


def _r2e_task_ids(row: dict[str, Any]) -> list[str]:
    explicit = _first_text(row, "task_id", "instance_id", "id")
    ids: list[str] = [explicit] if explicit else []
    repo = _first_text(row, "repo_name", "repo")
    commit = _first_text(row, "commit_hash", "base_commit")
    if repo and commit:
        ids.extend([f"{repo}__{commit[:12]}", f"{repo}__{commit}"])
    deduped: list[str] = []
    for task_id in ids:
        if task_id and task_id not in deduped:
            deduped.append(task_id)
    return deduped


def _gym_metadata_from_row(row: dict[str, Any], *, source_dataset: str) -> tuple[list[str], dict[str, Any]]:
    expected_output = _json_dict(row.get("expected_output_json"))
    task_ids = _r2e_task_ids(row) if source_dataset == "r2e_gym" else [_first_text(row, "task_id", "instance_id", "id")]
    task_ids = [task_id for task_id in task_ids if task_id]
    repo = _first_text(row, "repo", "repo_name")
    base_commit = _first_text(row, "base_commit", "commit_hash")
    docker_image = _first_text(row, "docker_image", "image_name")
    fail_to_pass = _test_id_list(row.get("FAIL_TO_PASS") or row.get("fail_to_pass"))
    pass_to_pass = _test_id_list(row.get("PASS_TO_PASS") or row.get("pass_to_pass"))
    metadata = {
        "source_dataset": source_dataset,
        "repo": repo,
        "repo_name": _first_text(row, "repo_name", "repo") or repo,
        "base_commit": base_commit,
        "environment_setup_commit": _first_text(row, "environment_setup_commit") or base_commit,
        "hints_text": _first_text(row, "hints_text"),
        "patch": _first_text(row, "patch") or _first_text(expected_output, "patch", "diff"),
        "test_patch": _first_text(row, "test_patch"),
        "problem_statement": _first_text(row, "problem_statement", "prompt"),
        "version": _first_text(row, "version") or source_dataset,
        "install_config": row.get("install_config"),
        "requirements": _first_text(row, "requirements"),
        "environment": _first_text(row, "environment"),
        "FAIL_TO_PASS": fail_to_pass,
        "FAIL_TO_FAIL": _test_id_list(row.get("FAIL_TO_FAIL")),
        "PASS_TO_PASS": pass_to_pass,
        "PASS_TO_FAIL": _test_id_list(row.get("PASS_TO_FAIL")),
        "docker_image": docker_image,
        "image_name": _first_text(row, "image_name") or docker_image,
        "run_tests": _first_text(row, "eval_script", "run_tests"),
        "parsed_commit": _first_text(row, "parsed_commit", "parsed_commit_content") or "{}",
        "parsed_commit_content": _first_text(row, "parsed_commit_content", "parsed_commit") or "{}",
        "execution_result_content": _first_text(row, "execution_result_content"),
        "expected_output_json": _first_text(row, "expected_output_json"),
        "fail_to_pass_test_code": _plain_list(row.get("fail_to_pass_test_code")),
        "fail_to_pass_test_ids": _test_id_list(row.get("fail_to_pass_test_ids")) or fail_to_pass,
        "fail_to_pass_test_names": _test_id_list(row.get("fail_to_pass_test_names")) or fail_to_pass,
    }
    return task_ids, metadata


def _load_gym_metadata(
    path: str | Path,
    *,
    source_dataset: str,
    merge_swe_gym_original: bool = True,
) -> dict[str, dict[str, Any]]:
    jsonl_files, parquet_files = _metadata_files(path)
    if not jsonl_files and not parquet_files:
        raise FileNotFoundError(f"No {source_dataset} metadata jsonl/parquet files found under {path}")

    metadata: dict[str, dict[str, Any]] = {}
    iterator: Iterable[Path] = [*jsonl_files, *parquet_files]
    if tqdm is not None:
        iterator = tqdm(list(iterator), desc=f"Loading {source_dataset} metadata")
    for metadata_file in iterator:
        if metadata_file.suffix == ".jsonl":
            rows = list(_iter_jsonl_records(metadata_file))
        else:
            rows = _read_parquet_records_available(metadata_file, columns=GYM_METADATA_COLUMNS)
        for row in rows:
            task_ids, item = _gym_metadata_from_row(row, source_dataset=source_dataset)
            for task_id in task_ids:
                metadata[task_id] = item
    if source_dataset == "swe_gym" and merge_swe_gym_original:
        metadata = _merge_swe_gym_original_metadata(metadata, primary_path=path)
    return metadata


def load_r2e_gym_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load task metadata from R2E-Gym/R2E-Gym-Subset parquet or local canonical JSONL."""

    return _load_gym_metadata(path, source_dataset="r2e_gym")


def load_swe_gym_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load task metadata from SumanthRH/SWE-Gym-Subset parquet or local canonical JSONL."""

    return _load_gym_metadata(path, source_dataset="swe_gym")


def load_swerebench_image_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load instance_id -> SWE-rebench runtime metadata from local SWE-rebench parquet files."""

    data_root = _resolve_swesmith_data_root(path)
    parquet_files = sorted(data_root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No SWE-rebench parquet files found under {data_root}")

    columns = [
        "instance_id",
        "base_commit",
        "environment_setup_commit",
        "hints_text",
        "patch",
        "problem_statement",
        "repo",
        "test_patch",
        "version",
        "install_config",
        "requirements",
        "environment",
        "FAIL_TO_PASS",
        "FAIL_TO_FAIL",
        "PASS_TO_PASS",
        "PASS_TO_FAIL",
        "docker_image",
        "image_name",
    ]
    metadata: dict[str, dict[str, Any]] = {}
    iterator = parquet_files
    if tqdm is not None:
        iterator = tqdm(parquet_files, desc="Loading SWE-rebench image metadata")
    for parquet_file in iterator:
        for row in _read_parquet_records(parquet_file, columns=columns):
            instance_id = str(row.get("instance_id") or "")
            docker_image = str(row.get("docker_image") or row.get("image_name") or "")
            if not instance_id or not docker_image:
                continue
            metadata[instance_id] = {
                "base_commit": str(row.get("base_commit") or ""),
                "environment_setup_commit": str(row.get("environment_setup_commit") or ""),
                "hints_text": str(row.get("hints_text") or ""),
                "patch": str(row.get("patch") or ""),
                "problem_statement": str(row.get("problem_statement") or ""),
                "repo": str(row.get("repo") or ""),
                "test_patch": str(row.get("test_patch") or ""),
                "version": str(row.get("version") or ""),
                "install_config": _offline_swerebench_install_config(row.get("install_config")),
                "requirements": str(row.get("requirements") or ""),
                "environment": str(row.get("environment") or ""),
                "FAIL_TO_PASS": _test_id_list(row.get("FAIL_TO_PASS")),
                "FAIL_TO_FAIL": _test_id_list(row.get("FAIL_TO_FAIL")),
                "PASS_TO_PASS": _test_id_list(row.get("PASS_TO_PASS")),
                "PASS_TO_FAIL": _test_id_list(row.get("PASS_TO_FAIL")),
                "docker_image": docker_image,
                "image_name": str(row.get("image_name") or docker_image),
            }
    return metadata


def _infer_source_dataset_from_image(image: str, access: dict[str, Any] | None = None) -> str:
    explicit = str((access or {}).get("source_dataset") or "").strip()
    if explicit:
        return explicit
    lowered = image.lower()
    if "swesmith" in lowered:
        return "swe_smith"
    if "swerebench" in lowered:
        return "swe_rebench"
    if "xingyaoww/sweb.eval.x86_64." in lowered:
        return "swe_gym"
    if "namanjain12/" in lowered or "_final:" in lowered:
        return "r2e_gym"
    return "generic_sandbox"


def _runtime_metadata_for_task(
    task_id: str,
    *,
    swesmith_metadata: dict[str, dict[str, Any]] | None = None,
    swerebench_metadata: dict[str, dict[str, Any]] | None = None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None = None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    if task_id in (swesmith_metadata or {}):
        return "swe_smith", (swesmith_metadata or {})[task_id]
    if task_id in (swerebench_metadata or {}):
        return "swe_rebench", (swerebench_metadata or {})[task_id]
    if task_id in (r2e_gym_metadata or {}):
        return "r2e_gym", (r2e_gym_metadata or {})[task_id]
    if task_id in (swe_gym_metadata or {}):
        return "swe_gym", (swe_gym_metadata or {})[task_id]
    return "", None


def _fail_to_pass_from_reproduction_tests(student_input: dict[str, Any]) -> list[str]:
    reproduction_tests = student_input.get("reproduction_tests")
    if not isinstance(reproduction_tests, dict):
        return []
    source = str(reproduction_tests.get("source") or "").upper()
    if source != "FAIL_TO_PASS":
        return []
    return _test_id_list(reproduction_tests.get("tests"))


def _extra_info_from_sample(
    sample: dict[str, Any],
    *,
    generic_image: str,
    generic_container_name: str | None = None,
    swesmith_metadata: dict[str, dict[str, Any]] | None = None,
    swerebench_metadata: dict[str, dict[str, Any]] | None = None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None = None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    access = _codebase_access(sample)
    student_input = sample.get("student_input") or {}
    teacher_pi = sample.get("teacher_pi") or {}
    repo = access.get("repo") or access.get("checkout_repo") or ""
    task_id = sample.get("task_id") or ""
    base_commit = access.get("base_commit") or access.get("checkout_ref") or ""
    source_dataset, image_info = _runtime_metadata_for_task(
        task_id,
        swesmith_metadata=swesmith_metadata,
        swerebench_metadata=swerebench_metadata,
        r2e_gym_metadata=r2e_gym_metadata,
        swe_gym_metadata=swe_gym_metadata,
    )
    swesmith_info = image_info if source_dataset == "swe_smith" else None
    swerebench_info = image_info if source_dataset == "swe_rebench" else None
    r2e_gym_info = image_info if source_dataset == "r2e_gym" else None
    swe_gym_info = image_info if source_dataset == "swe_gym" else None
    staged_problem_statement = _messages_to_text(sample["student_messages"])
    access_docker_image = str(access.get("docker_image") or access.get("image_name") or "")
    docker_image = (
        str((image_info or {}).get("docker_image") or (image_info or {}).get("image_name") or "")
        or access_docker_image
        or generic_image
    )
    if not source_dataset:
        source_dataset = _infer_source_dataset_from_image(access_docker_image, access) if access_docker_image else "generic_sandbox"
    repo_name = str((image_info or {}).get("repo_name") or (image_info or {}).get("repo") or repo)
    fail_to_pass = _test_id_list((image_info or {}).get("FAIL_TO_PASS")) or _fail_to_pass_from_reproduction_tests(student_input)
    pass_to_pass = _test_id_list((image_info or {}).get("PASS_TO_PASS"))
    fail_to_fail = _test_id_list((image_info or {}).get("FAIL_TO_FAIL"))
    pass_to_fail = _test_id_list((image_info or {}).get("PASS_TO_FAIL"))
    image_name = str((image_info or {}).get("image_name") or docker_image if docker_image != generic_image else "")

    extra_info: dict[str, Any] = {
        "repo": str((image_info or {}).get("repo") or repo),
        "repo_name": repo_name,
        "instance_id": task_id,
        "base_commit": (image_info or {}).get("base_commit") or base_commit,
        "patch": teacher_pi.get("golden_patch", "") or (image_info or {}).get("patch", ""),
        "test_patch": teacher_pi.get("fail_to_pass_tests", "") or teacher_pi.get("test_patch", "") or (image_info or {}).get("test_patch", ""),
        # For staged OPD samples the runtime agent must receive the staged
        # [SYSTEM]/[USER] prompt. If we pass the raw SWE issue text here,
        # SWEAgent will wrap it in the generic repair template and inject
        # patching instructions into localization/reproduction stages.
        "problem_statement": staged_problem_statement,
        "hints_text": student_input.get("hints_text") or (image_info or {}).get("hints_text", ""),
        "created_at": "",
        "version": (image_info or {}).get("version") or f"staged-opd-{sample['stage']}",
        "install_config": _offline_swerebench_install_config((image_info or {}).get("install_config")),
        "requirements": (image_info or {}).get("requirements", ""),
        "environment": (image_info or {}).get("environment", ""),
        "FAIL_TO_PASS": fail_to_pass,
        "FAIL_TO_FAIL": fail_to_fail,
        "PASS_TO_PASS": pass_to_pass,
        "PASS_TO_FAIL": pass_to_fail,
        "environment_setup_commit": (image_info or {}).get("environment_setup_commit") or base_commit,
        "parsed_commit": (image_info or {}).get("parsed_commit") or "{}",
        "parsed_commit_content": (image_info or {}).get("parsed_commit_content") or (image_info or {}).get("parsed_commit") or "{}",
        "execution_result_content": (image_info or {}).get("execution_result_content") or "",
        "expected_output_json": (image_info or {}).get("expected_output_json") or "",
        "run_tests": (image_info or {}).get("run_tests") or "",
        "docker_image": docker_image,
        "source_dataset": source_dataset,
        # Stage1/2 are trained from OPSD labels only. Stage3 is real repair,
        # so keep the hidden FAIL_TO_PASS/PASS_TO_PASS metadata for the
        # environment verifier while hiding it from model-visible prompts.
        "disable_execution_reward": sample["stage"] != "patch_generation",
        "image_name": image_name,
        "swesmith_repo": str(swesmith_info["repo"]) if swesmith_info else "",
        "swerebench_repo": str(swerebench_info["repo"]) if swerebench_info else "",
        "r2e_gym_repo": str(r2e_gym_info["repo"]) if r2e_gym_info else "",
        "swe_gym_repo": str(swe_gym_info["repo"]) if swe_gym_info else "",
        "task_id": task_id,
        "stage": sample["stage"],
        "split": sample["split"],
        "student_input": sample["student_input"],
        "teacher_pi": sample["teacher_pi"],
        "student_messages": sample["student_messages"],
        "teacher_messages": sample["teacher_messages"],
        "privileged_context": {
            "student_input": sample["student_input"],
            "teacher_pi": sample["teacher_pi"],
            "teacher_messages": sample["teacher_messages"],
        },
        "codebase_access": access,
        "raw_record": sample["raw_record"],
    }
    if sample.get("stage") == "localized_issue_reproduction":
        issue_reproduction_patch = (
            teacher_pi.get("issue_reproduction_gold_patch")
            or teacher_pi.get("golden_patch")
            or (image_info or {}).get("patch", "")
        )
        # SWE-smith runtime images are already buggy. Their staged patch is the
        # bug-introducing diff, so the verifier must apply it in reverse to
        # simulate the external repair. SWE-rebench/SWE-bench patches are
        # normal repair patches and should be applied forward.
        issue_reproduction_patch_reverse = bool(swesmith_info and str(issue_reproduction_patch).strip())
        extra_info.update(
            {
                "issue_reproduction_eval_enable": True,
                "issue_reproduction_reward_enable": False,
                "issue_reproduction_path": "/testbed/reproduce_issue.py",
                "issue_reproduction_timeout": 120,
                "issue_reproduction_gold_patch": issue_reproduction_patch,
                "issue_reproduction_patch_apply_reverse": issue_reproduction_patch_reverse,
            }
        )
    if generic_container_name:
        extra_info["generic_sandbox_container_name"] = generic_container_name.format(
            repo=repo,
            repo_leaf=_repo_leaf(repo),
            task_id=task_id,
            stage=sample["stage"],
            split=sample["split"],
        )
    return extra_info


def build_verl_records(
    samples: Iterable[dict[str, Any]],
    *,
    generic_image: str = DEFAULT_GENERIC_IMAGE,
    generic_container_name: str | None = None,
    swesmith_metadata: dict[str, dict[str, Any]] | None = None,
    swerebench_metadata: dict[str, dict[str, Any]] | None = None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None = None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None = None,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    records = []
    iterator = samples
    if tqdm is not None:
        iterator = tqdm(samples, desc=desc or "Building verl records")
    for sample in iterator:
        extra_info = _extra_info_from_sample(
            sample,
            generic_image=generic_image,
            generic_container_name=generic_container_name,
            swesmith_metadata=swesmith_metadata,
            swerebench_metadata=swerebench_metadata,
            r2e_gym_metadata=r2e_gym_metadata,
            swe_gym_metadata=swe_gym_metadata,
        )
        records.append(
            {
                "prompt": sample["student_messages"],
                "reward_model": {"style": "rule", "ground_truth": None},
                "extra_info": json.dumps(extra_info, ensure_ascii=False),
            }
        )
    return records


def _filter_samples(samples: list[dict[str, Any]], *, repo_filter: str | None) -> list[dict[str, Any]]:
    if not repo_filter:
        return samples
    repos = {repo.strip() for repo in repo_filter.split(",") if repo.strip()}
    if not repos:
        return samples
    return [sample for sample in samples if (_codebase_access(sample).get("repo") in repos)]


def _filter_samples_by_task_id(
    samples: list[dict[str, Any]],
    *,
    task_id_filter: set[str] | None,
) -> list[dict[str, Any]]:
    if not task_id_filter:
        return samples
    return [sample for sample in samples if str(sample.get("task_id") or "") in task_id_filter]


def _load_task_id_file(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    task_ids = {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}
    return task_ids or None


def _filter_samples_by_source_dataset(
    samples: list[dict[str, Any]],
    *,
    source_dataset_filter: str | None,
    swesmith_metadata: dict[str, dict[str, Any]] | None,
    swerebench_metadata: dict[str, dict[str, Any]] | None = None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None = None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not source_dataset_filter:
        return samples

    allowed = {item.strip() for item in source_dataset_filter.split(",") if item.strip()}
    invalid = allowed - {
        "swe_smith",
        "swesmith",
        "swe_rebench",
        "swerebench",
        "r2e_gym",
        "r2egym",
        "swe_gym",
        "swegym",
        "generic_sandbox",
    }
    if invalid:
        raise ValueError(
            f"Unsupported source_dataset_filter value(s) {sorted(invalid)}; "
            "allowed: swe_smith, swe_rebench, r2e_gym, swe_gym, generic_sandbox"
        )
    if not allowed:
        return samples

    normalized_allowed = {
        {
            "swesmith": "swe_smith",
            "swerebench": "swe_rebench",
            "r2egym": "r2e_gym",
            "swegym": "swe_gym",
        }.get(item, item)
        for item in allowed
    }
    swesmith = swesmith_metadata or {}
    swerebench = swerebench_metadata or {}
    r2e_gym = r2e_gym_metadata or {}
    swe_gym = swe_gym_metadata or {}

    def _source(sample: dict[str, Any]) -> str:
        task_id = sample.get("task_id") or ""
        if task_id in swesmith:
            return "swe_smith"
        if task_id in swerebench:
            return "swe_rebench"
        if task_id in r2e_gym:
            return "r2e_gym"
        if task_id in swe_gym:
            return "swe_gym"
        access = _codebase_access(sample)
        access_image = str(access.get("docker_image") or access.get("image_name") or "")
        if access_image:
            return _infer_source_dataset_from_image(access_image, access)
        return "generic_sandbox"

    return [sample for sample in samples if _source(sample) in normalized_allowed]


def _inject_swesmith_reproduction_teacher_evidence(
    sample: dict[str, Any],
    *,
    swesmith_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add neutral reproduction evidence from SWE-smith metadata when test diffs are absent."""

    if sample.get("stage") != "localized_issue_reproduction" or not swesmith_info:
        return sample

    teacher_pi = sample.get("teacher_pi") or {}
    if teacher_pi.get("fail_to_pass_tests"):
        return sample

    fail_to_pass = _test_id_list(swesmith_info.get("FAIL_TO_PASS"))
    if not fail_to_pass:
        return sample

    enriched = copy.deepcopy(sample)
    enriched_teacher_pi = copy.deepcopy(teacher_pi)
    enriched_teacher_pi["fail_to_pass_test_names"] = fail_to_pass
    enriched["teacher_pi"] = enriched_teacher_pi
    enriched["privileged_context"]["teacher_pi"] = enriched_teacher_pi
    enriched["raw_record"]["teacher_pi"] = enriched_teacher_pi
    enriched["teacher_messages"] = build_teacher_messages(enriched["raw_record"])
    return enriched


def _inject_reproduction_teacher_evidence(
    sample: dict[str, Any],
    *,
    image_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Add neutral reproduction evidence from runtime metadata when test diffs are absent."""

    if sample.get("stage") != "localized_issue_reproduction" or not image_info:
        return sample

    teacher_pi = sample.get("teacher_pi") or {}
    if teacher_pi.get("fail_to_pass_tests"):
        return sample

    enriched_teacher_pi = copy.deepcopy(teacher_pi)
    test_patch = str(image_info.get("test_patch") or "")
    if test_patch:
        enriched_teacher_pi["fail_to_pass_tests"] = test_patch
    test_code = image_info.get("fail_to_pass_test_code") or image_info.get("test_code")
    if test_code and not enriched_teacher_pi.get("fail_to_pass_test_code") and not enriched_teacher_pi.get("test_code"):
        enriched_teacher_pi["fail_to_pass_test_code"] = _plain_list(test_code)
    fail_to_pass = _test_id_list(image_info.get("FAIL_TO_PASS"))
    if fail_to_pass:
        enriched_teacher_pi["fail_to_pass_test_names"] = fail_to_pass
    if enriched_teacher_pi == teacher_pi:
        return sample

    enriched = copy.deepcopy(sample)
    enriched["teacher_pi"] = enriched_teacher_pi
    enriched["privileged_context"]["teacher_pi"] = enriched_teacher_pi
    enriched["raw_record"]["teacher_pi"] = enriched_teacher_pi
    enriched["teacher_messages"] = build_teacher_messages(enriched["raw_record"])
    return enriched


def _has_removed_line_clue(sample: dict[str, Any]) -> bool:
    student_input = sample.get("student_input") if isinstance(sample.get("student_input"), dict) else {}
    localized_buggy_files = student_input.get("localized_buggy_files")
    if not isinstance(localized_buggy_files, list):
        return False
    for file_entry in localized_buggy_files:
        if not isinstance(file_entry, dict) or not file_entry.get("file"):
            continue
        old_buggy_lines = file_entry.get("old_buggy_lines")
        if not isinstance(old_buggy_lines, list):
            continue
        for hunk_entry in old_buggy_lines:
            if isinstance(hunk_entry, dict) and hunk_entry.get("removed_lines"):
                return True
    return False


def _sample_filter_reasons(
    sample: dict[str, Any],
    *,
    drop_invalid_samples: bool,
    max_student_prompt_chars: int | None,
    max_teacher_prompt_chars: int | None,
    require_execution_metadata_for_patch: bool,
    swesmith_metadata: dict[str, dict[str, Any]] | None,
    swerebench_metadata: dict[str, dict[str, Any]] | None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None,
    generic_image: str,
) -> list[str]:
    reasons: list[str] = []
    stage = str(sample.get("stage") or "")
    student_input = sample.get("student_input") if isinstance(sample.get("student_input"), dict) else {}
    teacher_pi = sample.get("teacher_pi") if isinstance(sample.get("teacher_pi"), dict) else {}
    if drop_invalid_samples:
        if not sample.get("task_id"):
            reasons.append("missing_task_id")
        if not str(student_input.get("issue_statement") or "").strip():
            reasons.append("missing_issue_statement")
        access = _codebase_access(sample)
        if not (access.get("repo") or access.get("checkout_repo")):
            reasons.append("missing_repo")
        if not (access.get("base_commit") or access.get("checkout_ref")):
            reasons.append("missing_base_commit")
        if stage == "causal_localization":
            if not _plain_list(teacher_pi.get("gold_changed_non_test_files")):
                reasons.append("missing_gold_changed_non_test_files")
            if not _plain_list(teacher_pi.get("gold_hunk_ranges")):
                reasons.append("missing_gold_hunk_ranges")
            if not _plain_list(teacher_pi.get("old_buggy_lines")):
                reasons.append("missing_old_buggy_lines")
        if stage in {"localized_issue_reproduction", "patch_generation"}:
            if not _plain_list(student_input.get("localized_buggy_files")):
                reasons.append("missing_localized_buggy_files")
            if not _has_removed_line_clue(sample):
                reasons.append("missing_removed_line_clue")
        if stage == "localized_issue_reproduction":
            if not (teacher_pi.get("issue_reproduction_gold_patch") or teacher_pi.get("golden_patch") or teacher_pi.get("patch")):
                reasons.append("missing_reproduction_gold_patch")
            if not (
                teacher_pi.get("fail_to_pass_test_code")
                or teacher_pi.get("test_code")
                or teacher_pi.get("fail_to_pass_tests")
                or teacher_pi.get("test_patch")
                or teacher_pi.get("fail_to_pass_test_ids")
                or teacher_pi.get("fail_to_pass_test_names")
            ):
                reasons.append("missing_reproduction_test_evidence")
        if stage == "patch_generation" and not str(teacher_pi.get("golden_patch") or "").strip():
            reasons.append("missing_golden_patch")
    if require_execution_metadata_for_patch and stage == "patch_generation":
        task_id = str(sample.get("task_id") or "")
        _, runtime_info = _runtime_metadata_for_task(
            task_id,
            swesmith_metadata=swesmith_metadata,
            swerebench_metadata=swerebench_metadata,
            r2e_gym_metadata=r2e_gym_metadata,
            swe_gym_metadata=swe_gym_metadata,
        )
        access = _codebase_access(sample)
        docker_image = str((runtime_info or {}).get("docker_image") or (runtime_info or {}).get("image_name") or access.get("docker_image") or "")
        fail_to_pass = _test_id_list((runtime_info or {}).get("FAIL_TO_PASS")) or _fail_to_pass_from_reproduction_tests(student_input)
        if not docker_image or docker_image == generic_image:
            reasons.append("missing_execution_docker_image")
        if not fail_to_pass:
            reasons.append("missing_fail_to_pass")
    if max_student_prompt_chars is not None:
        student_len = len(_messages_to_text(sample.get("student_messages") or []))
        if student_len > max_student_prompt_chars:
            reasons.append("student_prompt_too_long")
    if max_teacher_prompt_chars is not None:
        teacher_len = len(_messages_to_text(sample.get("teacher_messages") or []))
        if teacher_len > max_teacher_prompt_chars:
            reasons.append("teacher_prompt_too_long")
    return reasons


def _filter_invalid_or_long_samples(
    samples: list[dict[str, Any]],
    *,
    drop_invalid_samples: bool,
    max_student_prompt_chars: int | None,
    max_teacher_prompt_chars: int | None,
    require_execution_metadata_for_patch: bool,
    swesmith_metadata: dict[str, dict[str, Any]] | None,
    swerebench_metadata: dict[str, dict[str, Any]] | None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None,
    generic_image: str,
) -> list[dict[str, Any]]:
    if (
        not drop_invalid_samples
        and max_student_prompt_chars is None
        and max_teacher_prompt_chars is None
        and not require_execution_metadata_for_patch
    ):
        return samples
    kept: list[dict[str, Any]] = []
    filtered = collections.Counter()
    filtered_sample_count = 0
    for sample in samples:
        reasons = _sample_filter_reasons(
            sample,
            drop_invalid_samples=drop_invalid_samples,
            max_student_prompt_chars=max_student_prompt_chars,
            max_teacher_prompt_chars=max_teacher_prompt_chars,
            require_execution_metadata_for_patch=require_execution_metadata_for_patch,
            swesmith_metadata=swesmith_metadata,
            swerebench_metadata=swerebench_metadata,
            r2e_gym_metadata=r2e_gym_metadata,
            swe_gym_metadata=swe_gym_metadata,
            generic_image=generic_image,
        )
        if reasons:
            filtered_sample_count += 1
            filtered.update(reasons)
            continue
        kept.append(sample)
    if filtered:
        print(
            json.dumps(
                {
                    "filtered_samples": filtered_sample_count,
                    "filter_reasons": dict(sorted(filtered.items())),
                    "kept_samples": len(kept),
                },
                ensure_ascii=False,
            )
        )
    return kept


def _inject_swesmith_patch_generation_repair_teacher_evidence(
    sample: dict[str, Any],
    *,
    swesmith_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """SWE-smith stage3 images are already buggy, so reverse bug patches for teacher repair evidence."""

    if sample.get("stage") != "patch_generation" or not swesmith_info:
        return sample

    teacher_pi = sample.get("teacher_pi") or {}
    golden_patch = str(teacher_pi.get("golden_patch") or "")
    if not golden_patch.strip():
        return sample

    enriched = copy.deepcopy(sample)
    enriched_teacher_pi = copy.deepcopy(teacher_pi)
    enriched_teacher_pi["golden_patch"] = _reverse_unified_diff(golden_patch)
    enriched["teacher_pi"] = enriched_teacher_pi
    enriched["privileged_context"]["teacher_pi"] = enriched_teacher_pi
    enriched["raw_record"]["teacher_pi"] = enriched_teacher_pi
    enriched["teacher_messages"] = build_teacher_messages(enriched["raw_record"])
    return enriched


def write_verl_parquet(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return output_path


def convert_split(
    *,
    stages_root: str | Path,
    stage: str,
    split: str,
    output_root: str | Path,
    generic_image: str = DEFAULT_GENERIC_IMAGE,
    generic_container_name: str | None = None,
    swesmith_data_root: str | Path | None = None,
    swesmith_metadata: dict[str, dict[str, Any]] | None = None,
    swerebench_data_root: str | Path | None = None,
    swerebench_metadata: dict[str, dict[str, Any]] | None = None,
    r2e_gym_data_root: str | Path | None = None,
    r2e_gym_metadata: dict[str, dict[str, Any]] | None = None,
    swe_gym_data_root: str | Path | None = None,
    swe_gym_metadata: dict[str, dict[str, Any]] | None = None,
    swe_gym_docker_host: str | None = None,
    swe_gym_docker_image_list: str | Path | None = None,
    repo_filter: str | None = None,
    task_id_filter: set[str] | None = None,
    source_dataset_filter: str | None = None,
    limit: int | None = None,
    drop_invalid_samples: bool = False,
    max_student_prompt_chars: int | None = None,
    max_teacher_prompt_chars: int | None = None,
    require_execution_metadata_for_patch: bool = False,
) -> Path:
    samples = load_staged_opd_dataset(stages_root, stage=stage, split=split, limit=None)
    if swesmith_metadata is None and swesmith_data_root is not None:
        swesmith_metadata = load_swesmith_image_metadata(swesmith_data_root)
    if swerebench_metadata is None and swerebench_data_root is not None:
        swerebench_metadata = load_swerebench_image_metadata(swerebench_data_root)
    if r2e_gym_metadata is None and r2e_gym_data_root is not None:
        r2e_gym_metadata = load_r2e_gym_metadata(r2e_gym_data_root)
    if swe_gym_metadata is None and swe_gym_data_root is not None:
        swe_gym_metadata = load_swe_gym_metadata(swe_gym_data_root)
    if swe_gym_metadata and (swe_gym_docker_host or swe_gym_docker_image_list):
        swe_gym_metadata = enrich_swe_gym_metadata_with_docker_images(
            swe_gym_metadata,
            docker_host=swe_gym_docker_host,
            image_list_path=swe_gym_docker_image_list,
        )
    samples = _filter_samples_by_task_id(samples, task_id_filter=task_id_filter)
    samples = _filter_samples(samples, repo_filter=repo_filter)
    samples = _filter_samples_by_source_dataset(
        samples,
        source_dataset_filter=source_dataset_filter,
        swesmith_metadata=swesmith_metadata,
        swerebench_metadata=swerebench_metadata,
        r2e_gym_metadata=r2e_gym_metadata,
        swe_gym_metadata=swe_gym_metadata,
    )
    if swesmith_metadata:
        samples = [
            _inject_swesmith_reproduction_teacher_evidence(
                sample,
                swesmith_info=swesmith_metadata.get(sample.get("task_id") or ""),
            )
            for sample in samples
        ]
        samples = [
            _inject_swesmith_patch_generation_repair_teacher_evidence(
                sample,
                swesmith_info=swesmith_metadata.get(sample.get("task_id") or ""),
            )
            for sample in samples
        ]
    if swerebench_metadata:
        samples = [
            _inject_reproduction_teacher_evidence(
                sample,
                image_info=swerebench_metadata.get(sample.get("task_id") or ""),
            )
            for sample in samples
        ]
    if r2e_gym_metadata:
        samples = [
            _inject_reproduction_teacher_evidence(
                sample,
                image_info=r2e_gym_metadata.get(sample.get("task_id") or ""),
            )
            for sample in samples
        ]
    if swe_gym_metadata:
        samples = [
            _inject_reproduction_teacher_evidence(
                sample,
                image_info=swe_gym_metadata.get(sample.get("task_id") or ""),
            )
            for sample in samples
        ]
    samples = _filter_invalid_or_long_samples(
        samples,
        drop_invalid_samples=drop_invalid_samples,
        max_student_prompt_chars=max_student_prompt_chars,
        max_teacher_prompt_chars=max_teacher_prompt_chars,
        require_execution_metadata_for_patch=require_execution_metadata_for_patch,
        swesmith_metadata=swesmith_metadata,
        swerebench_metadata=swerebench_metadata,
        r2e_gym_metadata=r2e_gym_metadata,
        swe_gym_metadata=swe_gym_metadata,
        generic_image=generic_image,
    )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No staged OPD samples after filtering: stage={stage} split={split}")
    records = build_verl_records(
        samples,
        generic_image=generic_image,
        generic_container_name=generic_container_name,
        swesmith_metadata=swesmith_metadata,
        swerebench_metadata=swerebench_metadata,
        r2e_gym_metadata=r2e_gym_metadata,
        swe_gym_metadata=swe_gym_metadata,
        desc=f"{stage}/{split}",
    )
    stage_dir = "all" if stage == "all" else stage
    output_path = Path(output_root) / stage_dir / f"{split}_verl.parquet"
    return write_verl_parquet(records, output_path)


def _parse_csv_choices(value: str, *, allowed: tuple[str, ...]) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported value(s) {invalid}; allowed: {', '.join(allowed)}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert staged OPD JSONL data to verl parquet files.")
    parser.add_argument("--stages-root", default="stages", help="Root directory containing <stage>/<split>.jsonl files.")
    parser.add_argument("--stage", default="all", help="Stage name or comma-separated stages; supports all.")
    parser.add_argument("--splits", default="train,valid", help="Comma-separated splits to convert.")
    parser.add_argument("--output-root", required=True, help="Output root for <stage>/<split>_verl.parquet files.")
    parser.add_argument("--generic-image", default=DEFAULT_GENERIC_IMAGE, help="Docker image recorded in extra_info.")
    parser.add_argument(
        "--swesmith-data-root",
        default=None,
        help="Optional SWE-smith root or data directory; matching instance_id rows use SWE-smith image_name as docker_image.",
    )
    parser.add_argument(
        "--swerebench-data-root",
        default=None,
        help="Optional SWE-rebench root or data directory; matching instance_id rows use SWE-rebench docker_image.",
    )
    parser.add_argument(
        "--r2e-gym-data-root",
        default=None,
        help="Optional R2E-Gym/R2E-Gym-Subset root, data directory, parquet, or canonical/tasks.jsonl metadata.",
    )
    parser.add_argument(
        "--swe-gym-data-root",
        default=None,
        help="Optional SumanthRH/SWE-Gym-Subset root, data directory, parquet, or canonical/tasks.jsonl metadata.",
    )
    parser.add_argument(
        "--swe-gym-docker-host",
        default=None,
        help="Optional Docker API host for inferring missing SWE-Gym images, e.g. tcp://127.0.0.1:23750.",
    )
    parser.add_argument(
        "--swe-gym-docker-image-list",
        default=None,
        help="Optional text file of image tags or saved Docker /images/json response for SWE-Gym image inference.",
    )
    parser.add_argument(
        "--generic-container-name",
        default=None,
        help="Optional container name template. Fields: {repo}, {repo_leaf}, {task_id}, {stage}, {split}.",
    )
    parser.add_argument("--repo-filter", default=None, help="Optional comma-separated repo filter, e.g. sympy/sympy.")
    parser.add_argument("--task-id-file", default=None, help="Optional newline-delimited task_id filter file.")
    parser.add_argument(
        "--source-dataset-filter",
        default=None,
        help="Optional comma-separated source filter after metadata matching; allowed: swe_smith,swe_rebench,r2e_gym,swe_gym,generic_sandbox.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-stage/per-split row limit after filtering.")
    parser.add_argument("--drop-invalid-samples", action="store_true", help="Drop rows missing stage-required prompt/evidence fields.")
    parser.add_argument("--max-student-prompt-chars", type=int, default=None, help="Drop rows whose rendered student prompt exceeds this many characters.")
    parser.add_argument("--max-teacher-prompt-chars", type=int, default=None, help="Drop rows whose rendered teacher prompt exceeds this many characters.")
    parser.add_argument(
        "--require-execution-metadata-for-patch",
        action="store_true",
        help="For patch_generation, drop rows without executable docker image or FAIL_TO_PASS metadata.",
    )
    args = parser.parse_args(argv)

    _load_staged_dataset_api()
    stage_choices = (*SUPPORTED_STAGES, "all")
    stages = _parse_csv_choices(args.stage, allowed=stage_choices)
    splits = _parse_csv_choices(args.splits, allowed=SUPPORTED_SPLITS)
    swesmith_metadata = (
        load_swesmith_image_metadata(args.swesmith_data_root) if args.swesmith_data_root else None
    )
    swerebench_metadata = (
        load_swerebench_image_metadata(args.swerebench_data_root) if args.swerebench_data_root else None
    )
    r2e_gym_metadata = load_r2e_gym_metadata(args.r2e_gym_data_root) if args.r2e_gym_data_root else None
    swe_gym_metadata = load_swe_gym_metadata(args.swe_gym_data_root) if args.swe_gym_data_root else None
    swe_gym_docker_host = (
        args.swe_gym_docker_host
        or os.environ.get("R2EGYM_DOCKER_HOST", "").strip()
        or os.environ.get("DOCKER_HOST", "").strip()
        or None
    )
    if swe_gym_metadata and (swe_gym_docker_host or args.swe_gym_docker_image_list):
        swe_gym_metadata = enrich_swe_gym_metadata_with_docker_images(
            swe_gym_metadata,
            docker_host=swe_gym_docker_host,
            image_list_path=args.swe_gym_docker_image_list,
        )
    task_id_filter = _load_task_id_file(args.task_id_file)

    written = []
    for stage in stages:
        for split in splits:
            path = convert_split(
                stages_root=args.stages_root,
                stage=stage,
                split=split,
                output_root=args.output_root,
                generic_image=args.generic_image,
                generic_container_name=args.generic_container_name,
                swesmith_metadata=swesmith_metadata,
                swerebench_metadata=swerebench_metadata,
                r2e_gym_metadata=r2e_gym_metadata,
                swe_gym_metadata=swe_gym_metadata,
                swe_gym_docker_host=None,
                swe_gym_docker_image_list=None,
                repo_filter=args.repo_filter,
                task_id_filter=task_id_filter,
                source_dataset_filter=args.source_dataset_filter,
                limit=args.limit,
                drop_invalid_samples=args.drop_invalid_samples,
                max_student_prompt_chars=args.max_student_prompt_chars,
                max_teacher_prompt_chars=args.max_teacher_prompt_chars,
                require_execution_metadata_for_patch=args.require_execution_metadata_for_patch,
            )
            written.append(path)
            print(f"wrote {path}")
    print(json.dumps({"num_files": len(written), "files": [str(path) for path in written]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
