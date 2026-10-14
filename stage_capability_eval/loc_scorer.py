"""Deterministic scoring for predicted source locations.

The scorer accepts the small JSON-compatible candidate language used by stage
evaluations: a path string, ``path``/``file`` dictionaries, and optional
``function``/``functions`` fields.  Invalid candidates are ignored and valid
duplicates are merged in first-seen order.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _path(value: Any) -> str:
    value = _text(value).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return value


def normalize_candidates(candidates: Any) -> list[dict[str, Any]]:
    """Normalize model output to unique ``{"file", "functions"}`` records."""
    if isinstance(candidates, Mapping):
        if "files" in candidates:
            files = candidates.get("files") or []
            function_by_file: dict[str, list[str]] = {}
            for function in candidates.get("functions", []) or []:
                if not isinstance(function, Mapping):
                    continue
                path = _path(function.get("path", function.get("file", function.get("file_path"))))
                name = _text(function.get("qualified_name", function.get("function", function.get("name"))))
                if path and name:
                    function_by_file.setdefault(path, []).append(name)
            normalized_files: list[Any] = []
            for item in files:
                path = _path(item if isinstance(item, str) else item.get("path", item.get("file", "")) if isinstance(item, Mapping) else "")
                if not path:
                    continue
                entry = dict(item) if isinstance(item, Mapping) else {"file": path}
                entry["file"] = path
                if function_by_file.get(path):
                    entry["functions"] = list(function_by_file[path])
                normalized_files.append(entry)
            candidates = normalized_files or [
                {"file": path, "functions": names} for path, names in function_by_file.items()
            ]
        else:
            candidates = candidates.get("candidates", candidates.get("predictions", candidates.get("locations", [candidates])))
    if isinstance(candidates, str) or candidates is None:
        candidates = [] if candidates is None else [candidates]
    if not isinstance(candidates, Iterable):
        return []
    result: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for raw in candidates:
        functions: list[str] = []
        if isinstance(raw, str):
            value = raw.strip()
            if "::" in value:
                file_name, function = value.split("::", 1)
                functions = [function.strip()] if function.strip() else []
            else:
                file_name = value
        elif isinstance(raw, Mapping):
            file_name = _path(raw.get("file", raw.get("path", raw.get("file_path", raw.get("filename")))))
            function_value = raw.get("functions", raw.get("function", raw.get("qualified_name", raw.get("name"))))
            if isinstance(function_value, str):
                functions = [function_value.strip()] if function_value.strip() else []
            elif isinstance(function_value, Iterable):
                functions = sorted({_text(x) for x in function_value if _text(x)})
        else:
            continue
        file_name = _path(file_name)
        if not file_name or file_name in {"/dev/null", "-"}:
            continue
        functions = sorted(set(functions))
        if file_name not in positions:
            positions[file_name] = len(result)
            result.append({"file": file_name, "functions": functions})
        else:
            current = result[positions[file_name]]
            current["functions"] = sorted(set(current["functions"]) | set(functions))
    return result


def _oracle(oracle: Any) -> list[dict[str, Any]]:
    if isinstance(oracle, Mapping):
        files = oracle.get("files", oracle.get("gold_files", oracle.get("locations", [])))
    else:
        files = oracle
    return normalize_candidates(files)


def _at_k(values: Sequence[dict[str, Any]], k: int) -> Sequence[dict[str, Any]]:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    return values[:k]


def score_location(candidates: Any, oracle: Any, *, ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """Return file/function Acc@k and Recall@k for one task.

    Accuracy is a task-level hit indicator; recall is the fraction of gold
    files/functions covered by the top-k predictions.  Empty gold dimensions
    score ``0.0`` so they cannot silently inflate aggregate results.
    """
    predicted = normalize_candidates(candidates)
    gold = _oracle(oracle)
    gold_files = {item["file"] for item in gold}
    gold_functions = {(item["file"], fn) for item in gold for fn in item["functions"]}
    output: dict[str, float] = {}
    for k in ks:
        top = _at_k(predicted, k)
        files = {item["file"] for item in top}
        functions = {(item["file"], fn) for item in top for fn in item["functions"]}
        output[f"file_acc@{k}"] = float(bool(gold_files and gold_files.issubset(files)))
        output[f"file_recall@{k}"] = len(files & gold_files) / len(gold_files) if gold_files else 0.0
        output[f"function_acc@{k}"] = float(bool(gold_functions and gold_functions.issubset(functions)))
        output[f"function_recall@{k}"] = len(functions & gold_functions) / len(gold_functions) if gold_functions else 0.0
    return output


def aggregate_scores(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Average finite metric values from task records, raising on bad data."""
    if not records:
        raise ValueError("cannot aggregate an empty record set")
    keys = sorted({key for record in records for key in record.get("metrics", {})})
    if not keys:
        raise ValueError("no metrics available for aggregation")
    result: dict[str, float] = {}
    for key in keys:
        values = [float(record["metrics"][key]) for record in records if key in record.get("metrics", {})]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite or missing metric: {key}")
        result[key] = sum(values) / len(values)
    return result
