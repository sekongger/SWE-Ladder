"""Shared checkpoint and in-memory stage-capability evaluation core."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .loc_scorer import aggregate_scores, score_location


class EvaluationError(RuntimeError):
    """A boundary failure with a stable machine-readable category."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class InMemoryModelAdapter:
    mode = "in_memory"

    def __init__(self, model: Any):
        self.model = model

    def predict(self, task: Mapping[str, Any]) -> Any:
        target = self.model
        if callable(target):
            return target(task)
        for name in ("predict", "generate", "evaluate"):
            method = getattr(target, name, None)
            if callable(method):
                return method(task)
        raise EvaluationError("restore_failure", "in-memory model has no callable prediction interface")


class CheckpointModelAdapter:
    mode = "checkpoint"

    def __init__(self, checkpoint_path: str | Path, loader: Callable[[Path], Any] | None = None, *, restore_fn: Callable[[Path], Any] | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise EvaluationError("missing_checkpoint", f"checkpoint does not exist: {self.checkpoint_path}")
        if loader is not None and restore_fn is not None:
            raise ValueError("provide either loader or restore_fn, not both")
        self.loader = loader or restore_fn
        self.load_started = time.monotonic()
        try:
            self.model = self.loader(self.checkpoint_path) if self.loader is not None else self.checkpoint_path
        except Exception as exc:
            raise EvaluationError("restore_failure", f"failed to restore checkpoint: {self.checkpoint_path}") from exc
        self.load_seconds = time.monotonic() - self.load_started
        self.sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()

    def predict(self, task: Mapping[str, Any]) -> Any:
        return InMemoryModelAdapter(self.model).predict(task)


class StageCapabilityEvaluator:
    """Run one schema through either model-entry mode."""

    def __init__(self, *, ks: Sequence[int] = (1, 3, 5), stall_timeout_seconds: float | None = None, clock: Callable[[], float] = time.monotonic, protocol_id: str = "stage_capability_v1", dataset_revision: str | None = None):
        self.ks = tuple(ks)
        self.stall_timeout_seconds = stall_timeout_seconds
        self.clock = clock
        self.protocol_id = protocol_id
        self.dataset_revision = dataset_revision

    def evaluate(self, tasks: Sequence[Mapping[str, Any]], adapter: Any, *, model_id: str | None = None, seed: int | None = None, subset: str | None = None) -> dict[str, Any]:
        if not tasks:
            raise EvaluationError("empty_task_set", "evaluation task set is empty")
        logs: list[dict[str, Any]] = [{"phase": "start", "event": "phase_start", "mode": getattr(adapter, "mode", "unknown"), "task_count": len(tasks)}]
        records: list[dict[str, Any]] = []
        started = self.clock()
        for index, task in enumerate(tasks, 1):
            if self.stall_timeout_seconds is not None and self.clock() - started > self.stall_timeout_seconds:
                raise EvaluationError("stall_timeout", f"evaluation stalled after {index - 1} tasks")
            try:
                candidates = adapter.predict(task)
                if candidates is None or (not isinstance(candidates, (str, Mapping, list, tuple, set))):
                    raise ValueError("model output must be a location string, mapping, or sequence")
                if isinstance(candidates, Mapping) and not ({"file", "path", "file_path", "filename", "files", "functions", "candidates", "predictions", "locations"} & set(candidates)):
                    raise ValueError("model output mapping has no recognized location field")
                metrics = score_location(candidates, task.get("oracle", task.get("oracle_loc", task.get("labels", {}))), ks=self.ks)
            except EvaluationError:
                raise
            except Exception as exc:
                # Keep model-invalid outputs in the metric denominator with a
                # deterministic zero score; infrastructure/not-applicable
                # records are handled outside this model-output evaluator.
                zero_metrics = score_location([], task.get("oracle", task.get("oracle_loc", task.get("labels", {}))), ks=self.ks)
                records.append({"schema_version": "stage_capability_eval_v1", "protocol_id": self.protocol_id, "dataset_revision": self.dataset_revision, "model_id": model_id, "seed": seed, "subset": subset, "instance_id": task.get("instance_id"), "status": "model_invalid", "failure_category": "prediction_error", "metrics": zero_metrics})
                logs.append({"phase": "progress", "event": "progress", "completed": index, "total": len(tasks), "mode": getattr(adapter, "mode", "unknown")})
                continue
            records.append({"schema_version": "stage_capability_eval_v1", "protocol_id": self.protocol_id, "dataset_revision": self.dataset_revision, "model_id": model_id, "seed": seed, "subset": subset, "instance_id": task.get("instance_id"), "status": "completed", "failure_category": None, "metrics": metrics})
            logs.append({"phase": "progress", "event": "progress", "completed": index, "total": len(tasks), "mode": getattr(adapter, "mode", "unknown")})
        if not records:
            raise EvaluationError("aggregation_failure", "no task records were produced")
        try:
            aggregate = aggregate_scores(records)
        except Exception as exc:
            raise EvaluationError("aggregation_failure", str(exc)) from exc
        if not all(math.isfinite(value) for value in aggregate.values()):
            raise EvaluationError("non_finite_metric", "aggregate contains a non-finite metric")
        duration = self.clock() - started
        logs.append({"phase": "end", "event": "phase_end", "mode": getattr(adapter, "mode", "unknown"), "completed": len(records), "duration_seconds": duration})
        result = {"schema_version": "stage_capability_eval_v1", "protocol_id": self.protocol_id, "dataset_revision": self.dataset_revision, "model_id": model_id, "seed": seed, "subset": subset, "mode": getattr(adapter, "mode", "unknown"), "records": records, "aggregate": aggregate, "efficiency": {"task_count": len(tasks), "completed": len(records), "duration_seconds": duration, "tasks_per_second": len(records) / duration if duration > 0 else None}, "logs": logs}
        if isinstance(adapter, CheckpointModelAdapter):
            result["checkpoint"] = {"path": str(adapter.checkpoint_path), "sha256": adapter.sha256, "load_seconds": adapter.load_seconds}
        return result


Evaluator = StageCapabilityEvaluator
