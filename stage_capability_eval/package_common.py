"""Public task-field allowlist shared by REPRO and PATCH packages."""
from __future__ import annotations

from typing import Any, Mapping


PUBLIC_TASK_FIELDS = {
    "instance_id",
    "repo",
    "base_commit",
    "environment_setup_commit",
    "docker_image",
    "problem_statement",
    "source_manifest",
    "dataset_revision",
    "version",
    "created_at",
}


def public_task_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only issue/environment identity fields into a model task package."""
    return {key: row[key] for key in PUBLIC_TASK_FIELDS if key in row}
