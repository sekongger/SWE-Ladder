"""Staged privileged distillation: dataset prep, stage routing, teacher losses.

`integration/install.sh` grafts this package onto the rllm tree as `rllm.opd`,
so every module here is importable under both names. Nothing at import time
touches rllm or torch: the re-exports below are lazy, which is what lets the
offline tools under `analysis/` import a single module (most often
`swe_ladder.stage_classifier`) with no training stack installed.
"""

_LAZY = {
    "SUPPORTED_STAGES": "staged_dataset",
    "build_staged_opd_sample": "staged_dataset",
    "build_student_messages": "staged_dataset",
    "build_teacher_messages": "staged_dataset",
    "load_staged_opd_dataset": "staged_dataset",
    "teacher_topk_reverse_kl": "topk_loss",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(__all__) | set(globals()))
