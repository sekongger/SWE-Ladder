"""Wrapper to run verl's legacy_model_merger under this CPU-only Ascend env.

Two imports must be neutralized to merge FSDP shards -> HF on CPU here:
  * deepspeed  -> transformers.modeling_utils does `if is_deepspeed_available(): import deepspeed`,
                  and this env's deepspeed eagerly imports cpu_adam and dies with
                  "name 'torch' is not defined".
  * torch_npu  -> transformers' models/qwen3/modeling_qwen3.py -> modeling_flash_attention_utils
                  does `import torch_npu` when is_torch_npu_available(); torch_npu._C needs
                  libhccl.so, which is ABSENT from the `latest`(cann-8.5.0) toolkit (only 8.1.RC1
                  has the plain libhccl.so) -> ImportError. The merge is CPU-only; we don't want it.

Mechanism: patch `importlib.util.find_spec` to return None for these two names. That is the single
choke point both callers go through:
  * verl/scripts/legacy_model_merger.py:57 does `importlib.util.find_spec("torch_npu")` at module
    load; None -> it skips torch_npu.
  * transformers/accelerate `_is_package_available` do `importlib.util.find_spec(pkg) is not None`;
    None -> is_deepspeed_available()/is_torch_npu_available() report False -> the gated bare
    `import deepspeed`/`import torch_npu` lines never execute.
Because availability is uniformly False, no code path performs a bare import of either module, so we
do NOT need (and must NOT use) a meta_path finder that raises from find_spec — that is illegal for a
MetaPathFinder and propagates as a fatal ModuleNotFoundError (the previous failure at merger:57).

Run with: PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python merge_student.py <ckpt_actor_dir> <out_dir>
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys

# Modules whose *availability* must read as False for a pure-CPU merge (see module docstring).
_BLOCKED_PREFIXES = ("deepspeed", "torch_npu")


def _neutralize_deepspeed():
    # Patch the one choke point (importlib.util.find_spec) BEFORE transformers/accelerate import,
    # since accelerate evaluates device/package availability at import time.
    _orig_find_spec = importlib.util.find_spec

    def _patched_find_spec(name, package=None):
        top = (name or "").lstrip(".").split(".", 1)[0]
        if top in _BLOCKED_PREFIXES:
            return None
        return _orig_find_spec(name, package)

    importlib.util.find_spec = _patched_find_spec

    # Belt-and-suspenders: if these modules are already imported, force their availability helpers
    # to False too. Guarded so a probe that itself trips device detection can't abort us.
    def _false(*a, **k):
        return False

    for mod, attr in (("transformers.utils.import_utils", "is_deepspeed_available"),
                      ("transformers.utils.import_utils", "is_torch_npu_available")):
        try:
            m = sys.modules.get(mod) or __import__(mod, fromlist=[attr])
            setattr(m, attr, _false)
        except Exception as e:  # pragma: no cover
            print(f"[merge_student] note: could not patch {mod}.{attr} ({e})", file=sys.stderr)


def main():
    if len(sys.argv) < 3:
        print("usage: merge_student.py <ckpt_actor_dir> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt, out = sys.argv[1], sys.argv[2]

    _neutralize_deepspeed()

    merger = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "DeepSWE_RL", "rllm", "verl", "scripts", "legacy_model_merger.py")
    sys.argv = [merger, "merge", "--backend", "fsdp", "--local_dir", ckpt, "--target_dir", out]
    runpy.run_path(merger, run_name="__main__")


if __name__ == "__main__":
    main()
