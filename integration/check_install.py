#!/usr/bin/env python3
"""Static dependency check for an installed SWE-Ladder tree.

The release is an overlay: our files plus patches against pinned upstream trees.
The failure mode that static checks catch cheaply is a *dangling first-party
reference* — our code importing a module, or naming a file, that the release
forgot to ship. (That is exactly how the Leaf tool modules were caught.)

This walks every Python file we ship or patch, resolves every
`rllm.* / verl.* / r2egym.* / swe_ladder.*` import by path against the installed
tree, and reports the ones that do not exist. It never imports anything, so it
needs no training stack.

Usage:
  python3 integration/check_install.py --workspace /path/to/workspace
  python3 integration/check_install.py --rllm /path/rllm --r2egym /path/R2E-Gym

Exit status: 0 if every reference resolves (or is a documented optional), 1 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Lazily imported, behind a default-off config flag, and belonging to a separate
# internal eval suite that is deliberately not part of this release. Enabling
# `agent.enable_tip_eval_snapshot` in a config is what would need it.
KNOWN_OPTIONAL = {"rllm.tip_eval"}

FIRST_PARTY = ("rllm", "verl", "r2egym", "swe_ladder")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", help="tree holding DeepSWE_RL/rllm and R2E-Gym")
    ap.add_argument("--rllm", help="rllm checkout (default <workspace>/DeepSWE_RL/rllm)")
    ap.add_argument("--r2egym", help="R2E-Gym checkout (default <workspace>/R2E-Gym)")
    ap.add_argument("--verl", help="verl checkout (default <rllm>/verl)")
    args = ap.parse_args()
    ws = Path(args.workspace or os.environ.get("WORKSPACE_ROOT") or ".").resolve()
    args.rllm = Path(args.rllm or ws / "DeepSWE_RL/rllm").resolve()
    args.r2egym = Path(args.r2egym or ws / "R2E-Gym").resolve()
    args.verl = Path(args.verl or args.rllm / "verl").resolve()
    return args


def search_roots(args: argparse.Namespace) -> dict[str, list[Path]]:
    """Where each top-level package can legitimately live."""
    return {
        "rllm": [args.rllm],
        "verl": [args.verl],
        "r2egym": [args.r2egym / "src"],
        # swe_ladder is importable from the repo, and as rllm.opd once installed
        "swe_ladder": [REPO],
    }


def imported_modules(path: Path) -> set[str]:
    """Every first-party dotted module name imported by one file.

    `from pkg.mod import Name` contributes `pkg.mod` only: whether `Name` is a
    submodule or a symbol cannot be told apart without resolving `pkg.mod`
    itself, and the module-level check is what catches a file we forgot to ship.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as exc:
        print(f"  ! unparseable: {path}: {exc}")
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative import: resolved within its own package
                continue
            if node.module:
                out.add(node.module)
    return {m for m in out if m.split(".")[0] in FIRST_PARTY}


def resolve(module: str, roots: dict[str, list[Path]]) -> bool:
    top = module.split(".")[0]
    rel = module.replace(".", "/")
    for base in roots.get(top, []):
        # `rllm.opd.x` also resolves to the repo's own swe_ladder/x
        for candidate in (base / f"{rel}.py", base / rel / "__init__.py", base / rel):
            if candidate.exists():
                return True
        if top == "rllm" and module.startswith("rllm.opd"):
            tail = module[len("rllm.opd"):].lstrip(".").replace(".", "/")
            for candidate in (REPO / "swe_ladder" / f"{tail}.py", REPO / "swe_ladder" / tail):
                if tail and candidate.exists():
                    return True
    return False


def patched_files(patch: Path) -> list[str]:
    return [line.split(" b/")[-1].strip() for line in patch.read_text(errors="replace").splitlines()
            if line.startswith("diff --git a/")]


def added_lines(patch: Path) -> str:
    """The text the patch adds, for telling our references from upstream's.

    A patched file can contain dangling imports that upstream shipped that way
    (`swe_sim.py` soft-imports two r2egym modules that do not exist in the pin).
    Those are upstream's business; only a reference on a `+` line is ours.
    """
    return "\n".join(line[1:] for line in patch.read_text(errors="replace").splitlines()
                     if line.startswith("+") and not line.startswith("+++"))


def main() -> int:
    args = parse_args()
    roots = search_roots(args)

    for label, path in (("rllm", args.rllm), ("R2E-Gym", args.r2egym), ("verl", args.verl)):
        if not path.is_dir():
            print(f"ERROR: {label} checkout not found: {path}", file=sys.stderr)
            return 2

    # Everything we are responsible for: our own package, plus every upstream
    # file our patches rewrite (their new imports are the risky ones).
    targets: list[tuple[str, Path]] = []
    for py in sorted((REPO / "swe_ladder").rglob("*.py")):
        if "__pycache__" not in py.parts:
            targets.append(("swe_ladder", py))
    ours_text = ""
    for name, base in (("rllm", args.rllm), ("r2egym", args.r2egym), ("verl", args.verl)):
        patch = REPO / f"integration/patches/{name}-swe-ladder.patch"
        if not patch.exists():
            print(f"ERROR: missing patch {patch}", file=sys.stderr)
            return 2
        ours_text += added_lines(patch) + "\n"
        for rel in patched_files(patch):
            if not rel.endswith(".py"):
                continue
            installed = base / rel
            if installed.exists():
                targets.append((f"{name} (patched)", installed))
            else:
                print(f"ERROR: patched file not present in the tree: {installed}", file=sys.stderr)
                return 2

    refs: dict[str, set[str]] = {}
    for label, path in targets:
        for module in imported_modules(path):
            refs.setdefault(module, set()).add(f"{label}: {path.name}")

    bad = []
    optional = []
    upstream = []
    for module in sorted(refs):
        if resolve(module, roots):
            continue
        root = ".".join(module.split(".")[:2])
        if root in KNOWN_OPTIONAL:
            optional.append(module)
        elif module not in ours_text:
            upstream.append(module)   # already dangling in the pin; not ours
        else:
            bad.append(module)

    print(f"checked {len(targets)} files, {len(refs)} distinct first-party module references")
    for module in optional:
        print(f"  optional (documented, not shipped): {module}  <- {', '.join(sorted(refs[module]))}")
    for module in upstream:
        print(f"  pre-existing upstream gap (soft-imported, not introduced here): {module}"
              f"  <- {', '.join(sorted(refs[module]))}")
    if bad:
        print("\nDANGLING references — the release is missing these modules:")
        for module in bad:
            print(f"  {module}  <- {', '.join(sorted(refs[module]))}")
        return 1
    print("all first-party imports introduced by this release resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
