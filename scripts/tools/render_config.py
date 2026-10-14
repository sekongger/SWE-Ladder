#!/usr/bin/env python3
"""Fill the ``{{PLACEHOLDER}}`` slots in a training config with site-local paths.

The released configs under ``training/configs/`` keep every machine-specific path
as a placeholder, so they are safe to read and diff but cannot be launched as-is.
This script turns one into a runnable config::

    scripts/tools/render_config.py \\
        training/configs/opsd_ladder_4b_openhands_leg1_loc.yaml \\
        --site training/configs/site.env \\
        -o /tmp/run.yaml

Why a renderer and not plain ``${VAR}`` expansion: ``run_npu_train_from_yaml.sh``
exports every ``env:`` entry verbatim (``export KEY=<shell-quoted value>``), so a
``${VAR}`` written inside a value would reach the trainer as six literal
characters rather than an expansion.

The site file is ``KEY=value`` lines (``#`` comments, blanks ignored). Values may
reference earlier keys with ``${KEY}``; those are expanded before substitution, so
one ``WORKSPACE_ROOT`` can drive the rest. Keys given on the command line with
``--set KEY=value`` win over the file, and ``--print-placeholders`` lists what a
config still needs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
REFERENCE_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")
MAX_REFERENCE_DEPTH = 10


def parse_site_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"{path}:{lineno}: expected KEY=value, got {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.isidentifier():
            raise SystemExit(f"{path}:{lineno}: {key!r} is not a valid key")
        values[key] = value.strip().strip('"').strip("'")
    return values


def expand_references(values: dict[str, str]) -> dict[str, str]:
    """Resolve ``${KEY}`` references between site values (e.g. ``${WORKSPACE_ROOT}``)."""
    resolved = dict(values)
    for _ in range(MAX_REFERENCE_DEPTH):
        changed = False
        for key, value in resolved.items():
            new_value = REFERENCE_RE.sub(lambda m: resolved.get(m.group(1), m.group(0)), value)
            if new_value != value:
                resolved[key] = new_value
                changed = True
        if not changed:
            return resolved
    raise SystemExit("site values contain a circular ${...} reference")


def placeholders_in(text: str) -> list[str]:
    seen: list[str] = []
    for name in PLACEHOLDER_RE.findall(text):
        if name not in seen:
            seen.append(name)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="config template under training/configs/")
    parser.add_argument("-o", "--output", type=Path, help="write here (default: stdout)")
    parser.add_argument(
        "-s",
        "--site",
        type=Path,
        action="append",
        default=[],
        help="KEY=value file; repeatable, later files win (default: training/configs/site.env if present)",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override one placeholder; repeatable")
    parser.add_argument("--print-placeholders", action="store_true",
                        help="list the placeholders this config needs and exit")
    parser.add_argument("--allow-missing", action="store_true",
                        help="leave unknown placeholders in place instead of failing")
    args = parser.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    text = args.config.read_text(encoding="utf-8")

    if args.print_placeholders:
        for name in placeholders_in(text):
            print(name)
        return 0

    site_files = list(args.site)
    if not site_files:
        default_site = args.config.parent / "site.env"
        if default_site.is_file():
            site_files = [default_site]

    values: dict[str, str] = {}
    for site in site_files:
        if not site.is_file():
            raise SystemExit(f"site file not found: {site}")
        values.update(parse_site_file(site))
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        values[key.strip()] = value
    values = expand_references(values)

    missing = [name for name in placeholders_in(text) if name not in values]
    if missing and not args.allow_missing:
        hint = "" if site_files else " (no site file was loaded; pass --site)"
        raise SystemExit(
            f"missing values for: {', '.join(missing)}{hint}\n"
            f"Copy {args.config.parent / 'site.env.example'} to site.env and fill it in."
        )

    rendered = PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        still = placeholders_in(rendered)
        suffix = f" (unresolved: {', '.join(still)})" if still else ""
        print(f"wrote {args.output}{suffix}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
