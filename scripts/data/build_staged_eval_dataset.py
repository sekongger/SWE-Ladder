"""Build the per-stage evaluation JSONL used by `stage_capability_eval/`.

Unlike the training parquet, an eval record carries no privileged evidence in its
prompt: it holds the student messages plus a `verifier` block (gold patch, test
patch, fail-to-pass ids) that the scorer — not the model — gets to see.

  python3 scripts/data/build_staged_eval_dataset.py \\
      --input-root stages/causal_localization \\
      --output-root eval_data --stage causal_localization --splits valid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# See the note in build_staged_verl_dataset.py: `swe_ladder` here, `rllm.opd` once installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from swe_ladder.staged_eval import (
        EVAL_STAGES,
        build_eval_records_from_jsonl,
        write_eval_records_jsonl,
    )
except ModuleNotFoundError:  # running from inside an installed workspace
    from rllm.opd.staged_eval import (
        EVAL_STAGES,
        build_eval_records_from_jsonl,
        write_eval_records_jsonl,
    )


SUPPORTED_SPLITS = ("train", "valid")


def _parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in splits if item not in SUPPORTED_SPLITS]
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}. Supported: {', '.join(SUPPORTED_SPLITS)}")
    return splits


def _messages_to_markdown(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role", "")).upper()
        parts.append(f"# {role}\n\n{message.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def _write_preview(records: list[dict], preview_dir: Path, *, stage: str, split: str) -> None:
    if not records:
        return
    preview_dir.mkdir(parents=True, exist_ok=True)
    first = records[0]
    (preview_dir / f"{stage}_{split}_prompt_preview.md").write_text(
        _messages_to_markdown(first["eval_messages"]),
        encoding="utf-8",
    )
    verifier_preview = dict(first["verifier"])
    if "golden_patch" in verifier_preview:
        verifier_preview["golden_patch"] = verifier_preview["golden_patch"][:2000]
    if "test_patch" in verifier_preview:
        verifier_preview["test_patch"] = verifier_preview["test_patch"][:2000]
    (preview_dir / f"{stage}_{split}_verifier_preview.json").write_text(
        json.dumps(verifier_preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build independent staged OPD evaluation JSONL records.")
    parser.add_argument("--input-root", required=True, help="Root containing <split>.jsonl files.")
    parser.add_argument("--output-root", required=True, help="Output root for <stage>/<split>_eval.jsonl.")
    parser.add_argument("--stage", choices=EVAL_STAGES, required=True, help="Evaluation stage to build.")
    parser.add_argument("--splits", default="valid", help="Comma-separated splits to build, e.g. valid or train,valid.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit per split for preview/smoke builds.")
    parser.add_argument("--preview-dir", default=None, help="Optional directory for a readable prompt/verifier preview.")
    args = parser.parse_args(argv)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None

    for split in _parse_splits(args.splits):
        input_path = input_root / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input split: {input_path}")
        records = build_eval_records_from_jsonl(
            input_path,
            stage=args.stage,
            split=split,
            limit=args.limit,
        )
        output_path = output_root / args.stage / f"{split}_eval.jsonl"
        write_eval_records_jsonl(records, output_path)
        if preview_dir:
            _write_preview(records, preview_dir, stage=args.stage, split=split)
        print(f"wrote {len(records)} {args.stage}/{split} eval record(s) to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
