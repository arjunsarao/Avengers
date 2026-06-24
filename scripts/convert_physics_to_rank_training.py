#!/usr/bin/env python3
"""Convert raw PHYSICS benchmark JSONL files to rank-router training rows.

The raw PHYSICS benchmark contains questions and gold solutions, but not
per-model correctness. This script preserves the benchmark metadata and emits
empty ``records`` maps that can later be populated from evaluation outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def category_from_path(path: Path) -> str:
    name = path.name
    suffix = "_dataset_textonly.jsonl"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def convert_file(path: Path) -> list[dict]:
    category = category_from_path(path)
    rows = []
    for item in iter_jsonl(path):
        question = item.get("questions")
        if not question:
            raise ValueError(f"Missing 'questions' field in item {item.get('id')} from {path}")

        rows.append(
            {
                "id": item.get("id"),
                "query": question,
                "dataset": f"PHYSICS/{category}",
                "records": {},
                "metadata": {
                    "source_file": str(path),
                    "solutions": item.get("solutions"),
                    "final_answers": item.get("final_answers"),
                    "graphs": item.get("graphs"),
                },
            }
        )
    return rows


def resolve_input_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw PHYSICS JSONL benchmark files to rank-router training format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/PHYSICS"),
        help="PHYSICS directory or a single raw PHYSICS .jsonl file.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/physics_rank_router_training.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    rows = []
    for path in resolve_input_files(args.input_dir):
        rows.extend(convert_file(path))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output_path}")
    print("Note: records are empty; populate them with per-model correctness before training a useful router.")


if __name__ == "__main__":
    main()
