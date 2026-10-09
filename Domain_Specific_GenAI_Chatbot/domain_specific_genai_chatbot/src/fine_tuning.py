"""Prepare and validate chat-format JSONL training data for compatible fine-tuning APIs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate domain-specific assistant. Answer clearly and do not invent facts."
)


def rows_to_training_examples(
    rows: Iterable[Mapping[str, object]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, list[dict[str, str]]]]:
    """Convert records with question/answer (or prompt/completion) fields to chat JSONL."""
    examples: list[dict[str, list[dict[str, str]]]] = []
    for row_number, row in enumerate(rows, start=1):
        normalized = {str(key).strip().lower(): str(value or "").strip() for key, value in row.items()}
        question = normalized.get("question") or normalized.get("prompt") or normalized.get("input") or ""
        answer = normalized.get("answer") or normalized.get("completion") or normalized.get("response") or ""
        if not question or not answer:
            continue
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt.strip() or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return examples


def prepare_csv_to_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    """Read a CSV and write valid chat-completion training examples as JSONL."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Input CSV was not found: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input CSV contains no data rows.")

    examples = rows_to_training_examples(rows)
    if not examples:
        raise ValueError(
            "No valid examples were found. Include non-empty question/answer or prompt/completion columns."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def validate_jsonl(path: str | Path) -> tuple[int, list[str]]:
    """Validate one JSON object per line and check chat message roles/content."""
    file_path = Path(path)
    errors: list[str] = []
    count = 0
    if not file_path.exists():
        return 0, [f"File was not found: {file_path}"]

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: invalid JSON ({exc.msg}).")
                continue
            messages = example.get("messages") if isinstance(example, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"Line {line_number}: expected a 'messages' list with at least two items.")
                continue
            roles = [message.get("role") for message in messages if isinstance(message, dict)]
            if "user" not in roles or "assistant" not in roles:
                errors.append(f"Line {line_number}: messages must include user and assistant roles.")
                continue
            if any(not isinstance(message, dict) or not str(message.get("content", "")).strip() for message in messages):
                errors.append(f"Line {line_number}: every message must have non-empty content.")
                continue
            count += 1
    if count == 0 and not errors:
        errors.append("No training examples were found.")
    return count, errors
