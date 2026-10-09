"""Convert a question/answer CSV into chat-format JSONL training data.

Usage:
    python scripts/prepare_finetuning_data.py data/fine_tuning_examples.csv data/training.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this script directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fine_tuning import prepare_csv_to_jsonl, validate_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning JSONL from a CSV file.")
    parser.add_argument("input_csv", help="CSV containing question/answer or prompt/completion columns")
    parser.add_argument("output_jsonl", help="Where the generated JSONL file should be written")
    args = parser.parse_args()

    try:
        count = prepare_csv_to_jsonl(args.input_csv, args.output_jsonl)
        valid_count, errors = validate_jsonl(args.output_jsonl)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Created {count} training examples: {args.output_jsonl}")
    print(f"Validation passed: {valid_count} valid JSONL examples.")
    print("Note: preparing data does not itself fine-tune a model. Submit this file to a compatible provider or training workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
