"""Optionally upload validated JSONL and start an OpenAI fine-tuning job.

A dry run is the default. Pass --submit to perform the API upload and create a
potentially billable training job. Confirm model availability and pricing first.

Example:
    python scripts/start_openai_finetuning.py data/fine_tuning_examples.jsonl --model YOUR_SUPPORTED_BASE_MODEL
    python scripts/start_openai_finetuning.py data/fine_tuning_examples.jsonl --model YOUR_SUPPORTED_BASE_MODEL --submit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fine_tuning import validate_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate JSONL and optionally launch an OpenAI fine-tuning job.")
    parser.add_argument("training_jsonl", help="Chat-format JSONL training file")
    parser.add_argument("--model", required=True, help="Provider-supported fine-tunable base model ID")
    parser.add_argument("--submit", action="store_true", help="Actually upload the file and create the training job")
    args = parser.parse_args()

    training_path = Path(args.training_jsonl)
    count, errors = validate_jsonl(training_path)
    if errors:
        print("Training data validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    print(f"Validated {count} JSONL training examples.")
    print(f"Requested base model: {args.model}")
    if not args.submit:
        print("Dry run only: no file was uploaded and no training job was created.")
        print("Check provider model availability, data requirements, and pricing before re-running with --submit.")
        return 0

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is required when --submit is used.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        with training_path.open("rb") as handle:
            uploaded_file = client.files.create(file=handle, purpose="fine-tune")
        job = client.fine_tuning.jobs.create(training_file=uploaded_file.id, model=args.model)
    except Exception as exc:
        safe_message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
        print(f"Could not start the fine-tuning job: {safe_message}", file=sys.stderr)
        return 1

    print("Fine-tuning job submitted successfully.")
    print(f"Training file ID: {uploaded_file.id}")
    print(f"Fine-tuning job ID: {job.id}")
    print(f"Initial status: {getattr(job, 'status', 'unknown')}")
    print("Store the job ID and monitor it in the provider dashboard or API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
