import json
import tempfile
import unittest
from pathlib import Path

from src.fine_tuning import rows_to_training_examples, validate_jsonl


class FineTuningTests(unittest.TestCase):
    def test_rows_convert_to_chat_format(self):
        examples = rows_to_training_examples([
            {"question": "What is RAG?", "answer": "Retrieval augmented generation."},
            {"question": "", "answer": "Skip missing question."},
        ])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["messages"][1]["role"], "user")
        self.assertEqual(examples[0]["messages"][2]["role"], "assistant")

    def test_jsonl_validator_accepts_valid_chat_example(self):
        example = {
            "messages": [
                {"role": "system", "content": "Be accurate."},
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "Retrieval augmented generation."},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(json.dumps(example) + "\n", encoding="utf-8")
            count, errors = validate_jsonl(path)
        self.assertEqual(count, 1)
        self.assertEqual(errors, [])

    def test_jsonl_validator_reports_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text("{broken json}\n", encoding="utf-8")
            count, errors = validate_jsonl(path)
        self.assertEqual(count, 0)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
