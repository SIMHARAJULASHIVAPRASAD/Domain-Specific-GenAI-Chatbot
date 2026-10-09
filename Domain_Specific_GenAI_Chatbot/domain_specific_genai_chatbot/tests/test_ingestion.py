import unittest

from src.ingestion import chunk_text, parse_file_to_chunks


class IngestionTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text("   \n  "), [])

    def test_chunker_advances_and_preserves_content(self):
        text = ("RAG combines retrieval and generation. " * 80).strip()
        chunks = chunk_text(text, chunk_size=250, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(piece.strip() for piece in chunks))
        self.assertIn("retrieval and generation", " ".join(chunks))

    def test_text_file_parses_into_citable_chunks(self):
        content = b"Python helps automate data processing.\nSQL queries relational tables."
        chunks = parse_file_to_chunks("notes.TXT", content, chunk_size=400, overlap=30)
        self.assertEqual(chunks[0].source, "notes.TXT")
        self.assertEqual(chunks[0].chunk_id, 1)
        self.assertIn("Python", chunks[0].text)

    def test_unsupported_file_type_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            parse_file_to_chunks("notes.exe", b"not text")

    def test_empty_upload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_file_to_chunks("notes.txt", b"")


if __name__ == "__main__":
    unittest.main()
