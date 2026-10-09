import unittest

from src.ingestion import TextChunk
from src.llm import grounded_fallback, generate_answer
from src.rag import KnowledgeRetriever


class RagTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            TextChunk(
                text="Retrieval augmented generation retrieves relevant passages and supplies them to an LLM as context.",
                source="rag.md",
                chunk_id=1,
            ),
            TextChunk(
                text="SQL JOIN combines records from related relational database tables.",
                source="sql.md",
                chunk_id=1,
            ),
        ]

    def test_tfidf_retrieves_relevant_passage_and_citation(self):
        retriever = KnowledgeRetriever(self.chunks)
        results = retriever.retrieve("How does retrieval augmented generation provide context to an LLM?", top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "rag.md")
        self.assertEqual(results[0].citation, "S1")

    def test_unrelated_query_returns_no_results(self):
        retriever = KnowledgeRetriever(self.chunks)
        self.assertEqual(retriever.retrieve("volcano oceanic magma", top_k=3), [])

    def test_fallback_includes_citation_and_source(self):
        retriever = KnowledgeRetriever(self.chunks)
        results = retriever.retrieve("SQL JOIN relational tables")
        answer = grounded_fallback(results)
        self.assertIn("[S1]", answer)
        self.assertIn("sql.md", answer)

    def test_generate_without_api_returns_grounded_fallback(self):
        retriever = KnowledgeRetriever(self.chunks)
        results = retriever.retrieve("SQL JOIN relational tables")
        answer = generate_answer("What is a JOIN?", results, "Databases")
        self.assertIn("retrieved passages", answer)
        self.assertIn("sql.md", answer)


if __name__ == "__main__":
    unittest.main()
