"""Retrieval layer: semantic embeddings when available, TF-IDF as a safe fallback."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - exercised when sklearn is unavailable in a runtime env.
    TfidfVectorizer = None
    cosine_similarity = None

from .ingestion import TextChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float
    citation: str


class _FallbackTfidfIndex:
    """Lightweight TF-IDF index used when scikit-learn is unavailable or blocked."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._token_re = re.compile(r"(?u)\b\w+\b")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in re.findall(r"(?u)\b\w+\b", text or "") if token.strip()]

    def fit(self, texts: Sequence[str]) -> None:
        tokenized = [self._tokenize(text) for text in texts]
        doc_freq: defaultdict[str, int] = defaultdict(int)
        for tokens in tokenized:
            unique = set(tokens)
            for token in unique:
                doc_freq[token] += 1

        vocabulary = sorted(doc_freq)
        self.vocabulary = {token: index for index, token in enumerate(vocabulary)}
        doc_count = max(1, len(tokenized))
        self.idf = {
            token: math.log((1.0 + doc_count) / (1.0 + doc_freq[token])) + 1.0
            for token in vocabulary
        }

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self.vocabulary:
            return np.zeros((len(texts), 0), dtype=np.float32)

        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row_index, text in enumerate(texts):
            counts = Counter(self._tokenize(text))
            if not counts:
                continue
            max_count = max(counts.values())
            for token, count in counts.items():
                index = self.vocabulary.get(token)
                if index is None:
                    continue
                tf = count / max_count if max_count else 1.0
                matrix[row_index, index] = tf * self.idf[token]
        return matrix


def _cosine_scores(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    dense_matrix = np.asarray(matrix, dtype=np.float32)
    if dense_matrix.ndim == 1:
        dense_matrix = dense_matrix.reshape(1, -1)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query))
    matrix_norms = np.linalg.norm(dense_matrix, axis=1)
    if query_norm <= 1e-12:
        return np.zeros(dense_matrix.shape[0], dtype=np.float32)
    denom = matrix_norms * query_norm
    scores = (dense_matrix @ query) / denom
    scores[denom <= 1e-12] = 0.0
    return np.clip(np.asarray(scores, dtype=np.float32), 0.0, None)


class KnowledgeRetriever:
    """Rank knowledge-base chunks by semantic or lexical relevance.

    Pass a loaded sentence-transformers model to use semantic retrieval. If the
    model is absent or embedding fails, a lightweight TF-IDF index is used.
    """

    def __init__(self, chunks: Sequence[TextChunk], embedding_model: Any | None = None):
        self.chunks = list(chunks)
        self.embedding_model = embedding_model
        self.backend = "semantic embeddings" if embedding_model is not None else "TF-IDF"
        self._embeddings: np.ndarray | None = None
        self._vectorizer: Any | None = None
        self._tfidf_matrix: Any | None = None

        if not self.chunks:
            self.backend = "empty"
            return

        texts = [chunk.text for chunk in self.chunks]
        if embedding_model is not None:
            try:
                self._embeddings = np.asarray(
                    embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
                    dtype=np.float32,
                )
                if self._embeddings.ndim != 2 or self._embeddings.shape[0] != len(self.chunks):
                    raise ValueError("Embedding model returned an unexpected shape.")
                return
            except Exception:
                self.embedding_model = None
                self._embeddings = None
                self.backend = "TF-IDF (semantic model unavailable)"

        try:
            if TfidfVectorizer is not None:
                self._vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w+\b",
                )
                self._tfidf_matrix = self._vectorizer.fit_transform(texts)
            else:
                self._vectorizer = _FallbackTfidfIndex()
                self._vectorizer.fit(texts)
                self._tfidf_matrix = self._vectorizer.transform(texts)
        except ValueError:
            # Empty/whitespace-only chunks should not crash an indexing operation.
            self.chunks = []
            self.backend = "empty"
            self._vectorizer = None
            self._tfidf_matrix = None

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return the most relevant chunks with stable citations [S1], [S2], ..."""
        query = (query or "").strip()
        if not query or not self.chunks:
            return []

        limit = max(1, min(int(top_k), 10))
        if self._embeddings is not None and self.embedding_model is not None:
            try:
                query_vector = np.asarray(
                    self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True),
                    dtype=np.float32,
                )
                scores = (self._embeddings @ query_vector[0]).reshape(-1)
            except Exception:
                # A transient embedding error is safer as no result than incorrect indexing.
                return []
        elif self._vectorizer is not None and self._tfidf_matrix is not None:
            query_vector = self._vectorizer.transform([query])
            if cosine_similarity is not None:
                scores = cosine_similarity(query_vector, self._tfidf_matrix).reshape(-1)
            else:
                scores = _cosine_scores(query_vector[0], self._tfidf_matrix)
        else:
            return []

        if not len(scores) or float(np.max(scores)) <= 1e-8:
            return []

        ordered = np.argsort(scores)[::-1]
        results: list[RetrievedChunk] = []
        for index in ordered:
            score = float(scores[index])
            if score <= 1e-8:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(index)],
                    score=score,
                    citation=f"S{len(results) + 1}",
                )
            )
            if len(results) >= limit:
                break
        return results
