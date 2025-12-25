"""TF-IDF reranking utilities with pluggable backends."""

from __future__ import annotations

import re
from collections import Counter
from math import log

from context7_reranker.chunker import DocChunk
from context7_reranker.protocols import BaseReranker

# Common English stopwords to filter from term extraction
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "by",
    "from",
    "as",
    "or",
    "and",
    "if",
    "then",
    "else",
    "when",
    "where",
    "which",
    "who",
    "what",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "here",
    "there",
    "but",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "once",
}


class TfidfReranker(BaseReranker):
    """Local TF-IDF based reranker."""

    def __init__(self, stopwords: set[str] | None = None):
        """Initialize TF-IDF reranker.

        Args:
            stopwords: Custom stopwords set (defaults to STOPWORDS).
        """
        self.stopwords = stopwords if stopwords is not None else STOPWORDS

    def extract_terms(self, text: str) -> list[str]:
        """Extract terms from text for TF-IDF scoring.

        Args:
            text: The text to extract terms from.

        Returns:
            List of lowercase terms with stopwords removed.
        """
        # Lowercase and extract words (including underscores for code identifiers)
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
        # Filter stopwords and short words
        return [w for w in words if w not in self.stopwords and len(w) > 2]

    def compute_tfidf_score(
        self,
        query_terms: list[str],
        doc_terms: list[str],
        idf: dict[str, float],
    ) -> float:
        """Compute TF-IDF similarity score between query and document.

        Args:
            query_terms: Terms extracted from the query.
            doc_terms: Terms extracted from the document.
            idf: Inverse document frequency mapping.

        Returns:
            TF-IDF similarity score.
        """
        if not doc_terms or not query_terms:
            return 0.0

        # Term frequency in document
        tf = Counter(doc_terms)
        doc_len = len(doc_terms)

        # Score based on query term matches
        score = 0.0
        for term in query_terms:
            if term in tf:
                term_freq = tf[term] / doc_len
                term_idf = idf.get(term, 1.0)
                score += term_freq * term_idf

        return score

    def rerank(
        self,
        chunks: list[DocChunk],
        query: str,
        top_k: int = 5,
    ) -> list[DocChunk]:
        """Rerank chunks by relevance to query using TF-IDF.

        Args:
            chunks: List of document chunks to rerank.
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            Top-k chunks sorted by relevance score (descending).
        """
        if not chunks:
            return []

        query_terms = self.extract_terms(query)
        if not query_terms:
            return chunks[:top_k]

        # Build IDF from all chunks
        doc_count = len(chunks)
        term_doc_freq: Counter[str] = Counter()
        chunk_terms = []

        for chunk in chunks:
            terms = self.extract_terms(chunk.content)
            chunk_terms.append(terms)
            unique_terms = set(terms)
            for term in unique_terms:
                term_doc_freq[term] += 1

        # Compute IDF with proper Laplace smoothing: log((N + 1) / (df + 1)) + 1
        idf = {
            term: log((doc_count + 1) / (freq + 1)) + 1
            for term, freq in term_doc_freq.items()
        }

        # Score each chunk (create new instances to avoid mutating input)
        scored_chunks = [
            DocChunk(
                content=chunk.content,
                source=chunk.source,
                tokens=chunk.tokens,
                score=self.compute_tfidf_score(query_terms, chunk_terms[i], idf),
            )
            for i, chunk in enumerate(chunks)
        ]

        # Sort by score descending
        ranked = sorted(scored_chunks, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]


# Default reranker instance
_default_reranker: BaseReranker | None = None


def get_default_reranker() -> BaseReranker:
    """Get the default reranker instance.

    Returns:
        The default reranker (creates TfidfReranker if not set).
    """
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = TfidfReranker()
    return _default_reranker


def set_default_reranker(reranker: BaseReranker) -> None:
    """Set the default reranker instance.

    Args:
        reranker: The reranker to use as default.
    """
    global _default_reranker
    _default_reranker = reranker


# Backward-compatible module-level functions


def extract_terms(text: str) -> list[str]:
    """Extract terms from text for TF-IDF scoring.

    Args:
        text: The text to extract terms from.

    Returns:
        List of lowercase terms with stopwords removed.
    """
    reranker = get_default_reranker()
    if isinstance(reranker, TfidfReranker):
        return reranker.extract_terms(text)
    # Fallback for non-TF-IDF rerankers
    words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def compute_tfidf_score(
    query_terms: list[str],
    doc_terms: list[str],
    idf: dict[str, float],
) -> float:
    """Compute TF-IDF similarity score between query and document.

    Args:
        query_terms: Terms extracted from the query.
        doc_terms: Terms extracted from the document.
        idf: Inverse document frequency mapping.

    Returns:
        TF-IDF similarity score.
    """
    reranker = get_default_reranker()
    if isinstance(reranker, TfidfReranker):
        return reranker.compute_tfidf_score(query_terms, doc_terms, idf)
    # Fallback implementation
    if not doc_terms or not query_terms:
        return 0.0
    tf = Counter(doc_terms)
    doc_len = len(doc_terms)
    score = 0.0
    for term in query_terms:
        if term in tf:
            term_freq = tf[term] / doc_len
            term_idf = idf.get(term, 1.0)
            score += term_freq * term_idf
    return score


def rerank_chunks(
    chunks: list[DocChunk],
    query: str,
    top_k: int = 5,
) -> list[DocChunk]:
    """Rerank chunks by relevance to query.

    Args:
        chunks: List of document chunks to rerank.
        query: The search query.
        top_k: Number of top results to return (default 5).

    Returns:
        Top-k chunks sorted by relevance score (descending).
    """
    return get_default_reranker().rerank(chunks, query, top_k)
