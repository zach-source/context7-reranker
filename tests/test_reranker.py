"""Tests for reranker module."""

from context7_reranker.chunker import DocChunk
from context7_reranker.reranker import (
    compute_tfidf_score,
    extract_terms,
    rerank_chunks,
)


class TestExtractTerms:
    """Tests for extract_terms function."""

    def test_basic_extraction(self):
        terms = extract_terms("Hello world from Python")
        assert "hello" in terms
        assert "world" in terms
        assert "python" in terms

    def test_filters_stopwords(self):
        terms = extract_terms("the quick brown fox")
        assert "the" not in terms
        assert "quick" in terms
        assert "brown" in terms
        assert "fox" in terms

    def test_filters_short_words(self):
        terms = extract_terms("a is to be or")
        assert len(terms) == 0

    def test_preserves_code_identifiers(self):
        terms = extract_terms("user_id get_data handle_request")
        assert "user_id" in terms
        assert "get_data" in terms
        assert "handle_request" in terms

    def test_empty_string(self):
        assert extract_terms("") == []


class TestComputeTfidfScore:
    """Tests for compute_tfidf_score function."""

    def test_matching_terms_score_positive(self):
        query = ["authentication", "middleware"]
        doc = ["authentication", "middleware", "express", "route"]
        idf = {"authentication": 2.0, "middleware": 1.5, "express": 1.0, "route": 1.0}
        score = compute_tfidf_score(query, doc, idf)
        assert score > 0

    def test_no_matching_terms(self):
        query = ["react", "component"]
        doc = ["express", "middleware", "route"]
        idf = {"react": 2.0, "component": 1.5}
        score = compute_tfidf_score(query, doc, idf)
        assert score == 0

    def test_empty_query(self):
        score = compute_tfidf_score([], ["doc", "terms"], {})
        assert score == 0

    def test_empty_doc(self):
        score = compute_tfidf_score(["query"], [], {})
        assert score == 0


class TestRerankChunks:
    """Tests for rerank_chunks function."""

    def test_empty_chunks(self):
        result = rerank_chunks([], "query")
        assert result == []

    def test_returns_top_k(self):
        chunks = [
            DocChunk(content=f"Content {i}", source="test", tokens=10)
            for i in range(10)
        ]
        result = rerank_chunks(chunks, "query", top_k=3)
        assert len(result) == 3

    def test_ranks_relevant_first(self):
        chunks = [
            DocChunk(content="Cookie management and sessions", source="", tokens=10),
            DocChunk(content="Error handling patterns", source="", tokens=10),
            DocChunk(
                content="Authentication middleware for Express.js",
                source="",
                tokens=10,
            ),
        ]
        result = rerank_chunks(chunks, "authentication middleware", top_k=3)
        # Authentication chunk should be first
        assert "authentication" in result[0].content.lower()

    def test_assigns_scores(self):
        chunks = [
            DocChunk(content="React components", source="", tokens=10),
            DocChunk(content="Vue.js templates", source="", tokens=10),
        ]
        result = rerank_chunks(chunks, "React", top_k=2)
        # All should have scores assigned
        assert all(c.score >= 0 for c in result)

    def test_preserves_chunk_metadata(self):
        chunks = [
            DocChunk(content="Test content", source="context7", tokens=50),
        ]
        result = rerank_chunks(chunks, "test", top_k=1)
        assert result[0].source == "context7"
        assert result[0].tokens == 50

    def test_stopwords_only_query(self):
        # Query with only stopwords should return chunks without scoring
        chunks = [
            DocChunk(content="React components", source="", tokens=10),
            DocChunk(content="Vue.js templates", source="", tokens=10),
        ]
        result = rerank_chunks(chunks, "the is a to", top_k=2)
        # Should return chunks even with stopwords-only query
        assert len(result) == 2
        # All scores should be 0 since no meaningful terms to match
        assert all(c.score == 0 for c in result)

    def test_does_not_mutate_input_chunks(self):
        original_chunks = [
            DocChunk(content="Test content", source="test", tokens=10),
            DocChunk(content="Other content", source="test", tokens=10),
        ]
        # Store original scores
        original_scores = [c.score for c in original_chunks]

        # Rerank should not mutate original
        result = rerank_chunks(original_chunks, "test", top_k=2)

        # Original chunks should still have their original scores
        for i, chunk in enumerate(original_chunks):
            assert chunk.score == original_scores[i]

        # Result chunks should be different objects
        assert result[0] is not original_chunks[0]

    def test_unicode_query_and_content(self):
        chunks = [
            DocChunk(content="日本語のドキュメント", source="", tokens=10),
            DocChunk(content="English documentation", source="", tokens=10),
        ]
        # Should handle unicode without errors
        result = rerank_chunks(chunks, "日本語", top_k=2)
        assert len(result) == 2


class TestExtractTermsEdgeCases:
    """Edge case tests for extract_terms function."""

    def test_numbers_filtered(self):
        # Pure numbers should be filtered out
        terms = extract_terms("version 123 build 456")
        assert "123" not in terms
        assert "456" not in terms
        assert "version" in terms
        assert "build" in terms

    def test_mixed_case_normalized(self):
        terms = extract_terms("React REACT react")
        # All should normalize to lowercase
        assert terms.count("react") == 3

    def test_code_identifiers_with_underscores(self):
        terms = extract_terms("my_function MY_CONSTANT _private")
        assert "my_function" in terms
        assert "my_constant" in terms
        assert "_private" in terms
