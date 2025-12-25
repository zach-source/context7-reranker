"""Tests for HTTP reranker."""

import pytest
import respx
from httpx import Response

from context7_reranker.chunker import DocChunk
from context7_reranker.config import RerankerConfig
from context7_reranker.http_reranker import HttpReranker


class TestHttpReranker:
    """Tests for HttpReranker class."""

    def test_fallback_when_no_endpoint(self):
        """Should use fallback when no endpoint configured."""
        config = RerankerConfig(endpoint=None)
        reranker = HttpReranker(config)

        chunks = [
            DocChunk(content="Python is great", source="test", tokens=3),
            DocChunk(content="JavaScript rocks", source="test", tokens=3),
        ]

        result = reranker.rerank(chunks, "Python programming", top_k=1)
        assert len(result) == 1
        # TfidfReranker should rank Python higher
        assert "Python" in result[0].content

    def test_fallback_when_empty_chunks(self):
        """Should return empty list for empty chunks."""
        config = RerankerConfig(endpoint="http://localhost:8080/v1/rerank")
        reranker = HttpReranker(config)

        result = reranker.rerank([], "query", top_k=5)
        assert result == []


@pytest.mark.asyncio
class TestHttpRerankerAsync:
    """Async tests for HttpReranker."""

    @respx.mock
    async def test_cohere_format_response(self):
        """Should parse Cohere rerank format."""
        config = RerankerConfig(
            endpoint="http://localhost:8080/v1/rerank",
            format="cohere",
            model="bge-reranker",
        )
        reranker = HttpReranker(config)

        chunks = [
            DocChunk(content="Python is great", source="test", tokens=3),
            DocChunk(content="JavaScript rocks", source="test", tokens=3),
            DocChunk(content="Go is fast", source="test", tokens=3),
        ]

        respx.post("http://localhost:8080/v1/rerank").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.95},
                        {"index": 2, "relevance_score": 0.75},
                    ]
                },
            )
        )

        result = await reranker.rerank_async(chunks, "Python programming", top_k=2)
        assert len(result) == 2
        assert result[0].content == "Python is great"
        assert result[0].score == 0.95
        assert result[1].content == "Go is fast"
        assert result[1].score == 0.75

    @respx.mock
    async def test_openai_format_request(self):
        """Should build OpenAI format request."""
        config = RerankerConfig(
            endpoint="http://localhost:8080/v1/rerank",
            format="openai",
            model="rerank-model",
        )
        reranker = HttpReranker(config)

        chunks = [DocChunk(content="test content", source="test", tokens=2)]

        respx.post("http://localhost:8080/v1/rerank").mock(
            return_value=Response(
                200,
                json={"results": [{"index": 0, "relevance_score": 0.9}]},
            )
        )

        await reranker.rerank_async(chunks, "query", top_k=1)

        # Check request was made with OpenAI format
        request = respx.calls[0].request
        import json

        body = json.loads(request.content)
        assert "input" in body
        assert body["input"]["query"] == "query"

    @respx.mock
    async def test_alternative_response_fields(self):
        """Should handle alternative field names in response."""
        config = RerankerConfig(endpoint="http://localhost:8080/v1/rerank")
        reranker = HttpReranker(config)

        chunks = [
            DocChunk(content="doc1", source="test", tokens=1),
            DocChunk(content="doc2", source="test", tokens=1),
        ]

        # Alternative format with 'rankings' and 'score'
        # instead of 'results' and 'relevance_score'
        respx.post("http://localhost:8080/v1/rerank").mock(
            return_value=Response(
                200,
                json={
                    "rankings": [
                        {"document_index": 1, "score": 0.8},
                        {"document_index": 0, "score": 0.6},
                    ]
                },
            )
        )

        result = await reranker.rerank_async(chunks, "query", top_k=2)
        assert len(result) == 2
        assert result[0].content == "doc2"
        assert result[0].score == 0.8

    @respx.mock
    async def test_fallback_on_error(self):
        """Should fall back to TF-IDF on HTTP error."""
        config = RerankerConfig(endpoint="http://localhost:8080/v1/rerank")
        reranker = HttpReranker(config)

        chunks = [
            DocChunk(content="Python programming", source="test", tokens=2),
            DocChunk(content="JavaScript coding", source="test", tokens=2),
        ]

        respx.post("http://localhost:8080/v1/rerank").mock(
            return_value=Response(500, json={"error": "Server error"})
        )

        result = await reranker.rerank_async(chunks, "Python", top_k=1)
        # Should get result from fallback
        assert len(result) == 1

    @respx.mock
    async def test_close_client(self):
        """Should close HTTP client cleanly."""
        config = RerankerConfig(endpoint="http://localhost:8080/v1/rerank")
        reranker = HttpReranker(config)

        chunks = [DocChunk(content="test", source="test", tokens=1)]

        respx.post("http://localhost:8080/v1/rerank").mock(
            return_value=Response(
                200,
                json={"results": [{"index": 0, "relevance_score": 0.9}]},
            )
        )

        await reranker.rerank_async(chunks, "query", top_k=1)
        await reranker.close()
        assert reranker._client is None


class TestBuildRequest:
    """Tests for request building."""

    def test_cohere_format(self):
        """Should build Cohere format request."""
        config = RerankerConfig(format="cohere", model="bge-reranker")
        reranker = HttpReranker(config)

        chunks = [DocChunk(content="doc1", source="s", tokens=1)]
        request = reranker._build_request(chunks, "query", 5)

        assert request["model"] == "bge-reranker"
        assert request["query"] == "query"
        assert request["documents"] == ["doc1"]
        assert request["top_n"] == 5
        assert request["return_documents"] is False

    def test_openai_format(self):
        """Should build OpenAI format request."""
        config = RerankerConfig(format="openai", model="rerank")
        reranker = HttpReranker(config)

        chunks = [DocChunk(content="doc1", source="s", tokens=1)]
        request = reranker._build_request(chunks, "query", 3)

        assert request["model"] == "rerank"
        assert request["input"]["query"] == "query"
        assert request["input"]["documents"] == ["doc1"]
        assert request["top_k"] == 3

    def test_custom_format(self):
        """Should build custom format request."""
        config = RerankerConfig(format="custom", model="custom-model")
        reranker = HttpReranker(config)

        chunks = [DocChunk(content="doc1", source="s", tokens=1)]
        request = reranker._build_request(chunks, "query", 2)

        assert request["model"] == "custom-model"
        assert request["query"] == "query"
        assert request["documents"] == ["doc1"]
        assert request["top_k"] == 2
