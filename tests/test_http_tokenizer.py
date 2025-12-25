"""Tests for HTTP tokenizer."""

import pytest
import respx
from httpx import Response

from context7_reranker.config import TokenizerConfig
from context7_reranker.http_tokenizer import HttpTokenizer


class TestHttpTokenizer:
    """Tests for HttpTokenizer class."""

    def test_fallback_when_no_endpoint(self):
        """Should use fallback when no endpoint configured."""
        config = TokenizerConfig(endpoint=None)
        tokenizer = HttpTokenizer(config)
        result = tokenizer.count_tokens("hello world")
        # Should use LocalTokenizer fallback
        assert result > 0

    def test_fallback_tokenizer_custom(self):
        """Should use custom fallback tokenizer."""

        class FixedTokenizer:
            def count_tokens(self, text: str) -> int:
                return 42

        config = TokenizerConfig(endpoint=None)
        tokenizer = HttpTokenizer(config, fallback=FixedTokenizer())
        result = tokenizer.count_tokens("any text")
        assert result == 42


@pytest.mark.asyncio
class TestHttpTokenizerAsync:
    """Async tests for HttpTokenizer."""

    @respx.mock
    async def test_openai_format_response(self):
        """Should parse OpenAI embeddings format."""
        config = TokenizerConfig(endpoint="http://localhost:8080/v1/embeddings")
        tokenizer = HttpTokenizer(config)

        respx.post("http://localhost:8080/v1/embeddings").mock(
            return_value=Response(
                200,
                json={"usage": {"prompt_tokens": 15}},
            )
        )

        result = await tokenizer.count_tokens_async("hello world test")
        assert result == 15

    @respx.mock
    async def test_llamacpp_format_response(self):
        """Should parse llama.cpp tokenize format."""
        config = TokenizerConfig(endpoint="http://localhost:8080/tokenize")
        tokenizer = HttpTokenizer(config)

        respx.post("http://localhost:8080/tokenize").mock(
            return_value=Response(
                200,
                json={"tokens": [1, 2, 3, 4, 5]},
            )
        )

        result = await tokenizer.count_tokens_async("hello world")
        assert result == 5

    @respx.mock
    async def test_direct_count_format(self):
        """Should parse direct count format."""
        config = TokenizerConfig(endpoint="http://localhost:8080/count")
        tokenizer = HttpTokenizer(config)

        respx.post("http://localhost:8080/count").mock(
            return_value=Response(
                200,
                json={"token_count": 10},
            )
        )

        result = await tokenizer.count_tokens_async("hello")
        assert result == 10

    @respx.mock
    async def test_fallback_on_error(self):
        """Should fall back to local on HTTP error."""
        config = TokenizerConfig(endpoint="http://localhost:8080/v1/embeddings")
        tokenizer = HttpTokenizer(config)

        respx.post("http://localhost:8080/v1/embeddings").mock(
            return_value=Response(500, json={"error": "Server error"})
        )

        result = await tokenizer.count_tokens_async("hello world")
        # Should use fallback
        assert result > 0

    @respx.mock
    async def test_batch_tokens(self):
        """Should count tokens for multiple texts."""
        config = TokenizerConfig(endpoint="http://localhost:8080/v1/embeddings")
        tokenizer = HttpTokenizer(config)

        respx.post("http://localhost:8080/v1/embeddings").mock(
            return_value=Response(
                200,
                json={"usage": {"prompt_tokens": 5}},
            )
        )

        results = await tokenizer.count_tokens_batch_async(["hello", "world", "test"])
        assert len(results) == 3
        assert all(r == 5 for r in results)

    @respx.mock
    async def test_close_client(self):
        """Should close HTTP client cleanly."""
        config = TokenizerConfig(endpoint="http://localhost:8080/v1/embeddings")
        tokenizer = HttpTokenizer(config)

        respx.post("http://localhost:8080/v1/embeddings").mock(
            return_value=Response(200, json={"usage": {"prompt_tokens": 5}})
        )

        await tokenizer.count_tokens_async("hello")
        await tokenizer.close()
        assert tokenizer._client is None
