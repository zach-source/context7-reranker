"""HTTP-based tokenizer implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from context7_reranker.config import TokenizerConfig
from context7_reranker.http_client import HttpClient
from context7_reranker.protocols import BaseTokenizer
from context7_reranker.tokenizer import LocalTokenizer


class HttpTokenizer(BaseTokenizer):
    """Tokenizer that calls an external HTTP endpoint."""

    def __init__(
        self,
        config: TokenizerConfig,
        fallback: BaseTokenizer | None = None,
    ):
        """Initialize HTTP tokenizer.

        Args:
            config: Tokenizer configuration.
            fallback: Fallback tokenizer for errors (defaults to LocalTokenizer).
        """
        self.config = config
        self.fallback = fallback or LocalTokenizer()
        self._client: HttpClient | None = None

    @property
    def client(self) -> HttpClient:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.config.endpoint:
                raise ValueError("No endpoint configured for HTTP tokenizer")
            self._client = HttpClient(
                base_url=self.config.endpoint,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def count_tokens(self, text: str) -> int:
        """Count tokens via HTTP endpoint (sync wrapper).

        Falls back to local tokenizer if in async context or on error.
        """
        if not self.config.endpoint:
            return self.fallback.count_tokens(text)

        try:
            loop = asyncio.get_running_loop()
            # Already in async context, use fallback to avoid blocking
            return self.fallback.count_tokens(text)
        except RuntimeError:
            # No event loop running, we can create one
            pass

        try:
            return asyncio.run(self.count_tokens_async(text))
        except Exception:
            return self.fallback.count_tokens(text)

    async def count_tokens_async(self, text: str) -> int:
        """Count tokens via HTTP endpoint.

        Args:
            text: Text to tokenize.

        Returns:
            Token count from API or fallback on error.
        """
        if not self.config.endpoint:
            return self.fallback.count_tokens(text)

        try:
            # OpenAI-compatible embeddings format
            response = await self.client.post_with_retry(
                "",
                {
                    "input": text,
                    "model": self.config.model,
                    "encoding_format": "float",
                },
            )
            if response is None:
                return self.fallback.count_tokens(text)

            return self._extract_token_count(response, text)
        except Exception:
            return self.fallback.count_tokens(text)

    def _extract_token_count(self, data: dict[str, Any], text: str) -> int:
        """Extract token count from API response.

        Supports multiple response formats:
        - OpenAI: {"usage": {"prompt_tokens": N}}
        - llama.cpp: {"tokens": [...]} or {"token_count": N}
        - Custom: {"count": N}
        """
        # OpenAI embeddings format
        if "usage" in data:
            return data["usage"].get("prompt_tokens", 0)

        # llama.cpp tokenize format
        if "tokens" in data:
            tokens = data["tokens"]
            if isinstance(tokens, list):
                return len(tokens)

        # Direct count fields
        for field in ["token_count", "count", "num_tokens", "length"]:
            if field in data:
                return int(data[field])

        # Fallback
        return self.fallback.count_tokens(text)

    async def count_tokens_batch_async(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts concurrently.

        Args:
            texts: List of texts to tokenize.

        Returns:
            List of token counts.
        """
        if not self.config.endpoint:
            return [self.fallback.count_tokens(t) for t in texts]

        # Concurrent requests with semaphore to limit parallelism
        sem = asyncio.Semaphore(10)

        async def count_with_limit(text: str) -> int:
            async with sem:
                return await self.count_tokens_async(text)

        return await asyncio.gather(*[count_with_limit(t) for t in texts])

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
