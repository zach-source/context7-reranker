"""Abstract base classes for pluggable backends."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context7_reranker.chunker import DocChunk


@dataclass
class RerankResult:
    """Result from a reranking operation."""

    chunks: list[DocChunk]
    model: str | None = None
    usage: dict | None = None


class BaseTokenizer(ABC):
    """Abstract base class for tokenizer implementations."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: The text to tokenize.

        Returns:
            Token count.
        """
        pass

    async def count_tokens_async(self, text: str) -> int:
        """Async version of count_tokens.

        Default implementation wraps sync method.
        Override for true async behavior.
        """
        return self.count_tokens(text)

    async def count_tokens_batch_async(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts concurrently.

        Args:
            texts: List of texts to tokenize.

        Returns:
            List of token counts.
        """
        return await asyncio.gather(*[self.count_tokens_async(t) for t in texts])


class BaseReranker(ABC):
    """Abstract base class for reranker implementations."""

    @abstractmethod
    def rerank(
        self,
        chunks: list[DocChunk],
        query: str,
        top_k: int = 5,
    ) -> list[DocChunk]:
        """Rerank chunks by relevance to query.

        Args:
            chunks: Document chunks to rerank.
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            Top-k chunks sorted by relevance score (descending).
        """
        pass

    async def rerank_async(
        self,
        chunks: list[DocChunk],
        query: str,
        top_k: int = 5,
    ) -> list[DocChunk]:
        """Async version of rerank.

        Default implementation wraps sync method.
        Override for true async behavior.
        """
        return self.rerank(chunks, query, top_k)


class BaseChunker(ABC):
    """Abstract base class for chunker implementations."""

    @abstractmethod
    def split(
        self,
        content: str,
        source: str = "",
        max_chunk_tokens: int = 1000,
    ) -> list[DocChunk]:
        """Split content into chunks.

        Args:
            content: The content to split.
            source: Origin identifier for the chunks.
            max_chunk_tokens: Maximum tokens per chunk.

        Returns:
            List of DocChunk objects.
        """
        pass

    async def split_async(
        self,
        content: str,
        source: str = "",
        max_chunk_tokens: int = 1000,
    ) -> list[DocChunk]:
        """Async version of split.

        Default implementation wraps sync method.
        Override for true async behavior.
        """
        return self.split(content, source, max_chunk_tokens)
