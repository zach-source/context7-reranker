"""HTTP-based reranker implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from context7_reranker.chunker import DocChunk
from context7_reranker.config import RerankerConfig
from context7_reranker.http_client import HttpClient
from context7_reranker.protocols import BaseReranker
from context7_reranker.reranker import TfidfReranker


class HttpReranker(BaseReranker):
    """Reranker that calls an external HTTP endpoint."""

    def __init__(
        self,
        config: RerankerConfig,
        fallback: BaseReranker | None = None,
    ):
        """Initialize HTTP reranker.

        Args:
            config: Reranker configuration.
            fallback: Fallback reranker for errors (defaults to TfidfReranker).
        """
        self.config = config
        self.fallback = fallback or TfidfReranker()
        self._client: HttpClient | None = None

    @property
    def client(self) -> HttpClient:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.config.endpoint:
                raise ValueError("No endpoint configured for HTTP reranker")
            self._client = HttpClient(
                base_url=self.config.endpoint,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def rerank(
        self,
        chunks: list[DocChunk],
        query: str,
        top_k: int = 5,
    ) -> list[DocChunk]:
        """Rerank chunks via HTTP endpoint (sync wrapper).

        Falls back to local reranker if in async context or on error.
        """
        if not self.config.endpoint or not chunks:
            return self.fallback.rerank(chunks, query, top_k)

        try:
            loop = asyncio.get_running_loop()
            # Already in async context, use fallback to avoid blocking
            return self.fallback.rerank(chunks, query, top_k)
        except RuntimeError:
            # No event loop running, we can create one
            pass

        try:
            return asyncio.run(self.rerank_async(chunks, query, top_k))
        except Exception:
            return self.fallback.rerank(chunks, query, top_k)

    async def rerank_async(
        self,
        chunks: list[DocChunk],
        query: str,
        top_k: int = 5,
    ) -> list[DocChunk]:
        """Rerank chunks via HTTP endpoint.

        Args:
            chunks: Document chunks to rerank.
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            Reranked chunks from API or fallback on error.
        """
        if not self.config.endpoint or not chunks:
            return self.fallback.rerank(chunks, query, top_k)

        try:
            request_body = self._build_request(chunks, query, top_k)
            response = await self.client.post_with_retry("", request_body)

            if response is None:
                return self.fallback.rerank(chunks, query, top_k)

            return self._parse_response(response, chunks, top_k)
        except Exception:
            return self.fallback.rerank(chunks, query, top_k)

    def _build_request(
        self,
        chunks: list[DocChunk],
        query: str,
        top_k: int,
    ) -> dict[str, Any]:
        """Build request based on configured format.

        Supports:
        - cohere: Cohere /v1/rerank format (llama.cpp, vLLM, etc.)
        - openai: OpenAI-style custom endpoints
        - custom: Generic format
        """
        documents = [chunk.content for chunk in chunks]

        if self.config.format == "cohere":
            # Cohere /v1/rerank format (used by llama.cpp, vLLM, etc.)
            return {
                "model": self.config.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
                "return_documents": False,
            }
        elif self.config.format == "openai":
            # Some custom OpenAI-style endpoints
            return {
                "model": self.config.model,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "top_k": top_k,
            }
        else:
            # Custom/generic format - send everything
            return {
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "model": self.config.model,
            }

    def _parse_response(
        self,
        data: dict[str, Any],
        chunks: list[DocChunk],
        top_k: int,
    ) -> list[DocChunk]:
        """Parse response based on format.

        Handles:
        - Cohere: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        - Custom: {"data": [...]} or {"rankings": [...]}
        """
        # Try different result field names
        results = data.get("results", data.get("data", data.get("rankings", [])))

        if not results:
            return self.fallback.rerank(chunks, "", top_k)

        # Build scored chunks from results
        scored_chunks = []
        for result in results[:top_k]:
            # Handle different index field names
            idx = result.get(
                "index", result.get("document_index", result.get("doc_id", 0))
            )
            # Handle different score field names
            score = result.get(
                "relevance_score",
                result.get("score", result.get("similarity", 0.0)),
            )

            if 0 <= idx < len(chunks):
                original = chunks[idx]
                scored_chunks.append(
                    DocChunk(
                        content=original.content,
                        source=original.source,
                        tokens=original.tokens,
                        score=float(score),
                    )
                )

        return scored_chunks

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
