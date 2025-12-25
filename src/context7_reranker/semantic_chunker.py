"""Semantic chunking using sentence embeddings."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from context7_reranker.chunker import DocChunk, RegexChunker
from context7_reranker.config import ChunkerConfig
from context7_reranker.protocols import BaseChunker, BaseTokenizer

if TYPE_CHECKING:
    pass


class SemanticChunker(BaseChunker):
    """Chunks based on semantic similarity between sentences.

    Uses sentence-transformers to compute embeddings and groups
    semantically similar sentences together.
    """

    def __init__(
        self,
        model: str = "all-mpnet-base-v1",
        threshold: float = 0.5,
        tokenizer: BaseTokenizer | None = None,
    ):
        """Initialize semantic chunker.

        Args:
            model: sentence-transformers model name.
            threshold: Similarity threshold (0-1). Lower = more splits.
            tokenizer: Tokenizer for counting tokens (uses default if None).
        """
        self._model_name = model
        self.threshold = threshold
        self._tokenizer = tokenizer
        self._model: Any = None
        self._nltk_ready = False

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Get tokenizer, creating default if needed."""
        if self._tokenizer is None:
            from context7_reranker.tokenizer import get_default_tokenizer

            self._tokenizer = get_default_tokenizer()
        return self._tokenizer

    def _ensure_model(self) -> Any:
        """Lazy load sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required for SemanticChunker. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    def _ensure_nltk(self) -> None:
        """Ensure NLTK punkt tokenizer is available."""
        if self._nltk_ready:
            return
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            self._nltk_ready = True
        except ImportError as e:
            raise ImportError(
                "nltk required for SemanticChunker. Install with: pip install nltk"
            ) from e

    def _sent_tokenize(self, text: str) -> list[str]:
        """Split text into sentences using NLTK."""
        self._ensure_nltk()
        import nltk

        return nltk.sent_tokenize(text)

    def _compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            sim = cosine_similarity(
                np.array(embedding1).reshape(1, -1),
                np.array(embedding2).reshape(1, -1),
            )
            return float(sim[0][0])
        except ImportError as e:
            raise ImportError(
                "scikit-learn required for SemanticChunker. "
                "Install with: pip install scikit-learn"
            ) from e

    def split(
        self,
        content: str,
        source: str = "",
        max_chunk_tokens: int = 1000,
    ) -> list[DocChunk]:
        """Split content into semantically coherent chunks.

        Groups sentences with semantic similarity above threshold.
        Respects max_chunk_tokens by splitting oversized groups.

        Args:
            content: The text content to split.
            source: Origin identifier for the chunks.
            max_chunk_tokens: Maximum tokens per chunk.

        Returns:
            List of DocChunk objects.
        """
        if not content.strip():
            return []

        sentences = self._sent_tokenize(content)
        if not sentences:
            return []

        if len(sentences) == 1:
            tokens = self.tokenizer.count_tokens(sentences[0])
            return [DocChunk(content=sentences[0], source=source, tokens=tokens)]

        # Get embeddings for all sentences
        model = self._ensure_model()
        embeddings = model.encode(sentences)

        # Group sentences by semantic similarity
        chunks: list[DocChunk] = []
        current_sentences: list[str] = [sentences[0]]
        current_tokens = self.tokenizer.count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            # Check semantic similarity with previous sentence
            similarity = self._compute_similarity(
                embeddings[i - 1].tolist(), embeddings[i].tolist()
            )

            # Check if we should start a new chunk
            should_split = (
                similarity < self.threshold
                or current_tokens + sentence_tokens > max_chunk_tokens
            )

            if should_split and current_sentences:
                # Save current chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(
                    DocChunk(
                        content=chunk_content,
                        source=source,
                        tokens=current_tokens,
                    )
                )
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget last chunk
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append(
                DocChunk(
                    content=chunk_content,
                    source=source,
                    tokens=current_tokens,
                )
            )

        return chunks


class HttpSemanticChunker(BaseChunker):
    """Semantic chunker using external embedding endpoint.

    Calls an HTTP endpoint to get embeddings, computes similarity locally.
    Falls back to RegexChunker on error.
    """

    def __init__(
        self,
        config: ChunkerConfig,
        fallback: BaseChunker | None = None,
        tokenizer: BaseTokenizer | None = None,
    ):
        """Initialize HTTP semantic chunker.

        Args:
            config: Chunker configuration with endpoint.
            fallback: Fallback chunker for errors (defaults to RegexChunker).
            tokenizer: Tokenizer for counting tokens (uses default if None).
        """
        self.config = config
        self.fallback = fallback or RegexChunker(tokenizer)
        self._tokenizer = tokenizer
        self._client: Any = None
        self._nltk_ready = False

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Get tokenizer, creating default if needed."""
        if self._tokenizer is None:
            from context7_reranker.tokenizer import get_default_tokenizer

            self._tokenizer = get_default_tokenizer()
        return self._tokenizer

    @property
    def client(self) -> Any:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.config.endpoint:
                raise ValueError("No endpoint configured for HTTP semantic chunker")
            from context7_reranker.http_client import HttpClient

            self._client = HttpClient(
                base_url=self.config.endpoint,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def _ensure_nltk(self) -> None:
        """Ensure NLTK punkt tokenizer is available."""
        if self._nltk_ready:
            return
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            self._nltk_ready = True
        except ImportError as e:
            raise ImportError(
                "nltk required for HttpSemanticChunker. Install with: pip install nltk"
            ) from e

    def _sent_tokenize(self, text: str) -> list[str]:
        """Split text into sentences using NLTK."""
        self._ensure_nltk()
        import nltk

        return nltk.sent_tokenize(text)

    def _compute_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            sim = cosine_similarity(
                np.array(embedding1).reshape(1, -1),
                np.array(embedding2).reshape(1, -1),
            )
            return float(sim[0][0])
        except ImportError as e:
            raise ImportError(
                "scikit-learn required for HttpSemanticChunker. "
                "Install with: pip install scikit-learn"
            ) from e

    def split(
        self,
        content: str,
        source: str = "",
        max_chunk_tokens: int = 1000,
    ) -> list[DocChunk]:
        """Split content using external embeddings (sync wrapper).

        Falls back to regex chunker if in async context or on error.
        """
        if not self.config.endpoint or not content.strip():
            return self.fallback.split(content, source, max_chunk_tokens)

        try:
            asyncio.get_running_loop()
            # Already in async context, use fallback to avoid blocking
            return self.fallback.split(content, source, max_chunk_tokens)
        except RuntimeError:
            # No event loop running, we can create one
            pass

        try:
            return asyncio.run(self.split_async(content, source, max_chunk_tokens))
        except Exception:
            return self.fallback.split(content, source, max_chunk_tokens)

    async def split_async(
        self,
        content: str,
        source: str = "",
        max_chunk_tokens: int = 1000,
    ) -> list[DocChunk]:
        """Split content using external embeddings.

        Args:
            content: The text content to split.
            source: Origin identifier for the chunks.
            max_chunk_tokens: Maximum tokens per chunk.

        Returns:
            List of DocChunk objects.
        """
        if not self.config.endpoint or not content.strip():
            return self.fallback.split(content, source, max_chunk_tokens)

        try:
            sentences = self._sent_tokenize(content)
            if not sentences:
                return []

            if len(sentences) == 1:
                tokens = self.tokenizer.count_tokens(sentences[0])
                return [DocChunk(content=sentences[0], source=source, tokens=tokens)]

            # Get embeddings from HTTP endpoint
            embeddings = await self._get_embeddings(sentences)
            if embeddings is None:
                return self.fallback.split(content, source, max_chunk_tokens)

            # Group sentences by semantic similarity
            return self._group_by_similarity(
                sentences, embeddings, source, max_chunk_tokens
            )
        except Exception:
            return self.fallback.split(content, source, max_chunk_tokens)

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Get embeddings from HTTP endpoint.

        Supports OpenAI embeddings format:
        Request: {"input": [...], "model": "..."}
        Response: {"data": [{"embedding": [...], "index": 0}, ...]}
        """
        try:
            response = await self.client.post_with_retry(
                "",
                {
                    "input": texts,
                    "model": self.config.model,
                },
            )
            if response is None:
                return None

            return self._parse_embeddings_response(response, len(texts))
        except Exception:
            return None

    def _parse_embeddings_response(
        self, data: dict[str, Any], expected_count: int
    ) -> list[list[float]] | None:
        """Parse embeddings from API response.

        Supports:
        - OpenAI: {"data": [{"embedding": [...], "index": 0}, ...]}
        - Simple: {"embeddings": [[...], [...]]}
        """
        # OpenAI format
        if "data" in data:
            results = data["data"]
            if len(results) < expected_count:
                return None
            # Sort by index to ensure correct order
            sorted_results = sorted(results, key=lambda x: x.get("index", 0))
            return [r["embedding"] for r in sorted_results[:expected_count]]

        # Simple format
        if "embeddings" in data:
            embeddings = data["embeddings"]
            if len(embeddings) >= expected_count:
                return embeddings[:expected_count]

        return None

    def _group_by_similarity(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
        source: str,
        max_chunk_tokens: int,
    ) -> list[DocChunk]:
        """Group sentences by semantic similarity."""
        chunks: list[DocChunk] = []
        current_sentences: list[str] = [sentences[0]]
        current_tokens = self.tokenizer.count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            # Check semantic similarity with previous sentence
            similarity = self._compute_similarity(embeddings[i - 1], embeddings[i])

            # Check if we should start a new chunk
            should_split = (
                similarity < self.config.threshold
                or current_tokens + sentence_tokens > max_chunk_tokens
            )

            if should_split and current_sentences:
                # Save current chunk
                chunk_content = " ".join(current_sentences)
                chunks.append(
                    DocChunk(
                        content=chunk_content,
                        source=source,
                        tokens=current_tokens,
                    )
                )
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget last chunk
        if current_sentences:
            chunk_content = " ".join(current_sentences)
            chunks.append(
                DocChunk(
                    content=chunk_content,
                    source=source,
                    tokens=current_tokens,
                )
            )

        return chunks

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
