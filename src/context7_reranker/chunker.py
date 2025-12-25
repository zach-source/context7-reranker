"""Document chunking utilities with pluggable backends."""

from __future__ import annotations

import re
from dataclasses import dataclass

from context7_reranker.protocols import BaseChunker, BaseTokenizer


@dataclass
class DocChunk:
    """A chunk of documentation with metadata.

    Attributes:
        content: The text content of the chunk.
        source: Origin identifier (e.g., "context7").
        tokens: Approximate token count.
        score: Relevance score after reranking (default 0.0).
    """

    content: str
    source: str
    tokens: int
    score: float = 0.0


class RegexChunker(BaseChunker):
    """Regex-based chunker using hierarchical splitting."""

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        """Initialize regex chunker.

        Args:
            tokenizer: Tokenizer for counting tokens (uses default if None).
        """
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> BaseTokenizer:
        """Get tokenizer, creating default if needed."""
        if self._tokenizer is None:
            from context7_reranker.tokenizer import get_default_tokenizer

            self._tokenizer = get_default_tokenizer()
        return self._tokenizer

    def split(
        self,
        content: str,
        source: str = "",
        max_chunk_tokens: int = 1000,
    ) -> list[DocChunk]:
        """Split content into chunks based on headers or paragraphs.

        Uses a 3-tier hierarchical splitting approach:
        1. Markdown headers (# ## ###)
        2. Paragraphs (double newlines)
        3. Sentences (fallback for oversized sections)

        Args:
            content: The markdown content to split.
            source: Origin identifier for the chunks.
            max_chunk_tokens: Maximum tokens per chunk.

        Returns:
            List of DocChunk objects.
        """
        chunks: list[DocChunk] = []

        # Split by markdown headers or double newlines
        sections = re.split(r"\n(?=#{1,3}\s)|(?:\n\n)+", content)

        current_chunk = ""
        current_tokens = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_tokens = self.tokenizer.count_tokens(section)

            # If section fits in current chunk
            if current_tokens + section_tokens <= max_chunk_tokens:
                current_chunk += "\n\n" + section if current_chunk else section
                current_tokens += section_tokens
            else:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(
                        DocChunk(
                            content=current_chunk,
                            source=source,
                            tokens=current_tokens,
                        )
                    )

                # Start new chunk
                if section_tokens <= max_chunk_tokens:
                    current_chunk = section
                    current_tokens = section_tokens
                else:
                    # Section too large, split by sentences
                    chunks.extend(
                        self._split_by_sentences(section, source, max_chunk_tokens)
                    )
                    current_chunk = ""
                    current_tokens = 0

        # Don't forget last chunk
        if current_chunk:
            chunks.append(
                DocChunk(
                    content=current_chunk,
                    source=source,
                    tokens=current_tokens,
                )
            )

        return chunks

    def _split_by_sentences(
        self,
        section: str,
        source: str,
        max_chunk_tokens: int,
    ) -> list[DocChunk]:
        """Split a large section by sentence boundaries.

        Args:
            section: The text section to split.
            source: Origin identifier.
            max_chunk_tokens: Maximum tokens per chunk.

        Returns:
            List of DocChunk objects.
        """
        chunks: list[DocChunk] = []
        sentences = re.split(r"(?<=[.!?])\s+", section)
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.tokenizer.count_tokens(sentence)
            if current_tokens + sent_tokens <= max_chunk_tokens:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    chunks.append(
                        DocChunk(
                            content=current_chunk,
                            source=source,
                            tokens=current_tokens,
                        )
                    )
                current_chunk = sentence
                current_tokens = sent_tokens

        if current_chunk:
            chunks.append(
                DocChunk(
                    content=current_chunk,
                    source=source,
                    tokens=current_tokens,
                )
            )

        return chunks


# Default chunker instance
_default_chunker: BaseChunker | None = None


def get_default_chunker() -> BaseChunker:
    """Get the default chunker instance.

    Returns:
        The default chunker (creates RegexChunker if not set).
    """
    global _default_chunker
    if _default_chunker is None:
        _default_chunker = RegexChunker()
    return _default_chunker


def set_default_chunker(chunker: BaseChunker) -> None:
    """Set the default chunker instance.

    Args:
        chunker: The chunker to use as default.
    """
    global _default_chunker
    _default_chunker = chunker


# Backward-compatible module-level functions


def split_into_chunks(
    content: str,
    source: str = "",
    max_chunk_tokens: int = 1000,
) -> list[DocChunk]:
    """Split content into chunks based on headers or paragraphs.

    Uses a 3-tier hierarchical splitting approach:
    1. Markdown headers (# ## ###)
    2. Paragraphs (double newlines)
    3. Sentences (fallback for oversized sections)

    Args:
        content: The markdown content to split.
        source: Origin identifier for the chunks.
        max_chunk_tokens: Maximum tokens per chunk (default 1000).

    Returns:
        List of DocChunk objects.
    """
    return get_default_chunker().split(content, source, max_chunk_tokens)


# Legacy internal function for backward compatibility
def _split_by_sentences(
    section: str,
    source: str,
    max_chunk_tokens: int,
) -> list[DocChunk]:
    """Split a large section by sentence boundaries.

    Deprecated: Use RegexChunker directly.
    """
    chunker = get_default_chunker()
    if isinstance(chunker, RegexChunker):
        return chunker._split_by_sentences(section, source, max_chunk_tokens)
    # Fallback - create temporary chunker
    return RegexChunker()._split_by_sentences(section, source, max_chunk_tokens)
