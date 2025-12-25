"""Factory functions for creating configured backends."""

from __future__ import annotations

from context7_reranker.chunker import RegexChunker, set_default_chunker
from context7_reranker.config import ChunkerConfig, RerankerConfig, TokenizerConfig
from context7_reranker.protocols import BaseChunker, BaseReranker, BaseTokenizer
from context7_reranker.reranker import TfidfReranker, set_default_reranker
from context7_reranker.tokenizer import LocalTokenizer, set_default_tokenizer


def create_tokenizer(config: TokenizerConfig | None = None) -> BaseTokenizer:
    """Create a tokenizer based on configuration.

    Args:
        config: Tokenizer configuration (uses env vars if None).

    Returns:
        Configured tokenizer instance.
    """
    config = config or TokenizerConfig.from_env()

    if config.endpoint:
        from context7_reranker.http_tokenizer import HttpTokenizer

        return HttpTokenizer(config, fallback=LocalTokenizer())

    return LocalTokenizer()


def create_reranker(config: RerankerConfig | None = None) -> BaseReranker:
    """Create a reranker based on configuration.

    Args:
        config: Reranker configuration (uses env vars if None).

    Returns:
        Configured reranker instance.
    """
    config = config or RerankerConfig.from_env()

    if config.endpoint:
        from context7_reranker.http_reranker import HttpReranker

        return HttpReranker(config, fallback=TfidfReranker())

    return TfidfReranker()


def create_chunker(config: ChunkerConfig | None = None) -> BaseChunker:
    """Create a chunker based on configuration.

    Args:
        config: Chunker configuration (uses env vars if None).

    Returns:
        Configured chunker instance.
    """
    config = config or ChunkerConfig.from_env()

    if config.mode == "semantic":
        try:
            from context7_reranker.semantic_chunker import SemanticChunker

            return SemanticChunker(
                model=config.model,
                threshold=config.threshold,
            )
        except ImportError:
            # Fall back to regex if sentence-transformers not available
            return RegexChunker()

    if config.mode == "http":
        if config.endpoint:
            try:
                from context7_reranker.semantic_chunker import HttpSemanticChunker

                return HttpSemanticChunker(config, fallback=RegexChunker())
            except ImportError:
                # Fall back to regex if dependencies not available
                return RegexChunker()

    return RegexChunker()


def configure_from_env() -> None:
    """Configure all default backends from environment variables.

    Reads configuration from environment and sets up defaults for:
    - Tokenizer (TOKENIZER_ENDPOINT, TOKENIZER_API_KEY, etc.)
    - Reranker (RERANKER_ENDPOINT, RERANKER_API_KEY, etc.)
    - Chunker (CHUNKER_MODE, CHUNKER_ENDPOINT, etc.)

    Call this at application startup to enable external services.
    """
    # Configure tokenizer
    tokenizer_config = TokenizerConfig.from_env()
    tokenizer = create_tokenizer(tokenizer_config)
    set_default_tokenizer(tokenizer)

    # Configure reranker
    reranker_config = RerankerConfig.from_env()
    reranker = create_reranker(reranker_config)
    set_default_reranker(reranker)

    # Configure chunker
    chunker_config = ChunkerConfig.from_env()
    chunker = create_chunker(chunker_config)
    set_default_chunker(chunker)


def reset_defaults() -> None:
    """Reset all defaults to local implementations.

    Useful for testing or when switching from external to local backends.
    """
    set_default_tokenizer(LocalTokenizer())
    set_default_reranker(TfidfReranker())
    set_default_chunker(RegexChunker())
