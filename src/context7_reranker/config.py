"""Configuration for pluggable backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer backends."""

    endpoint: str | None = None
    model: str = "default"
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> TokenizerConfig:
        """Create config from environment variables."""
        return cls(
            endpoint=os.environ.get("TOKENIZER_ENDPOINT"),
            model=os.environ.get("TOKENIZER_MODEL", "default"),
            api_key=os.environ.get("TOKENIZER_API_KEY"),
            timeout=float(os.environ.get("TOKENIZER_TIMEOUT", "30")),
            max_retries=int(os.environ.get("TOKENIZER_MAX_RETRIES", "3")),
        )


@dataclass
class RerankerConfig:
    """Configuration for reranker backends."""

    endpoint: str | None = None
    model: str = "default"
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    format: str = "cohere"  # cohere | openai | custom

    @classmethod
    def from_env(cls) -> RerankerConfig:
        """Create config from environment variables."""
        return cls(
            endpoint=os.environ.get("RERANKER_ENDPOINT"),
            model=os.environ.get("RERANKER_MODEL", "default"),
            api_key=os.environ.get("RERANKER_API_KEY"),
            timeout=float(os.environ.get("RERANKER_TIMEOUT", "60")),
            max_retries=int(os.environ.get("RERANKER_MAX_RETRIES", "3")),
            format=os.environ.get("RERANKER_FORMAT", "cohere"),
        )


@dataclass
class ChunkerConfig:
    """Configuration for chunker backends."""

    mode: str = "regex"  # regex | semantic | http
    endpoint: str | None = None
    model: str = "all-mpnet-base-v1"
    api_key: str | None = None
    timeout: float = 60.0
    threshold: float = 0.5
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> ChunkerConfig:
        """Create config from environment variables."""
        return cls(
            mode=os.environ.get("CHUNKER_MODE", "regex"),
            endpoint=os.environ.get("CHUNKER_ENDPOINT"),
            model=os.environ.get("CHUNKER_MODEL", "all-mpnet-base-v1"),
            api_key=os.environ.get("CHUNKER_API_KEY"),
            timeout=float(os.environ.get("CHUNKER_TIMEOUT", "60")),
            threshold=float(os.environ.get("CHUNKER_THRESHOLD", "0.5")),
            max_retries=int(os.environ.get("CHUNKER_MAX_RETRIES", "3")),
        )


@dataclass
class Config:
    """Combined configuration for all backends."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)

    @classmethod
    def from_env(cls) -> Config:
        """Create combined config from environment variables."""
        return cls(
            tokenizer=TokenizerConfig.from_env(),
            reranker=RerankerConfig.from_env(),
            chunker=ChunkerConfig.from_env(),
        )
