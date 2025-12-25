"""Token counting utilities with pluggable backends."""

from __future__ import annotations

from typing import Any

from context7_reranker.protocols import BaseTokenizer


class LocalTokenizer(BaseTokenizer):
    """Local tokenizer using tiktoken or word-based approximation."""

    def __init__(self, prefer_tiktoken: bool = True):
        """Initialize local tokenizer.

        Args:
            prefer_tiktoken: If True, try to use tiktoken first.
        """
        self._prefer_tiktoken = prefer_tiktoken
        self._encoder: Any = None
        if prefer_tiktoken:
            self._try_load_tiktoken()

    def _try_load_tiktoken(self) -> None:
        """Try to load tiktoken encoder."""
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Uses tiktoken if available, otherwise word-based approximation.

        Args:
            text: The text to tokenize.

        Returns:
            Token count.
        """
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return self._count_tokens_approximate(text)

    def _count_tokens_approximate(self, text: str) -> int:
        """Count tokens using word-based approximation.

        Uses word count plus adjustment for punctuation and special characters.
        More accurate than pure character division, especially for code.
        """
        words = text.split()
        word_count = len(words)
        # Add ~30% for punctuation, operators, and subword tokens in code
        adjustment = (
            len([c for c in text if c in ".,;:!?()[]{}\"'-=+*/<>@#$%^&|\\"]) // 2
        )
        return word_count + adjustment


# Default tokenizer instance
_default_tokenizer: BaseTokenizer | None = None


def get_default_tokenizer() -> BaseTokenizer:
    """Get the default tokenizer instance.

    Returns:
        The default tokenizer (creates LocalTokenizer if not set).
    """
    global _default_tokenizer
    if _default_tokenizer is None:
        _default_tokenizer = LocalTokenizer()
    return _default_tokenizer


def set_default_tokenizer(tokenizer: BaseTokenizer) -> None:
    """Set the default tokenizer instance.

    Args:
        tokenizer: The tokenizer to use as default.
    """
    global _default_tokenizer
    _default_tokenizer = tokenizer


# Backward-compatible module-level functions


def count_tokens(text: str) -> int:
    """Count tokens using a word-based approximation.

    Uses word count plus adjustment for punctuation and special characters.
    More accurate than pure character division, especially for code.

    Args:
        text: The text to tokenize.

    Returns:
        Token count.
    """
    return get_default_tokenizer().count_tokens(text)


def tokenize(text: str) -> int:
    """Count tokens, preferring tiktoken if available.

    Args:
        text: The text to tokenize.

    Returns:
        Token count (exact if tiktoken available, approximate otherwise).
    """
    return count_tokens(text)


# Legacy function for backward compatibility
def try_tiktoken(text: str) -> int | None:
    """Try to count tokens using tiktoken if available.

    Deprecated: Use LocalTokenizer or count_tokens() instead.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return None
