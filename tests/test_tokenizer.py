"""Tests for tokenizer module."""

from context7_reranker.tokenizer import count_tokens, tokenize


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_short_text(self):
        # ~4 chars per token
        assert count_tokens("hello") == 1

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        # 9 words, no special punctuation = 9 tokens
        assert count_tokens(text) == 9

    def test_code_snippet(self):
        code = "def hello_world():\n    print('Hello, World!')"
        tokens = count_tokens(code)
        assert tokens > 0


class TestTokenize:
    """Tests for tokenize function."""

    def test_returns_positive_for_text(self):
        assert tokenize("Hello world") > 0

    def test_empty_string(self):
        assert tokenize("") == 0

    def test_consistent_results(self):
        text = "Test string for consistency"
        assert tokenize(text) == tokenize(text)
