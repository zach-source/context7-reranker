"""Tests for chunker module."""

from context7_reranker.chunker import DocChunk, split_into_chunks


class TestDocChunk:
    """Tests for DocChunk dataclass."""

    def test_creation(self):
        chunk = DocChunk(content="Test", source="test", tokens=10)
        assert chunk.content == "Test"
        assert chunk.source == "test"
        assert chunk.tokens == 10
        assert chunk.score == 0.0

    def test_score_default(self):
        chunk = DocChunk(content="Test", source="", tokens=5)
        assert chunk.score == 0.0

    def test_score_custom(self):
        chunk = DocChunk(content="Test", source="", tokens=5, score=0.75)
        assert chunk.score == 0.75


class TestSplitIntoChunks:
    """Tests for split_into_chunks function."""

    def test_empty_content(self):
        result = split_into_chunks("")
        assert result == []

    def test_single_section(self):
        content = "This is a single section of content."
        result = split_into_chunks(content)
        assert len(result) == 1
        assert "single section" in result[0].content

    def test_splits_on_headers(self):
        content = """# Header 1
Content for section 1.

# Header 2
Content for section 2."""
        result = split_into_chunks(content, max_chunk_tokens=50)
        assert len(result) >= 1

    def test_splits_on_double_newlines(self):
        content = """First paragraph with some content here for testing purposes.

Second paragraph with different content that should be separate.

Third paragraph here with more words to ensure splitting."""
        result = split_into_chunks(content, max_chunk_tokens=20)
        assert len(result) >= 2

    def test_respects_max_tokens(self):
        # Create content with sentence boundaries for proper splitting
        content = "This is a sentence. " * 100
        result = split_into_chunks(content, max_chunk_tokens=50)
        # Should create multiple chunks
        assert len(result) >= 2
        # Each chunk should respect token limit (with tolerance for boundaries)
        for chunk in result:
            assert chunk.tokens <= 60  # Allow some overflow for sentence boundary

    def test_sets_source(self):
        content = "Test content"
        result = split_into_chunks(content, source="context7")
        assert result[0].source == "context7"

    def test_counts_tokens(self):
        content = "This is some test content for tokenization."
        result = split_into_chunks(content)
        assert result[0].tokens > 0

    def test_handles_markdown_headers_levels(self):
        content = """# H1
Content 1

## H2
Content 2

### H3
Content 3"""
        result = split_into_chunks(content, max_chunk_tokens=30)
        assert len(result) >= 1

    def test_oversized_single_sentence(self):
        # Single sentence that exceeds max_chunk_tokens
        long_sentence = "word " * 200  # ~200 tokens
        result = split_into_chunks(long_sentence, max_chunk_tokens=50)
        # Should still produce chunks even if sentence is too long
        assert len(result) >= 1
        # Total content should be preserved
        total_content = " ".join(c.content for c in result)
        assert "word" in total_content

    def test_unicode_content(self):
        content = """# Unicode Test
日本語のテキスト

## Émojis and Accents
Café ☕ naïve résumé 你好世界

### Mixed Content
Regular text with ñ, ü, and 中文字符"""
        result = split_into_chunks(content, max_chunk_tokens=100)
        assert len(result) >= 1
        # Verify unicode is preserved
        combined = "\n".join(c.content for c in result)
        assert "日本語" in combined
        assert "Café" in combined
        assert "中文" in combined

    def test_code_blocks_preserved(self):
        content = """# Code Example

```python
def hello():
    print("Hello, World!")
```

More text after code."""
        result = split_into_chunks(content, max_chunk_tokens=100)
        combined = "\n".join(c.content for c in result)
        assert "```python" in combined
        assert 'print("Hello, World!")' in combined
