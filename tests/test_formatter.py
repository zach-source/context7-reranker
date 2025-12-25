"""Tests for formatter module."""

from context7_reranker.chunker import DocChunk
from context7_reranker.formatter import format_output


class TestFormatOutput:
    """Tests for format_output function."""

    def test_empty_chunks(self):
        result = format_output([], "test query")
        assert "Top 0 Results" in result
        assert "test query" in result
        assert "Total tokens: 0" in result

    def test_single_chunk(self):
        chunks = [
            DocChunk(content="Test content", source="context7", tokens=10, score=0.5)
        ]
        result = format_output(chunks, "my query")
        assert "Top 1 Results" in result
        assert "my query" in result
        assert "Test content" in result
        assert "score: 0.500" in result
        assert "tokens: 10" in result
        assert "context7" in result

    def test_multiple_chunks(self):
        chunks = [
            DocChunk(content="First chunk", source="src1", tokens=10, score=0.9),
            DocChunk(content="Second chunk", source="src2", tokens=20, score=0.5),
            DocChunk(content="Third chunk", source="src3", tokens=15, score=0.3),
        ]
        result = format_output(chunks, "query")
        assert "Top 3 Results" in result
        assert "Result 1" in result
        assert "Result 2" in result
        assert "Result 3" in result
        assert "First chunk" in result
        assert "Second chunk" in result
        assert "Third chunk" in result

    def test_total_tokens_sum(self):
        chunks = [
            DocChunk(content="A", source="", tokens=100, score=0.1),
            DocChunk(content="B", source="", tokens=200, score=0.2),
        ]
        result = format_output(chunks, "query")
        assert "Total tokens: 300" in result

    def test_empty_source_not_shown(self):
        chunks = [DocChunk(content="Content", source="", tokens=10, score=0.5)]
        result = format_output(chunks, "query")
        # Should not have "Source:" line when source is empty
        assert "Source:" not in result

    def test_score_formatting(self):
        chunks = [DocChunk(content="Content", source="", tokens=10, score=0.12345)]
        result = format_output(chunks, "query")
        # Score should be formatted to 3 decimal places
        assert "score: 0.123" in result

    def test_preserves_content_formatting(self):
        content = """# Header

Some paragraph with **bold** text.

- List item 1
- List item 2"""
        chunks = [DocChunk(content=content, source="", tokens=50, score=1.0)]
        result = format_output(chunks, "query")
        assert "# Header" in result
        assert "**bold**" in result
        assert "- List item 1" in result
