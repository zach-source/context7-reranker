"""Output formatting utilities."""

from context7_reranker.chunker import DocChunk


def format_output(chunks: list[DocChunk], query: str) -> str:
    """Format reranked chunks for output.

    Args:
        chunks: List of reranked document chunks.
        query: The original query.

    Returns:
        Formatted markdown string with results.
    """
    output = []
    output.append(f"# Top {len(chunks)} Results for: {query}\n")

    total_tokens = sum(c.tokens for c in chunks)
    output.append(f"_Total tokens: {total_tokens}_\n")

    for i, chunk in enumerate(chunks, 1):
        output.append(
            f"\n## Result {i} (score: {chunk.score:.3f}, tokens: {chunk.tokens})"
        )
        if chunk.source:
            output.append(f"_Source: {chunk.source}_")
        output.append("")
        output.append(chunk.content)

    return "\n".join(output)
