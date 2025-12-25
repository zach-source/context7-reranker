"""Command-line interface for context7-reranker."""

import argparse
import json
import sys

from context7_reranker.chunker import split_into_chunks
from context7_reranker.factory import configure_from_env
from context7_reranker.formatter import format_output
from context7_reranker.reranker import rerank_chunks


def build_resolve_request(library_name: str) -> dict:
    """Build MCP tool call request for resolve-library-id.

    Args:
        library_name: The library name to resolve.

    Returns:
        Dict with tool call information (does not execute the call).
    """
    request = {"libraryName": library_name}
    return {
        "tool": "mcp__context7__resolve-library-id",
        "input": request,
        "instruction": (
            f"Call resolve-library-id with libraryName='{library_name}' "
            "to get the Context7-compatible library ID"
        ),
    }


def build_docs_request(
    library_id: str, topic: str | None = None, tokens: int = 10000
) -> dict:
    """Build MCP tool call request for get-library-docs.

    Args:
        library_id: The Context7-compatible library ID.
        topic: Optional topic to focus on.
        tokens: Maximum tokens to retrieve.

    Returns:
        Dict with tool call information (does not execute the call).
    """
    request = {"context7CompatibleLibraryID": library_id, "tokens": tokens}
    if topic:
        request["topic"] = topic

    instruction = f"Call get-library-docs for '{library_id}'"
    if topic:
        instruction += f" with topic='{topic}'"

    return {
        "tool": "mcp__context7__get-library-docs",
        "input": request,
        "instruction": instruction,
    }


def main():
    """Main entry point for the CLI."""
    # Configure backends from environment variables
    configure_from_env()

    parser = argparse.ArgumentParser(
        description="Context7 wrapper with tokenization and reranking"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve library name to ID")
    resolve_parser.add_argument("library_name", help="Library name to resolve")

    # Docs command
    docs_parser = subparsers.add_parser("docs", help="Get and rerank library docs")
    docs_parser.add_argument("library_id", help="Context7-compatible library ID")
    docs_parser.add_argument("--topic", "-t", help="Topic to focus on")
    docs_parser.add_argument(
        "--tokens", "-n", type=int, default=10000, help="Max tokens to retrieve"
    )
    docs_parser.add_argument(
        "--top", "-k", type=int, default=5, help="Top K results to return"
    )
    docs_parser.add_argument(
        "--query", "-q", help="Query for reranking (defaults to topic)"
    )

    # Process command (for processing raw content)
    process_parser = subparsers.add_parser(
        "process", help="Process and rerank raw content"
    )
    process_parser.add_argument(
        "--query", "-q", required=True, help="Query for reranking"
    )
    process_parser.add_argument(
        "--top", "-k", type=int, default=5, help="Top K results"
    )
    process_parser.add_argument("--input", "-i", help="Input file (or stdin)")
    process_parser.add_argument(
        "--max-chunk-tokens",
        "-c",
        type=int,
        default=1000,
        help="Max tokens per chunk (default 1000)",
    )

    args = parser.parse_args()

    if args.command == "resolve":
        result = build_resolve_request(args.library_name)
        print(json.dumps(result, indent=2))

    elif args.command == "docs":
        result = build_docs_request(args.library_id, args.topic, args.tokens)
        print(json.dumps(result, indent=2))

    elif args.command == "process":
        # Read content from file or stdin
        if args.input:
            with open(args.input) as f:
                content = f.read()
        else:
            content = sys.stdin.read()

        # Split into chunks
        chunks = split_into_chunks(
            content, source="context7", max_chunk_tokens=args.max_chunk_tokens
        )

        # Rerank by query
        query = args.query
        ranked = rerank_chunks(chunks, query, top_k=args.top)

        # Output
        print(format_output(ranked, query))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
