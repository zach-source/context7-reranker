# CLAUDE.md - context7-reranker

TF-IDF reranker for Context7 library documentation with pluggable backends.

## Quick Reference

| Task | Command |
|------|---------|
| Run tests | `uv run pytest` |
| Format | `uv run black src tests` |
| Lint | `uv run ruff check src tests` |
| Install dev | `uv pip install -e ".[dev]"` |
| Install all | `uv pip install -e ".[all]"` |

## Version Control (jj)

This project uses [Jujutsu (jj)](https://github.com/martinvonz/jj) for version control.

| Task | Command |
|------|---------|
| Status | `jj status` or `jj st` |
| Log | `jj log` |
| Describe change | `jj describe -m "message"` |
| New change | `jj new` |
| Squash into parent | `jj squash` |
| Show diff | `jj diff` |
| Push to remote | `jj git push` |
| Fetch from remote | `jj git fetch` |

Common workflows:
```bash
# Make changes, then describe and create new change
jj describe -m "Add feature X"
jj new

# Squash current into parent (like amending)
jj squash

# Push current bookmark to GitHub
jj git push
```

## Project Structure

```
src/context7_reranker/
├── __init__.py          # Package exports
├── config.py            # Configuration dataclasses
├── protocols.py         # Abstract base classes
├── tokenizer.py         # Token counting (LocalTokenizer)
├── http_tokenizer.py    # HTTP tokenizer backend
├── chunker.py           # Document chunking (RegexChunker)
├── semantic_chunker.py  # Semantic chunking with embeddings
├── reranker.py          # TF-IDF scoring (TfidfReranker)
├── http_reranker.py     # HTTP reranker backend
├── http_client.py       # Shared async HTTP client
├── factory.py           # Factory functions
├── formatter.py         # Output formatting
└── cli.py               # Command-line interface
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TOKENIZER_ENDPOINT` | HTTP tokenizer endpoint | `http://localhost:8080/v1/embeddings` |
| `TOKENIZER_API_KEY` | API key for tokenizer | `sk-...` |
| `RERANKER_ENDPOINT` | HTTP reranker endpoint | `http://localhost:8080/v1/rerank` |
| `RERANKER_API_KEY` | API key for reranker | `sk-...` |
| `RERANKER_FORMAT` | API format | `cohere` or `openai` |
| `CHUNKER_MODE` | Chunking mode | `regex`, `semantic`, `http` |
| `CHUNKER_THRESHOLD` | Semantic similarity threshold | `0.5` |

## Optional Dependencies

```bash
pip install context7-reranker[http]      # HTTP backends (httpx)
pip install context7-reranker[semantic]  # Semantic chunking (sentence-transformers)
pip install context7-reranker[all]       # Everything
```

## Key Modules

### Backward-Compatible Functions
- `split_into_chunks(content, source, max_chunk_tokens)` - Chunk documents
- `rerank_chunks(chunks, query, top_k)` - Rerank by TF-IDF
- `count_tokens(text)` - Count tokens

### Class-Based API
- `LocalTokenizer` / `HttpTokenizer` - Token counting backends
- `TfidfReranker` / `HttpReranker` - Reranking backends
- `RegexChunker` / `SemanticChunker` - Chunking backends

### Factory Functions
- `configure_from_env()` - Configure all backends from env vars
- `create_tokenizer(config)` - Create tokenizer from config
- `create_reranker(config)` - Create reranker from config
- `create_chunker(config)` - Create chunker from config

## Testing

```bash
uv run pytest                     # Run all tests
uv run pytest -v                  # Verbose
uv run pytest tests/test_reranker.py  # Specific file
```

## Integration with Context7 Skill

This reranker is used by the context7 Claude skill to improve documentation relevance:

1. Context7 MCP returns raw documentation
2. Reranker splits into chunks
3. TF-IDF (or HTTP backend) scores chunks against user query
4. Top 5 most relevant chunks returned to Claude
