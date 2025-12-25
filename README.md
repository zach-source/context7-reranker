# context7-reranker

TF-IDF reranker for Context7 library documentation. Improves relevance of documentation chunks returned by the Context7 MCP server.

## Features

- **Hierarchical chunking**: Splits documentation by headers → paragraphs → sentences
- **TF-IDF reranking**: Scores chunks by relevance to user query
- **Token counting**: Uses tiktoken (if available) or char-based approximation
- **CLI interface**: Process documentation from files or stdin

## Installation

```bash
# From source
pip install -e .

# With tiktoken for accurate token counting
pip install -e ".[tiktoken]"

# With dev dependencies
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Process documentation and rerank
context7-reranker process --query "authentication middleware" --input docs.md

# Pipe from stdin
cat docs.md | context7-reranker process --query "authentication middleware"

# Control chunk size and result count
context7-reranker process \
  --query "authentication middleware" \
  --input docs.md \
  --max-chunk-tokens 500 \
  --top 3

# Generate MCP tool call for resolving library
context7-reranker resolve express

# Generate MCP tool call for fetching docs
context7-reranker docs /expressjs/express --topic "middleware"
```

### Python API

```python
from context7_reranker import split_into_chunks, rerank_chunks

# Split documentation into chunks
content = open("docs.md").read()
chunks = split_into_chunks(content, source="context7", max_chunk_tokens=1000)

# Rerank by query
query = "How do I set up authentication middleware?"
ranked = rerank_chunks(chunks, query, top_k=5)

for chunk in ranked:
    print(f"Score: {chunk.score:.3f}")
    print(chunk.content[:100])
    print()
```

## How It Works

### Chunking Strategy

1. **Primary split**: Markdown headers (`#`, `##`, `###`) and double newlines
2. **Accumulation**: Combines sections until hitting `max_chunk_tokens`
3. **Fallback**: Splits oversized sections by sentence boundaries

### TF-IDF Scoring

1. Extract terms from query and each chunk (filtering stopwords)
2. Compute IDF across all chunks: `log(N / df) + 1`
3. Score each chunk by sum of `tf * idf` for matching query terms
4. Return top-k chunks sorted by score

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests

# Lint
ruff check src tests
```

## License

MIT
