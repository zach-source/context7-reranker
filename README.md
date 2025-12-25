# context7-reranker

TF-IDF reranker for Context7 library documentation with pluggable backends. Improves relevance of documentation chunks returned by the Context7 MCP server.

## Features

- **Hierarchical chunking**: Splits documentation by headers → paragraphs → sentences
- **Semantic chunking**: Optional similarity-based splitting with sentence-transformers
- **TF-IDF reranking**: Scores chunks by relevance to user query
- **HTTP backends**: Support for external tokenizers and rerankers (llama.cpp, AI Gateway)
- **Token counting**: Uses tiktoken (if available) or word-based approximation
- **CLI interface**: Process documentation from files or stdin
- **Async support**: First-class async/await with sync wrappers

## Installation

```bash
# Basic installation
pip install context7-reranker

# With HTTP backends
pip install context7-reranker[http]

# With semantic chunking
pip install context7-reranker[semantic]

# Everything included
pip install context7-reranker[all]

# From source with dev dependencies
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
```

### Pluggable Backends

Configure external services via environment variables:

```bash
# Use external reranker (Cohere format - llama.cpp, vLLM compatible)
export RERANKER_ENDPOINT="http://localhost:8080/v1/rerank"
export RERANKER_FORMAT="cohere"

# Use external tokenizer
export TOKENIZER_ENDPOINT="http://localhost:8080/v1/embeddings"

# Use semantic chunking
export CHUNKER_MODE="semantic"
export CHUNKER_THRESHOLD="0.5"
```

```python
from context7_reranker import configure_from_env

# Auto-configure all backends from environment
configure_from_env()

# Or manually create backends
from context7_reranker import create_reranker, RerankerConfig

config = RerankerConfig(
    endpoint="http://localhost:8080/v1/rerank",
    format="cohere",
    model="bge-reranker-base",
)
reranker = create_reranker(config)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOKENIZER_ENDPOINT` | HTTP tokenizer endpoint | None (local) |
| `TOKENIZER_API_KEY` | API key for tokenizer | None |
| `RERANKER_ENDPOINT` | HTTP reranker endpoint | None (TF-IDF) |
| `RERANKER_FORMAT` | API format (`cohere`/`openai`) | `cohere` |
| `RERANKER_API_KEY` | API key for reranker | None |
| `CHUNKER_MODE` | `regex`, `semantic`, or `http` | `regex` |
| `CHUNKER_THRESHOLD` | Semantic similarity threshold | `0.5` |

## How It Works

### Chunking Strategy

1. **Primary split**: Markdown headers (`#`, `##`, `###`) and double newlines
2. **Accumulation**: Combines sections until hitting `max_chunk_tokens`
3. **Fallback**: Splits oversized sections by sentence boundaries

### Semantic Chunking (optional)

1. Split text into sentences using NLTK
2. Compute embeddings with sentence-transformers
3. Group sentences with similarity above threshold
4. Respect max token limits

### TF-IDF Scoring

1. Extract terms from query and each chunk (filtering stopwords)
2. Compute IDF with Laplace smoothing: `log((N + 1) / (df + 1)) + 1`
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
