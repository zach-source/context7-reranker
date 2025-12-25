# context7-reranker

TF-IDF reranker for Context7 library documentation with pluggable backends. Improves relevance of documentation chunks returned by the Context7 MCP server.

## Features

- **LLM Query Parsing**: Extract library names and topics from natural language using OpenAI-compatible APIs
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
# Parse a query to extract library and topic (uses LLM if configured)
context7-reranker parse "How do I use React hooks?"

# Output in different formats
context7-reranker parse "FastAPI authentication" --format json
context7-reranker parse "pandas dataframe filtering" --format context7
context7-reranker parse "Next.js routing" --format text

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
from context7_reranker import parse_query, split_into_chunks, rerank_chunks

# Parse a natural language query to extract library and topic
parsed = parse_query("How do I use React hooks for state management?")
print(f"Library: {parsed.library_name}")  # "react"
print(f"Topic: {parsed.topic}")           # "hooks for state management"
print(f"Confidence: {parsed.confidence}") # 0.9

# Get Context7 MCP parameters
params = parsed.to_context7_params()
# {"libraryName": "react", "topic": "hooks for state management"}

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

### Query Parser with LLM

```python
from context7_reranker import LLMQueryParser, LLMConfig

# Configure LLM for better query parsing
config = LLMConfig(
    endpoint="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini",
    temperature=0.0,
)

parser = LLMQueryParser(config)
result = parser.parse("tensorflow vs pytorch for image classification")

print(result.library_name)           # "tensorflow"
print(result.topic)                  # "image classification"
print(result.alternative_libraries)  # ["pytorch"]
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
| `LLM_ENDPOINT` | OpenAI-compatible chat API endpoint | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key for LLM (also reads `OPENAI_API_KEY`) | None |
| `LLM_MODEL` | Model name for query parsing | `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Temperature for LLM responses | `0` |
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

## Claude Code Integration

### Install the Skill

Copy the skill to your Claude Code skills directory:

```bash
# User-level installation
mkdir -p ~/.claude/skills
cp skills/context7-reranker.md ~/.claude/skills/

# Or project-level installation
mkdir -p .claude/skills
cp skills/context7-reranker.md .claude/skills/
```

The skill provides Claude with guidance on parsing library queries and reranking documentation.

### Configure Hooks (Optional)

Hooks can automatically enhance Context7 MCP calls. Add to your Claude Code settings:

**~/.claude/settings.json** or **.claude/settings.json**:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__context7__resolve-library-id",
        "hooks": [
          {
            "type": "command",
            "command": "context7-reranker parse \"$TOOL_INPUT_libraryName\" --format json 2>/dev/null || true"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "mcp__context7__get-library-docs",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$TOOL_OUTPUT\" | context7-reranker process --query \"$TOOL_INPUT_topic\" --top 5 2>/dev/null || cat"
          }
        ]
      }
    ]
  }
}
```

### NixOS / nix-darwin Installation

Add the flake to your configuration:

```nix
# flake.nix
{
  inputs.context7-reranker.url = "github:zach-source/context7-reranker";
}

# configuration.nix
{ inputs, ... }: {
  imports = [ inputs.context7-reranker.nixosModules.default ];

  nixpkgs.overlays = [ inputs.context7-reranker.overlays.default ];

  services.context7-reranker = {
    enable = true;
    server.enable = true;  # Run as HTTP service
    server.port = 8000;

    llm = {
      endpoint = "https://api.openai.com/v1";
      apiKeyFile = "/run/secrets/openai-api-key";
      model = "gpt-4o-mini";
    };
  };
}
```

Or just add the package to your environment:

```nix
{ inputs, ... }: {
  environment.systemPackages = [
    inputs.context7-reranker.packages.${system}.default
  ];
}
```

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

# Nix development shell
nix develop
```

## License

MIT
