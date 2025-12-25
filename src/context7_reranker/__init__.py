"""Context7 Reranker - TF-IDF reranking for library documentation."""

# Core types
# Class-based implementations
from context7_reranker.chunker import (
    DocChunk,
    RegexChunker,
    get_default_chunker,
    set_default_chunker,
    split_into_chunks,
)

# Configuration
from context7_reranker.config import (
    ChunkerConfig,
    LLMConfig,
    RerankerConfig,
    TokenizerConfig,
)

# Factory functions
from context7_reranker.factory import (
    configure_from_env,
    create_chunker,
    create_query_parser,
    create_reranker,
    create_tokenizer,
    reset_defaults,
)

# Protocols
from context7_reranker.protocols import BaseChunker, BaseReranker, BaseTokenizer

# Query parser
from context7_reranker.query_parser import (
    BaseQueryParser,
    LLMQueryParser,
    ParsedQuery,
    SimpleQueryParser,
    get_default_parser,
    parse_query,
    parse_query_async,
    set_default_parser,
)

# Backward-compatible functions
from context7_reranker.reranker import (
    TfidfReranker,
    get_default_reranker,
    rerank_chunks,
    set_default_reranker,
)
from context7_reranker.tokenizer import (
    LocalTokenizer,
    count_tokens,
    get_default_tokenizer,
    set_default_tokenizer,
    tokenize,
)

__version__ = "0.3.0"
__all__ = [
    # Core types
    "DocChunk",
    "ParsedQuery",
    # Backward-compatible functions
    "split_into_chunks",
    "rerank_chunks",
    "count_tokens",
    "tokenize",
    "parse_query",
    "parse_query_async",
    # Class-based implementations
    "RegexChunker",
    "TfidfReranker",
    "LocalTokenizer",
    "SimpleQueryParser",
    "LLMQueryParser",
    # Default getters/setters
    "get_default_chunker",
    "set_default_chunker",
    "get_default_reranker",
    "set_default_reranker",
    "get_default_tokenizer",
    "set_default_tokenizer",
    "get_default_parser",
    "set_default_parser",
    # Protocols
    "BaseChunker",
    "BaseReranker",
    "BaseTokenizer",
    "BaseQueryParser",
    # Configuration
    "ChunkerConfig",
    "LLMConfig",
    "RerankerConfig",
    "TokenizerConfig",
    # Factory functions
    "configure_from_env",
    "create_chunker",
    "create_query_parser",
    "create_reranker",
    "create_tokenizer",
    "reset_defaults",
]
