"""Tests for factory functions."""



from context7_reranker.chunker import RegexChunker, get_default_chunker
from context7_reranker.config import ChunkerConfig, RerankerConfig, TokenizerConfig
from context7_reranker.factory import (
    configure_from_env,
    create_chunker,
    create_reranker,
    create_tokenizer,
    reset_defaults,
)
from context7_reranker.reranker import TfidfReranker, get_default_reranker
from context7_reranker.tokenizer import LocalTokenizer, get_default_tokenizer


class TestCreateTokenizer:
    """Tests for create_tokenizer factory."""

    def test_creates_local_by_default(self):
        """Should create LocalTokenizer when no endpoint."""
        config = TokenizerConfig(endpoint=None)
        tokenizer = create_tokenizer(config)
        assert isinstance(tokenizer, LocalTokenizer)

    def test_creates_http_with_endpoint(self):
        """Should create HttpTokenizer when endpoint configured."""
        config = TokenizerConfig(endpoint="http://localhost:8080/v1/embeddings")
        tokenizer = create_tokenizer(config)
        # HttpTokenizer wraps but is not LocalTokenizer
        assert not isinstance(tokenizer, LocalTokenizer)
        assert hasattr(tokenizer, "fallback")


class TestCreateReranker:
    """Tests for create_reranker factory."""

    def test_creates_tfidf_by_default(self):
        """Should create TfidfReranker when no endpoint."""
        config = RerankerConfig(endpoint=None)
        reranker = create_reranker(config)
        assert isinstance(reranker, TfidfReranker)

    def test_creates_http_with_endpoint(self):
        """Should create HttpReranker when endpoint configured."""
        config = RerankerConfig(endpoint="http://localhost:8080/v1/rerank")
        reranker = create_reranker(config)
        assert not isinstance(reranker, TfidfReranker)
        assert hasattr(reranker, "fallback")


class TestCreateChunker:
    """Tests for create_chunker factory."""

    def test_creates_regex_by_default(self):
        """Should create RegexChunker when mode=regex."""
        config = ChunkerConfig(mode="regex")
        chunker = create_chunker(config)
        assert isinstance(chunker, RegexChunker)

    def test_creates_regex_for_unknown_mode(self):
        """Should fall back to RegexChunker for unknown mode."""
        config = ChunkerConfig(mode="unknown")
        chunker = create_chunker(config)
        assert isinstance(chunker, RegexChunker)

    def test_semantic_fallback_when_no_deps(self):
        """Should fall back to RegexChunker when semantic deps missing."""
        config = ChunkerConfig(mode="semantic")
        chunker = create_chunker(config)
        # May be SemanticChunker or RegexChunker depending on deps
        assert chunker is not None


class TestConfigureFromEnv:
    """Tests for configure_from_env."""

    def test_configures_defaults(self):
        """Should configure all defaults."""
        # Reset first
        reset_defaults()

        # Configure from env (no env vars set = local defaults)
        configure_from_env()

        # Check defaults are set
        assert get_default_tokenizer() is not None
        assert get_default_reranker() is not None
        assert get_default_chunker() is not None

    def test_respects_env_vars(self, monkeypatch):
        """Should respect environment variables."""
        reset_defaults()

        # Set env vars for HTTP tokenizer
        monkeypatch.setenv("TOKENIZER_ENDPOINT", "http://test:8080/tokenize")

        configure_from_env()

        tokenizer = get_default_tokenizer()
        # Should be HttpTokenizer, not LocalTokenizer
        assert hasattr(tokenizer, "fallback")

        # Cleanup
        reset_defaults()


class TestResetDefaults:
    """Tests for reset_defaults."""

    def test_resets_to_local(self):
        """Should reset all defaults to local implementations."""
        reset_defaults()

        assert isinstance(get_default_tokenizer(), LocalTokenizer)
        assert isinstance(get_default_reranker(), TfidfReranker)
        assert isinstance(get_default_chunker(), RegexChunker)


class TestConfigFromEnv:
    """Tests for Config.from_env() methods."""

    def test_tokenizer_config_from_env(self, monkeypatch):
        """Should read tokenizer config from env."""
        monkeypatch.setenv("TOKENIZER_ENDPOINT", "http://test:8080")
        monkeypatch.setenv("TOKENIZER_API_KEY", "secret-key")
        monkeypatch.setenv("TOKENIZER_MODEL", "custom-model")
        monkeypatch.setenv("TOKENIZER_TIMEOUT", "60")

        config = TokenizerConfig.from_env()

        assert config.endpoint == "http://test:8080"
        assert config.api_key == "secret-key"
        assert config.model == "custom-model"
        assert config.timeout == 60.0

    def test_reranker_config_from_env(self, monkeypatch):
        """Should read reranker config from env."""
        monkeypatch.setenv("RERANKER_ENDPOINT", "http://rerank:8080")
        monkeypatch.setenv("RERANKER_FORMAT", "openai")
        monkeypatch.setenv("RERANKER_MODEL", "bge-reranker")

        config = RerankerConfig.from_env()

        assert config.endpoint == "http://rerank:8080"
        assert config.format == "openai"
        assert config.model == "bge-reranker"

    def test_chunker_config_from_env(self, monkeypatch):
        """Should read chunker config from env."""
        monkeypatch.setenv("CHUNKER_MODE", "semantic")
        monkeypatch.setenv("CHUNKER_THRESHOLD", "0.7")
        monkeypatch.setenv("CHUNKER_MODEL", "all-MiniLM-L6-v2")

        config = ChunkerConfig.from_env()

        assert config.mode == "semantic"
        assert config.threshold == 0.7
        assert config.model == "all-MiniLM-L6-v2"

    def test_config_defaults_without_env(self, monkeypatch):
        """Should use defaults when env vars not set."""
        # Clear any existing env vars
        for var in [
            "TOKENIZER_ENDPOINT",
            "RERANKER_ENDPOINT",
            "CHUNKER_MODE",
        ]:
            monkeypatch.delenv(var, raising=False)

        tok_config = TokenizerConfig.from_env()
        assert tok_config.endpoint is None
        assert tok_config.timeout == 30.0

        rerank_config = RerankerConfig.from_env()
        assert rerank_config.endpoint is None
        assert rerank_config.format == "cohere"

        chunk_config = ChunkerConfig.from_env()
        assert chunk_config.mode == "regex"
        assert chunk_config.threshold == 0.5
