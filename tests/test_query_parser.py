"""Tests for query parser."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from context7_reranker.config import LLMConfig
from context7_reranker.query_parser import (
    LLMQueryParser,
    ParsedQuery,
    SimpleQueryParser,
    parse_query,
)


class TestParsedQuery:
    """Tests for ParsedQuery dataclass."""

    def test_basic_creation(self):
        """Test creating a ParsedQuery."""
        query = ParsedQuery(
            library_name="react",
            topic="hooks",
            confidence=0.9,
        )
        assert query.library_name == "react"
        assert query.topic == "hooks"
        assert query.confidence == 0.9
        assert query.version is None
        assert query.alternative_libraries == []

    def test_to_context7_params(self):
        """Test conversion to Context7 parameters."""
        query = ParsedQuery(
            library_name="fastapi",
            topic="authentication",
            confidence=0.95,
        )
        params = query.to_context7_params()
        assert params == {"libraryName": "fastapi", "topic": "authentication"}

    def test_to_context7_params_no_topic(self):
        """Test conversion without topic."""
        query = ParsedQuery(library_name="pandas", confidence=0.8)
        params = query.to_context7_params()
        assert params == {"libraryName": "pandas"}


class TestSimpleQueryParser:
    """Tests for SimpleQueryParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = SimpleQueryParser()

    def test_parse_react_hooks(self):
        """Test parsing React hooks query."""
        result = self.parser.parse("How do I use React hooks?")
        assert result.library_name == "react"
        assert result.confidence >= 0.5

    def test_parse_fastapi(self):
        """Test parsing FastAPI query."""
        result = self.parser.parse("FastAPI authentication with JWT")
        assert result.library_name == "fastapi"
        assert "jwt" in result.topic.lower() or "authentication" in result.topic.lower()

    def test_parse_pandas(self):
        """Test parsing pandas query."""
        result = self.parser.parse("pandas dataframe filtering")
        assert result.library_name == "pandas"

    def test_parse_nextjs(self):
        """Test parsing Next.js query."""
        result = self.parser.parse("Next.js app router")
        assert result.library_name == "next.js"

    def test_parse_tensorflow(self):
        """Test parsing TensorFlow query."""
        result = self.parser.parse("tensorflow image classification")
        assert result.library_name == "tensorflow"

    def test_parse_kubernetes(self):
        """Test parsing Kubernetes query."""
        result = self.parser.parse("k8s deployment")
        assert result.library_name == "kubernetes"

    def test_parse_quoted_library(self):
        """Test parsing query with quoted library name."""
        result = self.parser.parse('How to use "axios" for HTTP requests?')
        assert result.library_name == "axios"

    def test_parse_unknown_library(self):
        """Test parsing query with unknown library."""
        result = self.parser.parse("SomeUnknownLib configuration")
        # Should extract the capitalized word
        assert result.library_name.lower() == "someunknownlib"
        assert result.confidence < 0.7

    def test_stores_raw_query(self):
        """Test that raw query is stored."""
        query = "React hooks tutorial"
        result = self.parser.parse(query)
        assert result.raw_query == query


class TestLLMQueryParser:
    """Tests for LLMQueryParser with mocked HTTP client."""

    def test_init_without_config(self):
        """Test initialization without config uses env."""
        parser = LLMQueryParser()
        assert parser._fallback is not None

    def test_init_with_config(self):
        """Test initialization with config."""
        config = LLMConfig(
            endpoint="http://localhost:8080/v1",
            api_key="test-key",
            model="gpt-4o-mini",
        )
        parser = LLMQueryParser(config)
        assert parser.config.endpoint == "http://localhost:8080/v1"
        assert parser.config.model == "gpt-4o-mini"

    def test_build_request(self):
        """Test building request payload."""
        config = LLMConfig(
            endpoint="http://localhost:8080/v1",
            api_key="test-key",
            model="gpt-4o",
            temperature=0.1,
            max_tokens=200,
        )
        parser = LLMQueryParser(config)
        request = parser._build_request("How do I use React hooks?")

        assert request["model"] == "gpt-4o"
        assert request["temperature"] == 0.1
        assert request["max_tokens"] == 200
        assert len(request["messages"]) == 2
        assert request["messages"][0]["role"] == "system"
        assert request["messages"][1]["role"] == "user"
        assert request["messages"][1]["content"] == "How do I use React hooks?"
        assert request["response_format"] == {"type": "json_object"}

    def test_parse_response_valid(self):
        """Test parsing valid LLM response."""
        config = LLMConfig(endpoint="http://localhost", api_key="test")
        parser = LLMQueryParser(config)

        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "library_name": "react",
                                "topic": "hooks",
                                "version": "18",
                                "confidence": 0.95,
                                "alternative_libraries": ["preact"],
                            }
                        )
                    }
                }
            ]
        }

        result = parser._parse_response(response, "test query")
        assert result.library_name == "react"
        assert result.topic == "hooks"
        assert result.version == "18"
        assert result.confidence == 0.95
        assert result.alternative_libraries == ["preact"]

    def test_parse_response_invalid_json(self):
        """Test fallback on invalid JSON response."""
        config = LLMConfig(endpoint="http://localhost", api_key="test")
        parser = LLMQueryParser(config)

        response = {"choices": [{"message": {"content": "not valid json"}}]}

        result = parser._parse_response(response, "React hooks")
        # Should fall back to simple parser
        assert result.library_name == "react"

    def test_parse_response_missing_content(self):
        """Test fallback on missing content."""
        config = LLMConfig(endpoint="http://localhost", api_key="test")
        parser = LLMQueryParser(config)

        response = {"choices": []}

        result = parser._parse_response(response, "fastapi auth")
        # Should fall back to simple parser
        assert result.library_name == "fastapi"

    @pytest.mark.asyncio
    async def test_parse_async_no_endpoint(self):
        """Test async parse falls back when no endpoint configured."""
        config = LLMConfig(endpoint=None, api_key=None)
        parser = LLMQueryParser(config)

        result = await parser.parse_async("React hooks")
        assert result.library_name == "react"

    @pytest.mark.asyncio
    async def test_parse_async_with_mock_client(self):
        """Test async parse with mocked HTTP client."""
        config = LLMConfig(
            endpoint="http://localhost:8080/v1",
            api_key="test-key",
        )
        parser = LLMQueryParser(config)

        # Mock the client.post method
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "library_name": "django",
                                "topic": "authentication",
                                "confidence": 0.9,
                                "alternative_libraries": [],
                            }
                        )
                    }
                }
            ]
        }

        # Create a mock client and set it on the parser's internal attribute
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        parser._client = mock_client

        result = await parser.parse_async("Django authentication setup")

        assert result.library_name == "django"
        assert result.topic == "authentication"
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_parse_async_error_fallback(self):
        """Test async parse falls back on HTTP error."""
        config = LLMConfig(
            endpoint="http://localhost:8080/v1",
            api_key="test-key",
        )
        parser = LLMQueryParser(config)

        # Create a mock client that raises an exception
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection error"))
        parser._client = mock_client

        result = await parser.parse_async("Flask routing")

        # Should fall back to simple parser
        assert result.library_name == "flask"


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_parse_query_uses_default(self):
        """Test parse_query uses default parser."""
        result = parse_query("pandas dataframe operations")
        assert result.library_name == "pandas"
        assert result.raw_query == "pandas dataframe operations"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_query(self):
        """Test handling empty query."""
        parser = SimpleQueryParser()
        result = parser.parse("")
        assert result.library_name == "unknown"
        assert result.confidence < 0.5

    def test_whitespace_query(self):
        """Test handling whitespace-only query."""
        parser = SimpleQueryParser()
        result = parser.parse("   ")
        # Should handle gracefully
        assert result.confidence < 0.5

    def test_very_long_query(self):
        """Test handling very long query."""
        parser = SimpleQueryParser()
        long_query = "React " + "hooks " * 100
        result = parser.parse(long_query)
        assert result.library_name == "react"
        # Topic should be truncated
        if result.topic:
            assert len(result.topic) <= 100

    def test_special_characters(self):
        """Test handling special characters."""
        parser = SimpleQueryParser()
        result = parser.parse("Next.js @latest with TypeScript!!!")
        assert result.library_name == "next.js"

    def test_case_insensitivity(self):
        """Test case insensitive matching."""
        parser = SimpleQueryParser()

        result1 = parser.parse("REACT hooks")
        result2 = parser.parse("react HOOKS")
        result3 = parser.parse("ReAcT Hooks")

        assert result1.library_name == "react"
        assert result2.library_name == "react"
        assert result3.library_name == "react"
