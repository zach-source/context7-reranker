"""LLM-based query parser for identifying libraries and topics."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from context7_reranker.config import LLMConfig
from context7_reranker.http_client import HttpClient


@dataclass
class ParsedQuery:
    """Structured output from query parsing."""

    library_name: str
    """Primary library/package name (e.g., 'react', 'fastapi', 'pandas')."""

    topic: str | None = None
    """Specific topic or focus area (e.g., 'hooks', 'authentication', 'dataframes')."""

    version: str | None = None
    """Version constraint if specified (e.g., '>=2.0', 'v18', 'latest')."""

    confidence: float = 1.0
    """Confidence score from 0.0 to 1.0."""

    alternative_libraries: list[str] = field(default_factory=list)
    """Alternative library names that might match the query."""

    raw_query: str = ""
    """Original query text."""

    def to_context7_params(self) -> dict[str, Any]:
        """Convert to Context7 MCP tool parameters.

        Returns:
            Dict suitable for resolve-library-id or get-library-docs calls.
        """
        params: dict[str, Any] = {"libraryName": self.library_name}
        if self.topic:
            params["topic"] = self.topic
        return params


# System prompt for query parsing
QUERY_PARSER_SYSTEM_PROMPT = """You are a library documentation query parser. Your task is to analyze user queries about programming libraries and extract structured information.

Given a user query, identify:
1. **library_name**: The primary library, package, or framework being asked about. Use the canonical/official name (e.g., "react" not "React.js", "fastapi" not "Fast API").
2. **topic**: The specific topic, feature, or concept within that library (e.g., "hooks", "authentication", "routing"). Leave null if the query is general.
3. **version**: Any version constraints mentioned (e.g., "v18", ">=2.0", "latest"). Leave null if not specified.
4. **confidence**: Your confidence in the parsing from 0.0 to 1.0. Lower confidence if the query is ambiguous.
5. **alternative_libraries**: Other libraries that might match if the primary is incorrect (max 3).

Examples:
- "How do I use React hooks?" -> library_name: "react", topic: "hooks"
- "FastAPI authentication with JWT" -> library_name: "fastapi", topic: "authentication"
- "pandas dataframe filtering" -> library_name: "pandas", topic: "dataframe filtering"
- "Next.js 14 app router" -> library_name: "next.js", topic: "app router", version: "14"
- "tensorflow vs pytorch for image classification" -> library_name: "tensorflow", topic: "image classification", alternatives: ["pytorch"]

Respond ONLY with valid JSON matching the schema. No explanation."""

# JSON schema for structured output
QUERY_PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "library_name": {
            "type": "string",
            "description": "Primary library/package name",
        },
        "topic": {
            "type": ["string", "null"],
            "description": "Specific topic or feature within the library",
        },
        "version": {
            "type": ["string", "null"],
            "description": "Version constraint if specified",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score",
        },
        "alternative_libraries": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
            "description": "Alternative library names",
        },
    },
    "required": ["library_name", "confidence"],
    "additionalProperties": False,
}


class BaseQueryParser(ABC):
    """Abstract base class for query parsers."""

    @abstractmethod
    def parse(self, query: str) -> ParsedQuery:
        """Parse a user query to extract library and topic information.

        Args:
            query: The user's natural language query.

        Returns:
            ParsedQuery with extracted information.
        """
        pass

    async def parse_async(self, query: str) -> ParsedQuery:
        """Async version of parse.

        Default implementation wraps sync method.
        Override for true async behavior.
        """
        return self.parse(query)


class SimpleQueryParser(BaseQueryParser):
    """Simple rule-based query parser as fallback."""

    # Common library name patterns
    LIBRARY_PATTERNS = [
        # JavaScript/TypeScript
        ("react", ["react", "reactjs", "react.js"]),
        ("next.js", ["next", "nextjs", "next.js"]),
        ("vue", ["vue", "vuejs", "vue.js"]),
        ("angular", ["angular", "angularjs"]),
        ("svelte", ["svelte", "sveltekit"]),
        ("express", ["express", "expressjs"]),
        ("fastify", ["fastify"]),
        ("nest", ["nest", "nestjs"]),
        # Python
        ("fastapi", ["fastapi", "fast api", "fast-api"]),
        ("django", ["django"]),
        ("flask", ["flask"]),
        ("pandas", ["pandas"]),
        ("numpy", ["numpy"]),
        ("tensorflow", ["tensorflow", "tf"]),
        ("pytorch", ["pytorch", "torch"]),
        ("scikit-learn", ["sklearn", "scikit-learn", "scikit learn"]),
        # Go
        ("gin", ["gin", "gin-gonic"]),
        ("echo", ["echo", "labstack echo"]),
        ("fiber", ["fiber", "gofiber"]),
        # Rust
        ("actix", ["actix", "actix-web"]),
        ("tokio", ["tokio"]),
        ("axum", ["axum"]),
        # Other
        ("docker", ["docker", "dockerfile"]),
        ("kubernetes", ["kubernetes", "k8s", "kubectl"]),
        ("terraform", ["terraform", "tf"]),
    ]

    def parse(self, query: str) -> ParsedQuery:
        """Parse query using simple pattern matching."""
        query_lower = query.lower()

        # Find matching library
        library_name = ""
        confidence = 0.5

        for canonical_name, patterns in self.LIBRARY_PATTERNS:
            for pattern in patterns:
                if pattern in query_lower:
                    library_name = canonical_name
                    confidence = 0.7
                    break
            if library_name:
                break

        # If no pattern match, extract first capitalized word or quoted term
        if not library_name:
            import re

            # Try quoted terms first
            quoted = re.findall(r'["\']([^"\']+)["\']', query)
            if quoted:
                library_name = quoted[0].lower()
                confidence = 0.6
            else:
                # Try capitalized words
                words = re.findall(r"\b([A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z]+)?)\b", query)
                if words:
                    library_name = words[0].lower()
                    confidence = 0.4
                else:
                    # Fall back to first word
                    library_name = (
                        query.split()[0].lower() if query.split() else "unknown"
                    )
                    confidence = 0.2

        # Extract topic (words after library name)
        topic = None
        if library_name:
            idx = query_lower.find(library_name.split(".")[0])
            if idx >= 0:
                remainder = query[idx + len(library_name) :].strip()
                if remainder:
                    # Clean up common prefixes
                    for prefix in ["for", "with", "in", "about", "-", ":"]:
                        if remainder.lower().startswith(prefix):
                            remainder = remainder[len(prefix) :].strip()
                    if remainder and len(remainder) > 2:
                        topic = remainder[:100]  # Limit length

        return ParsedQuery(
            library_name=library_name,
            topic=topic,
            confidence=confidence,
            raw_query=query,
        )


class LLMQueryParser(BaseQueryParser):
    """Query parser using LLM with structured output."""

    def __init__(self, config: LLMConfig | None = None):
        """Initialize LLM query parser.

        Args:
            config: LLM configuration. Uses environment if not provided.
        """
        self.config = config or LLMConfig.from_env()
        self._client: HttpClient | None = None
        self._fallback = SimpleQueryParser()

    @property
    def client(self) -> HttpClient:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            if not self.config.endpoint:
                raise ValueError("LLM endpoint not configured")
            self._client = HttpClient(
                base_url=self.config.endpoint,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._client

    def _build_request(self, query: str) -> dict[str, Any]:
        """Build OpenAI chat completion request."""
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": QUERY_PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _parse_response(self, response: dict, query: str) -> ParsedQuery:
        """Parse LLM response into ParsedQuery."""
        try:
            # Extract content from chat completion response
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            )
            data = json.loads(content)

            return ParsedQuery(
                library_name=data.get("library_name", "unknown"),
                topic=data.get("topic"),
                version=data.get("version"),
                confidence=float(data.get("confidence", 0.5)),
                alternative_libraries=data.get("alternative_libraries", []),
                raw_query=query,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
            # Fall back to simple parser on parse error
            return self._fallback.parse(query)

    def parse(self, query: str) -> ParsedQuery:
        """Parse query synchronously (uses asyncio.run internally)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.parse_async(query))
                    return future.result()
            else:
                return loop.run_until_complete(self.parse_async(query))
        except RuntimeError:
            return asyncio.run(self.parse_async(query))

    async def parse_async(self, query: str) -> ParsedQuery:
        """Parse query using LLM with structured output."""
        if not self.config.endpoint or not self.config.api_key:
            # No LLM configured, use fallback
            return self._fallback.parse(query)

        try:
            request = self._build_request(query)
            response = await self.client.post("chat/completions", request)
            return self._parse_response(response, query)
        except Exception:
            # Fall back to simple parser on any error
            return self._fallback.parse(query)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "LLMQueryParser":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


# Default parser instance
_default_parser: BaseQueryParser | None = None


def get_default_parser() -> BaseQueryParser:
    """Get the default query parser instance."""
    global _default_parser
    if _default_parser is None:
        config = LLMConfig.from_env()
        if config.endpoint and config.api_key:
            _default_parser = LLMQueryParser(config)
        else:
            _default_parser = SimpleQueryParser()
    return _default_parser


def set_default_parser(parser: BaseQueryParser) -> None:
    """Set the default query parser instance."""
    global _default_parser
    _default_parser = parser


def parse_query(query: str) -> ParsedQuery:
    """Parse a query using the default parser.

    Args:
        query: User query about a library.

    Returns:
        ParsedQuery with extracted library and topic information.
    """
    return get_default_parser().parse(query)


async def parse_query_async(query: str) -> ParsedQuery:
    """Parse a query asynchronously using the default parser.

    Args:
        query: User query about a library.

    Returns:
        ParsedQuery with extracted library and topic information.
    """
    return await get_default_parser().parse_async(query)
