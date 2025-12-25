"""Shared async HTTP client for external services."""

from __future__ import annotations

from typing import Any


class HttpClient:
    """Async HTTP client wrapper for external API calls."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """Initialize HTTP client.

        Args:
            base_url: Base URL for API requests.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts for failed requests.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-initialize httpx client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for HTTP backends. "
                    "Install with: pip install context7-reranker[http]"
                ) from e

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def post(self, path: str, json: dict) -> dict:
        """Make POST request to endpoint.

        Args:
            path: URL path (appended to base_url).
            json: JSON body to send.

        Returns:
            Response JSON as dict.

        Raises:
            httpx.HTTPStatusError: On non-2xx response.
        """
        if path:
            url = f"{self.base_url}/{path.lstrip('/')}"
        else:
            url = self.base_url
        response = await self.client.post(url, json=json)
        response.raise_for_status()
        return response.json()

    async def post_with_retry(self, path: str, json: dict) -> dict | None:
        """Make POST request with retry logic.

        Args:
            path: URL path.
            json: JSON body.

        Returns:
            Response JSON or None if all retries failed.
        """
        try:
            import httpx
        except ImportError:
            return None

        for attempt in range(self.max_retries):
            try:
                return await self.post(path, json)
            except httpx.TimeoutException:
                if attempt == self.max_retries - 1:
                    return None
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    if attempt == self.max_retries - 1:
                        return None
                    continue
                return None
            except Exception:
                return None
        return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "HttpClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
