"""Web search integration for Aphelion.

Provides web search capabilities using DuckDuckGo (free, no API key required).
Results are formatted for LLM context injection.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str


@dataclass
class SearchResponse:
    """Response from a web search."""

    query: str
    results: list[SearchResult]
    error: str | None = None

    def to_context(self, max_results: int = 5) -> str:
        """Format search results for LLM context injection.

        Args:
            max_results: Maximum number of results to include.

        Returns:
            Formatted string for LLM context.
        """
        if self.error:
            return f"[Web search failed: {self.error}]"

        if not self.results:
            return f"[No results found for: {self.query}]"

        lines = [f"**Web Search Results for: {self.query}**\n"]

        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"{i}. **{result.title}**")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   {result.snippet}\n")

        return "\n".join(lines)


class WebSearchTool:
    """Web search tool using DuckDuckGo.

    DuckDuckGo is free and doesn't require an API key.
    """

    def __init__(self, max_results: int = 5) -> None:
        """Initialize the search tool.

        Args:
            max_results: Default maximum results per search.
        """
        self.max_results = max_results
        self._ddgs = None

    def _get_ddgs(self):
        """Lazy load DuckDuckGo Search."""
        if self._ddgs is None:
            try:
                from ddgs import DDGS

                self._ddgs = DDGS()
            except ImportError:
                try:
                    # Fallback to old package name
                    from duckduckgo_search import DDGS
                    self._ddgs = DDGS()
                except ImportError:
                    logger.warning("ddgs not installed")
                    return None
        return self._ddgs

    def search(self, query: str, max_results: int | None = None) -> SearchResponse:
        """Perform a web search.

        Args:
            query: The search query.
            max_results: Maximum results to return.

        Returns:
            SearchResponse with results.
        """
        max_results = max_results or self.max_results

        ddgs = self._get_ddgs()
        if ddgs is None:
            return SearchResponse(
                query=query,
                results=[],
                error="DuckDuckGo search not available. Install: pip install duckduckgo-search",
            )

        try:
            logger.info("Performing web search", query=query)

            raw_results = list(ddgs.text(query, max_results=max_results))

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                )
                for r in raw_results
            ]

            logger.info("Search completed", query=query, result_count=len(results))

            return SearchResponse(query=query, results=results)

        except Exception as e:
            logger.exception("Web search failed", query=query)
            return SearchResponse(query=query, results=[], error=str(e))

    async def search_async(
        self, query: str, max_results: int | None = None
    ) -> SearchResponse:
        """Async wrapper for search (runs sync in thread pool).

        Args:
            query: The search query.
            max_results: Maximum results to return.

        Returns:
            SearchResponse with results.
        """
        import asyncio

        # Use asyncio.to_thread for reliable execution in thread pool
        return await asyncio.to_thread(self.search, query, max_results)


# Global instance
web_search = WebSearchTool()


def search_web(query: str, max_results: int = 5) -> str:
    """Convenience function to search the web and get formatted results.

    This is designed for easy LLM integration - just call this function
    and append the result to the conversation context.

    Args:
        query: The search query.
        max_results: Maximum results to return.

    Returns:
        Formatted search results string.
    """
    response = web_search.search(query, max_results)
    return response.to_context(max_results)


async def search_web_async(query: str, max_results: int = 5) -> str:
    """Async version of search_web.

    Args:
        query: The search query.
        max_results: Maximum results to return.

    Returns:
        Formatted search results string.
    """
    response = await web_search.search_async(query, max_results)
    return response.to_context(max_results)
