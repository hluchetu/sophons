from __future__ import annotations

from sophons.tools.base import ToolResult
from sophons.tools.decorator import FunctionTool, build_args_schema


def _format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"**{title}**\n{url}\n{content}")
    return "\n\n".join(lines)


def tavily_web_search(api_key: str, max_results: int = 5) -> FunctionTool:
    """Create a Tavily web search tool with the provided API key."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)

    def web_search(query: str) -> ToolResult:
        """Search the web for current information."""
        if not query:
            raise ValueError("'query' argument is required.")
        response = client.search(query=query, search_depth="basic", max_results=max_results)
        results = response.get("results", [])
        return {
            "query": query,
            "result_count": len(results),
            "content": _format_results(results) if results else f"No results found for: {query}",
            "results": results,
        }

    return FunctionTool(
        name="web_search",
        description="Search the web for current information.",
        args_schema=build_args_schema(web_search),
        fn=web_search,
    )
