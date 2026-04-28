from __future__ import annotations

from datetime import UTC, datetime

from radar.collector import _collect_single
from radar.models import Source


class _RegistryResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _RegistrySession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: int, headers: dict[str, str]) -> _RegistryResponse:
        self.urls.append(url)
        return _RegistryResponse(self.payload)


def test_mcp_registry_search_collects_github_repository_entries() -> None:
    payload = {
        "servers": [
            {
                "server": {
                    "name": "io.github.example/korean-news-mcp",
                    "title": "Korean News Hub",
                    "description": "Korean news aggregator for Naver and Daum trends",
                    "repository": {
                        "url": "https://github.com/example/korean-news-mcp",
                        "source": "github",
                    },
                },
                "_meta": {
                    "io.modelcontextprotocol.registry/official": {
                        "updatedAt": "2026-03-08T23:54:49.6001Z"
                    }
                },
            },
            {
                "server": {
                    "name": "com.example/no-repo",
                    "description": "A registry item without GitHub repository metadata",
                }
            },
            {
                "server": {
                    "name": "io.github.example/korean-news-mcp-older",
                    "description": "Duplicate repository from an older version",
                    "repository": {
                        "url": "https://github.com/example/korean-news-mcp",
                        "source": "github",
                    },
                }
            },
        ]
    }
    session = _RegistrySession(payload)
    source = Source(
        name="Official MCP Registry Korea search",
        type="mcp_registry_search",
        url="https://registry.modelcontextprotocol.io/v0.1/servers",
        config={"search_terms": ["korean"], "query_limit": 5},
    )

    articles = _collect_single(
        source,
        category="misc_mcp",
        limit=10,
        timeout=15,
        session=session,  # type: ignore[arg-type]
    )

    assert len(articles) == 1
    assert articles[0].title == "Korean News Hub"
    assert articles[0].link == "https://github.com/example/korean-news-mcp"
    assert articles[0].source == source.name
    assert articles[0].category == "misc_mcp"
    assert articles[0].published == datetime(
        2026, 3, 8, 23, 54, 49, 600100, tzinfo=UTC
    )
    assert session.urls == [
        "https://registry.modelcontextprotocol.io/v0.1/servers?search=korean&limit=5"
    ]


def test_mcp_registry_search_uses_category_as_default_query() -> None:
    session = _RegistrySession({"servers": []})
    source = Source(
        name="Official MCP Registry",
        type="mcp_registry_search",
        url="https://registry.modelcontextprotocol.io/v0.1/servers",
        config={},
    )

    articles = _collect_single(
        source,
        category="misc_mcp",
        limit=10,
        timeout=15,
        session=session,  # type: ignore[arg-type]
    )

    assert articles == []
    assert session.urls == [
        "https://registry.modelcontextprotocol.io/v0.1/servers?search=misc_mcp&limit=10"
    ]
