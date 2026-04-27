from __future__ import annotations

from radar.collector import parse_markdown_section_items


def test_parse_markdown_section_items_filters_target_section() -> None:
    markdown = """
### 🏦 Finance & Tax

**[KIS_MCP_Server](https://github.com/migusdn/KIS_MCP_Server)** – 한국투자증권 주문 기능을 제공하는 MCP 서버입니다.

### 🏠 Real Estate

**[real-estate-mcp](https://github.com/tae0y/real-estate-mcp)** – 청약 정보를 제공하는 MCP 서버입니다.
"""

    items = parse_markdown_section_items(markdown, "Finance & Tax")

    assert items == [
        {
            "title": "KIS_MCP_Server",
            "link": "https://github.com/migusdn/KIS_MCP_Server",
            "summary": "한국투자증권 주문 기능을 제공하는 MCP 서버입니다.",
        }
    ]
