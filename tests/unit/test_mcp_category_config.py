from __future__ import annotations

from pathlib import Path

from radar.analyzer import apply_entity_rules
from radar.collector import parse_markdown_section_items
from radar.config_loader import load_category_config
from radar.models import Article


def _category_name() -> str:
    configs = sorted(Path("config/categories").glob("*.yaml"))
    assert len(configs) == 1
    return configs[0].stem


def test_mcp_category_config_uses_readme_section_source() -> None:
    category = load_category_config(_category_name())

    assert len(category.sources) == 1
    source = category.sources[0]
    assert source.type == "github_readme_section"
    assert source.url == "https://raw.githubusercontent.com/darjeeling/awesome-mcp-korea/main/README.md"
    assert source.section
    assert {entity.name for entity in category.entities} >= {
        "MCPDomain",
        "Provider",
        "Capability",
        "RiskScope",
        "ProjectHealth",
    }


def test_mcp_category_config_matches_section_entries() -> None:
    category = load_category_config(_category_name())
    section = category.sources[0].section
    markdown = f"""
### {section}

**[example-mcp](https://github.com/example/example-mcp)** - {section} MCP server with API search tools.

### Other Section

**[other-mcp](https://github.com/example/other-mcp)** - Another MCP server.
"""

    items = parse_markdown_section_items(markdown, section)
    assert len(items) == 1

    article = Article(
        title=items[0]["title"],
        link=items[0]["link"],
        summary=items[0]["summary"],
        source=category.sources[0].name,
        category=category.category_name,
    )
    analyzed = apply_entity_rules([article], category.entities)

    assert analyzed[0].matched_entities
    assert "MCPDomain" in analyzed[0].matched_entities
    assert "ProjectHealth" in analyzed[0].matched_entities
